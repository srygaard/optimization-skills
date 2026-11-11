# Host-to-device copies: pinned memory and syncs

In the source workloads, every expensive copy problem was a **sync that drained
the GPU's queue**, not bytes moving. A packed batch was ~800 KB, about 65 µs over
PCIe. Fix the syncs first; build copy/compute overlap only if the bytes justify it.

## 1. What pinning buys

- **A copy from pinned (page-locked) memory with `non_blocking=True` is queued and returns at once.** The host keeps launching work while the DMA runs.
- **A pageable source can't be DMA'd directly.**
  - With `non_blocking=True`, the driver copies it into a staging buffer, and the host waits for that copy, though not for the GPU queue.
  - With `non_blocking=False`, the copy also synchronizes the stream, so the host waits for all GPU work queued before it.
  - Measured: `torch.tensor(0.0, device="cuda")` builds a pageable host tensor and copies it. The 4-byte copy was followed by a 63-73 ms `cudaStreamSynchronize`, which waited for the whole queued forward and backward. It ran on every micro-batch.
  - Fix: allocate on the device (`torch.zeros((), device=...)`, `torch.full`), which launches a fill kernel and copies nothing.
- **`pin_memory=True` without `non_blocking=True` still blocks.** Accelerate's `non_blocking` defaults to `False` (§3, SKILL.md §5).

## 2. Hidden syncs

Anything that needs a GPU value on the host waits for all queued work:
- `.item()`, `.tolist()`, `.cpu()`, `print(t)`;
- `if t:` and other uses of tensor truthiness;
- `torch.nonzero` and boolean-mask indexing (the output size depends on the data);
- blocking copies.

One per step in the training loop puts every step's queue drain on the critical path,
and makes `next()` timing misleading ([measurement.md](measurement.md) §1).
`torch.cuda.set_sync_debug_mode("warn")` (experimental) warns on syncs that go through
PyTorch's copy-and-sync and stream-sync paths.
- **It flags:**
  - `.item()` and `.cpu()`;
  - blocking H2D copies, including `torch.tensor(x, device="cuda")`.
- **It misses:**
  - `torch.cuda.synchronize()`;
  - a `non_blocking=True` copy from pageable memory;
  - `torch.distributed` and `torch.sparse` ops.

Use it while hunting, not in production runs, and confirm with a trace.

## 3. The DataLoader's pinning

- **One pin thread in the main process** copies each finished batch into pinned memory. With `num_workers=0` there is no thread, and pinning runs inline in `next()`.
- **Custom batch types are pinned only if they define `pin_memory()`.** Otherwise they come through unpinned, with no error. Check what reaches the copy: `t.is_pinned()`.
- **`pin_memory_device` is deprecated** (since torch 2.9): don't set it. The current accelerator is used.
- **How an Accelerate-prepared loader copies** (read from the v1.15.0 source, checked 2026-11-10):
  - `DataLoaderShard` and `DataLoaderDispatcher` call `send_to_device(batch, device, non_blocking=...)` on the host thread, inside `next()`. There is no side stream and no device prefetch.
  - `non_blocking` comes from `DataLoaderConfiguration(non_blocking=...)`, default `False`. The docs recommend `pin_memory=True` with it.
  - It looks one batch ahead on the host. Each `next()` copies batch N, then pulls N+1 from the base loader before yielding N. A `TimedIterator` around it therefore times N's copy plus the wait for N+1, and one extra batch is held, pinned.
  - In ≤ 1.15.0, `skip_first_batches` rebuilds the loader without `_non_blocking` (or `drop_last`, `slice_fn`, the device mesh), so a resumed epoch copies blocking. Check `loader._non_blocking`. Fixed on main by PR #4296, unreleased as of 2026-11.
- **Pinning yourself:** `.pin_memory()` is itself a host copy into page-locked memory. `t.pin_memory().to(dev, non_blocking=True)` on a fresh tensor does two copies on the critical path. It pays only when the pinning happens off that path (in the pin thread), or into a buffer you reuse.

## 4. Correctness

- **Device-to-host with `non_blocking=True` returns before the data arrives.** Record an event and wait on it (or synchronize) before the host reads the result.
- **Don't modify or reuse a pinned source buffer** until the copy reading it has completed (an event on the copy's stream). The copy reads the buffer later, when it runs.

## 5. Host memory cost

- **Pinned pages are locked:** the kernel can't reclaim or swap them.
- **Expect about one pinned copy per finished, unconsumed batch.**
  - The loader caps in-flight batches at `num_workers × prefetch_factor`. That cap includes batches waiting for the pin thread and batches already pinned.
  - Batches the training loop still holds come on top of it. A deep prefetch queue (SKILL.md §3) multiplies pinned memory with it. Check that batch bytes × depth fits.
- **Count both loaders.** A paused train iterator keeps its full pinned queue through every eval pass. A persistent eval loader keeps its worker processes through training, but no batches once its pass has ended.
- **PyTorch's caching host allocator keeps freed pinned blocks for reuse,** so the footprint stays at its peak. The `pinned_*` options of `PYTORCH_ALLOC_CONF` (registration threads, cache caps) vary by version; read the CUDA notes for your torch before tuning them.
- **APUs with unified memory (MI300A):** host and device draw on one memory pool. A host OOM kill and a GPU OOM came from the same budget; size them together.

## 6. Copy/compute overlap on a side stream

- **Check the bytes first:** batch bytes ÷ H2D bandwidth, against the step time. Measured: a side-stream prefetcher hid 81% of the copy and gained 1.00× train, 0.99× eval, because the copy was ~65 µs of a ~717 ms step.
- **On the default stream, a `non_blocking` copy doesn't overlap compute.** It runs in stream order, ahead of the step's kernels, but costs the host nothing. A side stream helps only when that copy time matters.
- **If you build one:**
  - Create the stream once, for the trainer's lifetime. A stream created per epoch and per eval pass was destroyed while tensors allocated on it were still live in the autograd graph.
  - Call `current_stream().wait_stream(copy_stream)` before using the batch, and `t.record_stream(current_stream())` on each tensor, so the allocator doesn't reuse the memory while the compute stream still reads it.
  - With Accelerate, prepare the loader with `device_placement=False`, or it copies on the default stream first.
  - Test every attention backend. Side-stream tensors reaching one flash-attention backward segfaulted at the first step; the suspect was `cu_seqlens`, which is saved for backward.

## 7. Reading a trace

| you see | it means |
|---|---|
| `Memcpy HtoD (Pageable -> Device)` | the source wasn't pinned; look for a sync right after it |
| `cudaStreamSynchronize` inside `next()` or a device-placement wrapper | a blocking batch copy (72 ms/step measured, all of it queue drain) |
| a long `cudaEventSynchronize` while the GPU is busy | backpressure: the host waiting on a saturated GPU. Healthy; not a stall |
| `next()` wait grows after you remove a sync | the host now runs ahead and waits on the loader, but the wait overlaps queued GPU work. Measured: 3.9 → 160.9 ms/step wait with device time unchanged (455 vs 450 ms) |

Judge a sync by GPU occupancy inside its window, and judge throughput by device time
from CUDA events, not by host-side spans ([measurement.md](measurement.md) §1).
