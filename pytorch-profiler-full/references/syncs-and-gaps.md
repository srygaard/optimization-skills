# Syncs, idle gaps and host overhead

## 1. What a sync costs

The host normally runs ahead of the GPU, filling a queue of launched kernels. A sync blocks the host until the GPU
has finished the work it waits on.

- **During the wait, the GPU is usually busy** running the queue. The wait is the queue draining, not lost time.
- **The cost is what comes after.** If the sync emptied the queue, the GPU idles from its return until the host
  launches again, and the host can't run ahead to hide its next stall.
- **A sync is where queued work surfaces, not its cause.** A long wait means the queued work was long.

`trace_report.py` reports each call site with the GPU-busy fraction during the wait and the GPU idle after it:

| kind | GPU during the wait | GPU after it returns | cost |
|---|---|---|---|
| drains queue | busy | idle until the next launch | the idle after, plus no run-ahead |
| queue stays full | busy | still busy (it waited on older work) | none: backpressure |
| GPU already idle | idle | idle | the gap was already there; look before it |

Measured:
- **Inductor's own `cudaEventSynchronize`,** inside `Torch-Compiled Region: 0/0`:
  - L40S: 8 per step, and 38-43% of the step's wall time sat inside `forward` waiting on it, with the GPU 99.8-100%
    busy. Backpressure, harmless.
  - 4×H100: the report showed 24 calls, a 121.9 ms wait, the GPU 98.1% busy during it, and 0.01 ms idle after:
    "queue stays full".
- **A blocking DataLoader copy** (Accelerate's `non_blocking` defaults to `False`): a 72-73 ms
  `cudaStreamSynchronize` at the start of every `next()`, with 961 GPU events still queued when it fired (4×H100).
  - It drains the queue, but the idle after it totalled 4.19 ms over 3 steps.
  - Fixing it (`DataLoaderConfiguration(non_blocking=True)`) moved trace idle 13.4% → 12.5%, train 1.01×, eval 1.05×.
- **An `.item()` for a progress bar** waited 73.9 ms with the GPU 98.1% busy throughout (4×H100). It only appeared
  once a bigger drain was removed, and cost almost nothing.
- **An `.item()` that waited 51.6 s and 22.9 s** (MI300A, eager) was the queue draining behind a GEMM running at
  ~0.02% of peak. The cause was a BLAS library regression, found with the library's own logging. Moving or removing
  the `.item()` would have hidden it.
- **A mean over syncs misleads.** In the L40S case, 6 of every 8 syncs were ~6 µs, so the mean over 24 suggested
  ~25 ms each. The report gives total, max and count.

## 2. Where syncs come from

| code | in the trace |
|---|---|
| `.item()`, `float(t)`, `.tolist()`, `print(t)`, `if t:` | `aten::item` → `aten::_local_scalar_dense` → `cudaMemcpyAsync` + `cudaStreamSynchronize` |
| `.cpu()`, `.numpy()`, `.to("cpu")` | `aten::copy_` → `Memcpy DtoH` + `cudaStreamSynchronize` |
| `torch.tensor(x, device="cuda")` | `aten::copy_` → `Memcpy HtoD (Pageable -> Device)` + `cudaStreamSynchronize` |
| `.to(device)` without `non_blocking=True` | `aten::copy_` → `Memcpy HtoD` + `cudaStreamSynchronize` |
| `nonzero`, boolean-mask indexing, `masked_select`, `unique` | `cudaStreamSynchronize` under the op: the output size has to reach the host |
| `torch.cuda.synchronize()` | `cudaDeviceSynchronize` |
| `event.synchronize()`, Inductor bookkeeping | `cudaEventSynchronize` |

The pageable-scalar case is easy to write by accident. `torch.tensor(0.0, device=device)` issued a 4-byte copy
followed by a 63-73 ms `cudaStreamSynchronize` every step (4×H100). It still syncs in torch 2.15, verified on 2×L40S.
- **Fixes:** `torch.zeros((), device=device)`, or `torch.full((), n, device=device, dtype=...)` for a value.
- **Same pattern elsewhere:** a token counter built with `torch.tensor(float(tokens), device=...)` cost a 3.1 ms
  sync per step.

## 3. Finding the line

- **The report's call site** names the enclosing `record_function` label and the aten op, e.g.
  `cudaStreamSynchronize | grad_norm | aten::copy_`. Narrow the labels until one statement is left.
- **`torch.cuda.set_sync_debug_mode("warn")`** warns with a traceback at each syncing call (`"error"` raises). It
  misses `torch.cuda.synchronize()` and non-blocking copies from pageable memory.
- **`with_stack=True` names the function, not the line.**
  - A sync in the training loop's body is reported under the loop's function, with the line where that frame stood
    when tracing started (the `prof.step()` line).
  - Factor the suspect code into a function, or use the two tools above.

## 4. Fixing syncs

- Keep metrics on the device. Accumulate losses into a device tensor, and `.item()` only at logging steps.
- Make device scalars with `torch.zeros`/`torch.full`, never `torch.tensor(python_value, device=...)`.
- Copy batches with `pin_memory=True` and `.to(device, non_blocking=True)`. Accelerate needs
  `DataLoaderConfiguration(non_blocking=True)`. The `pytorch-dataloader` skill has the details.
- Take data-dependent shapes (`nonzero`, masks) out of the step. Compute them once per step instead of once per
  layer, or in the DataLoader. Pad to fixed sizes.
- **After each fix, profile again.** A drain hides any smaller one behind it, and the `.item()` above only showed
  once the bigger drain was gone.

## 5. Idle gaps

The histogram tells discrete stalls from overhead:

| window (4×H100, 354M transformer) | idle | made of |
|---|---|---|
| cold, micro-batches 2-4 | 24.5% | 3 gaps over 10 ms (58% of idle), 19 of 1-10 ms, and 3,967 gaps under 100 µs adding up to only 10.3 ms |
| steady, micro-batches 61-63 | 1.8% | all 6,821 gaps under 10 µs |

- **A few big gaps:** look at what the host was doing during each one (the report prints it).
- **Thousands of tiny gaps adding up:** launch overhead (§7).
- **"Idle that began as a sync returned"** is a drain (§1). **"Idle while the host waited in DataLoader
  `__next__`"** means the run is input-bound: see the `pytorch-dataloader` skill.

## 6. Host work that leaves the GPU idle

The report's host-ops table lists long ops that launched almost nothing, with the GPU idle time while each ran. Rows
without idle time cost nothing.

**GEMM library heuristics at new shapes.** The first time it sees a shape, cuBLAS picks a kernel on the host, and
that shows up as long `aten::mm` host ops launching 1-2 kernels. Same job, 4×H100:

| | cold window | steady window |
|---|---|---|
| slow ops (≥ 200 µs, ≤ 8 calls) | 24 | 63 |
| their host time | 20.2 ms | 26.3 ms |
| GPU idle meanwhile | 14.7 ms (72.8%) | 0.58 ms (2.2%) |
| share of the window | 2.42% | 0.09% |

It was warmup: rounding the token budget to fewer shapes would have cost ~0.8% in padding to save 0.09%.

**Python dispatch in a custom autograd function.** One gap of 12.7-15.7 ms sat inside `ApplyRotaryEmbQKV_Backward`,
which issued no sync, just a few pointer queries and one launch (4×H100, cold window). Probably Python-side dispatch.
Not confirmed, and moot once the op was replaced by a native one.

**Copy overlap rarely pays.** A side-stream prefetcher hid 81% of the H2D copy, but the copy was only ~65 µs of a
~717 ms step. A/B: 1.00× (4×L40S). Compare batch bytes ÷ H2D bandwidth with the step time before building one.

## 7. Launch-bound steps

**Signature:**
- compute-stream busy well under 100%;
- thousands of 10-100 µs gaps;
- kernels averaging a few µs;
- each host op launching one kernel.

Measured:
- **A per-layer packing path with `@torch.compiler.disable` around attention** (MI300A):
  - Kernel time was ~38 ms per step, level with the fused path's ~37 ms, yet the step took 236 ms against 63 ms.
  - `nonzero` and `.item()` synced 14 times per step.
  - Moving the boundary so that only the attention kernels stayed outside the graph took it from 236 to 65 ms per
    step.
- **A loop of 300 tiny elementwise ops per step** (2×L40S toy): its label showed 21.2 ms of host time over 3 steps
  against 1.8 ms of GPU time from 600 launches per step. The window had 1,473 gaps of 10-100 µs.

**Levers:**
- `torch.compile`, which fuses elementwise chains into fewer kernels;
- CUDA graphs (`mode="reduce-overhead"`), which replay a whole step with one launch but need fixed shapes. On
  variable-shape batches, both `reduce-overhead` and `max-autotune` crashed;
- larger batches, which give each launch more work;
- moving small bookkeeping ops off the step.

## 8. Stalls the profiler can't see

When the GPU idles and no traced host event covers the gap, or one cheap op spans tens of ms with nothing under it,
the host thread was blocked by something the profiler doesn't record: the GIL held by another thread, garbage
collection, an allocator lock, a page fault, or CPU contention.

- **4×H100:** `optimizer_step` spans of 27.7 and 47.3 ms contained **zero CUDA calls and zero ops**. They happened
  on accumulation micro-batches in a cold window and never reappeared at steady state. GIL or allocator contention
  with the autograd thread was suspected, not verified.
- **2×L40S, torch 2.15 toy:** an `aten::add` that launched one 2 µs kernel spanned 80 and 104 ms with the GPU idle,
  on both ranks within ~30 ms of each other. It happened in 1 of 4 runs of that window.
  - In the same compiled process, full (gen2) garbage collections took 160-200 ms, on both ranks at once (timed with
    `gc.callbacks`), because both ranks allocate the same way.
  - So GC is a candidate for that gap, but the timing didn't confirm it.
  - Lever: `gc.freeze()` after setup and compilation takes the long-lived objects out of future collections.

**Tools:**
- `py-spy dump --pid <pid> --native` while it is stuck;
- `faulthandler.dump_traceback_later(t, repeat=True)` as a watchdog;
- `gc.callbacks` to time collections;
- `/proc/<pid>/wchan` and worker states (`D` means blocked on I/O);
- a watchdog thread that dumps all of these when a step exceeds a threshold.
