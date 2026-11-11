# Distributed: communication, stragglers, accumulation

Profile **every rank**: each writes its own trace (`worker_name=f"rank{rank}"`). One rank's trace can't tell its own
slowness apart from waiting on another rank. `trace_report.py traces/` prints a row per rank.

## 1. Exposed vs overlapped communication

DDP launches its all-reduces on a separate stream during backward, so they overlap with the remaining backward
kernels. Communication costs step time only where no compute runs alongside it.

- **Exposed = collective time not covered by compute on any other stream.** The report's comm section gives the
  collective kernel time, the exposed time, and the exposed share of the GPU window.
- **A collective kernel's duration is not its cost.** NCCL kernels start when launched and spin until every rank
  arrives, so a late peer inflates them. In one 3-step window, identical work took 49.6, 116.1 and 20.9 ms of NCCL
  kernel time per step (4×H100). That spread was rank skew, not communication.
- **"GPU busy" counts that spin.** A run-level GPU-idle metric read 1.0% on a run with 80.6% of its window in
  exposed communication (2×2 A100). The report prints compute-stream busy beside total busy for this reason.

Measured:
- **4×H100, one node, NVLink:** 96.4-96.9% of communication overlapped. 2.0 of 63.5 ms was exposed in a 3-step
  window.
- **2×2 A100, two nodes:** communication filled 94.3% of the window, and 80.6% of the window was exposed: ~2.6 s of
  all-reduce per micro-batch, against ~27 ms on one 4-GPU node.
- **2×L40S, `DDP(torch.compile(model))`:** 100% of the all-reduce was exposed. *(Wrap-order result pending.)*

## 2. Stragglers

A step waits for the slowest rank. A late rank spends less time in collectives than the others, because they spin
waiting for it while it arrives last.

- **The report's rank table** flags the rank with the most DataLoader wait and the rank with the least
  collective-kernel time. When both flags point at one rank, that rank is the straggler.
  - **Test on 2×L40S:** a 0.5 s DataLoader stall planted on rank 1 got both flags. Rank 0's trace showed the same
    stall only as a 526 ms wait in its next sync, with the GPU "busy" on a spinning NCCL kernel.
  - **4×H100 stall capture:** one rank was flagged in a 4-rank window that held a 4.2 s DataLoader stall.
- **A late batch on one rank shows up on the others as all-reduce time,** not as data wait. Check every rank before
  blaming communication.
- **Skew moves between ranks and between jobs.** Two ranks of one job differed by up to 5× in sync cost, and which
  rank was high flipped between jobs (2×A100). Rank-0 means produced two false features in a bucket-size sweep.
  - Take the per-step **minimum across ranks**, which removes waiting on peers, then the **median over steps**, which
    removes rare stalls.

## 3. Gradient accumulation

On accumulation micro-batches, DDP's `no_sync()` should skip the all-reduce. The report's "collectives per step"
table checks it: with `prof.step()` per micro-batch, accumulation steps should show zero all-reduces.

- **Measured:** every micro-batch all-reduced, 34 collectives carrying 1.24 GB each time (4×H100). Accelerate's
  `unwrap_model()`, at its default `keep_torch_compile=True`, had rewritten the compiled model's `_orig_mod`, which
  hid the DDP wrapper from `getattr(model, "no_sync")`. `no_sync()` silently did nothing.
  - **Fix:** `unwrap_model(..., keep_torch_compile=False)`.
  - **One node:** no throughput change, because the communication was hidden anyway.
  - **2×2 A100:** MFU went from 11.1% to 51.8%.
- **A window can miss the optimizer step.** At 8 micro-batches per optimizer step, a 3-step window landed on
  accumulation micro-batches only and showed zero collectives. The report warns about this: "world_size N but no
  collective kernels". Step the profiler per optimizer step, or make `active` cover a full cycle.

## 4. The network under multi-node communication

Before tuning DDP, check that the transport is what you think it is.
- Run once with `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH` and read the init lines. `GPU Direct RDMA
  Disabled for GPU N / HCA M (distance 9 > 5)` and `GDR 0` mean every byte is staged through host memory.
- **Measured** (2 nodes × 1 A100, GPU Direct RDMA refused by default because the GPU-NIC path crossed sockets):
  - the collectives ran back to back on the comm stream with no overlap: 772 ms per step, 20.3% of it;
  - `NCCL_NET_GDR_LEVEL=SYS` turned GPU Direct RDMA on: 140.5 ms per step (4.2%), +14.5% tokens/s.
- **Bandwidth from the trace:**
  - NCCL kernel events carry the message size (`args`: `"In msg nelems"`, `"dtype"`).
  - Fitting duration = t₀ + size / w over the collectives gave w = 1.64 GB/s with GPU Direct RDMA off and 8.67 GB/s
    with it on.
  - Trace bandwidth includes waiting on peers, so read it as a lower bound. On 2×2 A100 the trace gave 1.05 GB/s,
    against 1.57 GB/s from a standalone 512 MB all-reduce probe.

## 5. DDP knobs

- On one node with ~96% overlap, `bucket_cap_mb` and `gradient_as_bucket_view` were flat (4×H100, 2×A100). Exposed
  communication was too small to matter.
- **Only the profiler shows what `bucket_cap_mb` changes:** the number of all-reduces per backward. That was 35 at
  25 MB, 13 at 100 MB and 4 at 400 MB (2×A100). Their cost is read from the exposed time, not the kernel time.
- **Broadcast of buffers** (`broadcast_buffers=True`): 512 KB per micro-batch, about 0.02% of the step (4×H100).
  Not worth turning off for speed.
