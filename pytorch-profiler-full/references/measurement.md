# Measurement rules

Each rule comes from a measurement that was wrong first.

### 1. A trace diagnoses; an unprofiled A/B decides

- **Measured:** a host-sync fix cut trace idle from 24.5% to 10.9%, then gained 1.01× tokens/s in a 128-step
  unprofiled A/B (4×H100). The trace was 3 micro-batches of compile warmup. At steady state, throughput was gated
  by GPU compute, not by the stall the fix removed.
- **Profiler overhead is not why:** profiled and unprofiled throughput differed by 0.4%.
- **So:** report the win from tokens/s (or step time) of runs without the profiler, on the same node, back to back.

### 2. Profile steady state

The cold window above showed 12.7% idle and the steady window 1.8%, same job. See [capture.md](capture.md) §1. A
trace whose step spans don't match the unprofiled step time is a trace of something else.

### 3. Only kernels, memcpy and memset are GPU work

`gpu_user_annotation` spans sit on the GPU timeline but are labels. Counting them turned 75.5% busy / 24.5% idle
into "96.9% busy" and hid every gap over 10 ms (4×H100).

### 4. A host span is not a cost

A `record_function` span measures how long the host was inside it, including any time blocked behind GPU work.
- **Measured:** 38-43% of an L40S step's wall time sat inside `forward`, waiting on Inductor's
  `cudaEventSynchronize`, while the GPU was 99.8-100% busy.
- **For cost by region,** use the GPU time of the kernels each region launched (the report's Regions section), or
  CUDA events recorded on the stream:

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
loss = model(batch)
end.record()
# read once per logging interval, after a single sync, so timing adds no sync of its own
end.synchronize()
forward_ms = start.elapsed_time(end)
```

### 5. A sync's wait is not its cost

The cost is the GPU idle after it returns: see [syncs-and-gaps.md](syncs-and-gaps.md) §1. A 51.6 s `.item()` was a
BLAS regression surfacing, not a slow `.item()`.

### 6. Read distributions, not means

- **Over events:** 6 of every 8 syncs were ~6 µs, so a mean over 24 implied ~25 ms each (L40S).
- **Over steps** (MI300A, 60M transformer):
  - a control run's mean step was 138.0 ms against a 62.1 ms baseline, while its median was 39.5 ms against 35.9 ms
    and its kernel time was lower;
  - 12% of its steps took 300-1400 ms, and 97% of the steps over 200 ms coincided with a DataLoader stall recurring
    every 8 steps, with 8 workers.
  - Compare median step time and kernel time. A DataLoader stall can't reach those.

### 7. Re-profile after every fix

A drain hides any smaller one behind it, because the queue it empties is the one the next stall would have
drained. A 73.9 ms `.item()` appeared only after a 72 ms drain before it was fixed (4×H100). After a fix, read the
syncs and host-ops tables again before calling the hunt finished.

### 8. One rank and a few steps are a sample

- Three micro-batches on rank 0 can't carry a claim about a 128-step run on 4 ranks.
- Ranks of one job differed by up to 5× in sync cost, and which rank was high flipped between jobs (2×A100).
- **So:** profile every rank, and for per-rank costs take the per-step minimum across ranks, then the median over
  steps.

### 9. Pin the node

- **Run-to-run noise:** ~1-2% on one node, ~4% across nodes, in the multi-GPU runs behind this skill. Mixing two node
  pools skewed comparisons by up to 1.66×.
- **So:** run A/B arms back to back on one node, and check `sacct -j <job> --format=NodeList` before trusting a delta.

### 10. Don't compare kernel times across points in a run

Warmup-window kernels ran ~17% faster than steady-state ones in the same job (4×H100: GEMMs at ~89% vs ~77% of
bf16 peak). Most likely higher clocks before power and thermal steady state; not verified. Compare windows taken at
the same point of the run.

### 11. Normalize per unit of work

When batch fill or tokens per step change, per-step kernel time can look flat while per-token time moves. One
packing fix raised fill from 81.8% to 99.5% and confounded every comparison across it. Divide by tokens (or
samples) in the window.

### 12. Don't derive a metric by subtracting host times

Host step time minus DataLoader wait was once reported as a 1.56× "compute-only" speedup and later retracted:
the host's waits overlap queued GPU work, so host intervals don't subtract. Measure device time with CUDA events
(§4) or read it from the trace.

### 13. Confirm the configuration you profiled

- Log a line on the path under test, and read knobs from the resolved config.
- A resolved config dump didn't record whether the profiler was on.
- A label that doesn't appear in the trace means the code under it didn't run, or it sat inside a compiled function,
  where Dynamo drops labels by default and Inductor always does ([capture.md](capture.md) §2).
