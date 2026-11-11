# Capturing a trace

API facts are from the torch 2.15 docs and source (checked 2026-11). Numbers are in-house measurements, with the
setup each came from.

## 1. Choose the window: steady state only

A trace of warmup describes warmup. Before profiling, run the job **without** the profiler, log the wall time of
every step, and find the step after which the times stay within run-to-run noise. That step count is `skip_first`.

What keeps early steps slow:

| cause | seen in a trace as | measured |
|---|---|---|
| torch.compile, first call | one huge step | 60-100 s (4×H100, 354M transformer); 110 s (MI300A, 60M transformer) |
| a new input shape: recompile | a huge step at each new shape | 54-58 s per shape (`default`), 330-384 s (`max-autotune-no-cudagraphs`), MI300A |
| a new shape after compile, no recompile | a slower step, no log line | ~350-450 ms extra on first appearance, MI300A |
| GEMM library heuristics at new shapes | long `aten::mm` host ops launching 1-2 kernels while the GPU idles | new (kernel, grid) signatures stopped arriving after ~30% of a 3-step cold window (4×H100); 7.4-13.4 s steps in eager on ROCm until every shape was seen |
| caching allocator growth | `cudaMalloc` calls in the window (the report warns) | 23 in a cold window, 0 in a steady one (4×H100) |
| GPU clocks | kernels *faster* than later | ~17% faster in the warmup window (4×H100; cause not verified) |

Cold vs steady, same job (4×H100, 354M transformer, `torch.compile`, profiled 3 micro-batches each):

| | micro-batches 2-4 | micro-batches 61-63 |
|---|---|---|
| GPU idle | 12.7% | 1.8% |
| gaps of 1-10 ms | 10, 16.5 ms | 0 |
| gaps of 100 µs-1 ms | 151, 45.8 ms | 0 |
| largest gap | 3.34 ms | 7 µs |

Every host stall chased in the cold window was gone at steady state; fixing them gained 1.01× tokens/s.

- **Varying shapes:** profile after every shape has compiled, or at the most common shape. In one run 84% of steps
  had one sequence length; the profiled steps happened to be that one. Log the shapes of the profiled steps
  (`record_shapes=True`, or print them).
- **Check the window after the fact:** the profiled step spans should match the unprofiled steady step time.
  `trace_report.py` warns when step spans vary, when `cudaMalloc` or compile events fall inside the window.
- **A short run writes no trace** if it ends before `skip_first + wait + warmup + active` steps. `StepProfiler`
  ([SKILL.md §2](../SKILL.md)) warns at exit when that happens.

## 2. The snippet

The `StepProfiler` class and training loop are in [SKILL.md §2](../SKILL.md). Run with
`MYPROJ_PROFILE=skip_first,warmup,active` (prefixed with your project's name), e.g. `MYPROJ_PROFILE=50,1,3`. Unset,
the profiler is a `nullcontext`; an empty or malformed value raises. `wait=0` and `repeat=1` are fixed: one window,
then exit.

### Schedule semantics (`torch.profiler.schedule`)

Steps count from 0, one per `prof.step()` call.

- `skip_first`: steps ignored before the first cycle.
- `wait`: idle steps at the start of each cycle.
- `warmup`: steps traced but discarded. The docs: "used to discard the samples obtained by the profiler at the
  beginning of the trace since they are usually skewed by an extra overhead". `warmup=0` triggers the warning
  "Profiler won't be using warmup, this can skew profiler results".
- `active`: steps recorded. The trace is handed to `on_trace_ready` when the last one ends.
- `repeat`: number of cycles. `0` cycles until profiling stops (used for rare stalls, §4).
- `skip_first_wait`: when set, the first cycle skips its `wait`. Only useful with `wait` > 0 and repeated cycles.

The documented example, for reference:

```python
schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
on_trace_ready=trace_handler,
```

### Output

- `tensorboard_trace_handler(dir, worker_name, use_gzip=True)` writes `{worker_name}.{time_ns}.pt.trace.json.gz`.
  The worker name defaults to `{hostname}_{pid}`; `rank{N}` makes the files easy to match to ranks. Use a fresh
  directory per run: an old rank file left in it gets read as one more rank.
- `prof.export_chrome_trace(path)` gzips when `path` ends in `.gz`. If the schedule ran several cycles, only the
  last cycle is exported.
- **`on_trace_ready` runs inside the `prof.step()` that closes the window,** so that step also pays for building and
  writing the trace. Measured: the step that closed the window took 1160 ms against ~50 ms neighbours (MI300A, 60M
  transformer). If the job keeps training, leave that step out of timing; better, exit (below).

### Exit once captured

When the trace is the goal, stop:
- every rank breaks at the same step count, so no rank is left waiting in a collective;
- `destroy_process_group()` then `SystemExit(0)` skips eval, checkpointing and the rest of the allocation.

Profiling **every rank** costs each rank the same export time and keeps them in step. With only rank 0 profiling,
the other ranks wait for rank 0's export in their next collective.

### Labels

- Put `record_function` labels in the **training loop**, around forward, backward, optimizer, gradient clipping and
  metrics: outside compiled code (next subsection). As a decorator, `@record_function("name")` labels every call of a
  helper the loop calls (`clip_grad` in the SKILL.md snippet).
- Labels the profiler adds on its own: `ProfilerStep#N`, `enumerate(DataLoader)#..._DataLoaderIter.__next__`,
  `Optimizer.step#AdamW.step`, `Optimizer.zero_grad#...`, `nccl:all_reduce` and other collectives,
  `DistributedDataParallel.forward`, and `Torch-Compiled Region: i/j` around each compiled graph.
- The backward kernels are launched from the autograd thread, not the thread holding the `backward` label.
  `trace_report.py` attributes kernels by launch time on any thread, so they still count toward `backward`.
- One extra label made no measurable difference to throughput (4×H100, within run-to-run noise).

### Labels inside compiled code

Under the default backend, a `record_function` *inside* a `torch.compile`d function never reaches the trace, whatever
you set. Tested on torch 2.15 (CPU), one label inside `torch.compile(fn, fullgraph=True)`:

| backend | `capture_profiler_record_function=False` (default) | `=True` |
|---|---|---|
| no compile | kept | kept |
| `aot_eager` | dropped | kept |
| `inductor` (default) | dropped | dropped |

A label around the compiled call was kept in all six cases.

Why:
- **Off (the default):** Dynamo treats `record_function` and `torch.profiler.profile` inside traced code like
  `contextlib.nullcontext`. There's no graph break and no label.
- **On:** `torch._dynamo.config.capture_profiler_record_function = True` makes Dynamo write
  `torch.ops.profiler._record_function_enter_new` and `_record_function_exit` nodes into the graph, and keeps them
  through functionalization and AOTAutograd. A backend that runs that graph as it is, such as `aot_eager`, emits the
  label.
  - **Introduced** by [pytorch#167787](https://github.com/pytorch/pytorch/pull/167787), "[dynamo] Preserve record
    function in aot_eager". It landed in December 2025, was reverted the same day after internal test failures,
    re-landed five days later, and first shipped in torch 2.11.
  - **Decorator form:** `@record_function("name")` on a function called from compiled code follows the same flag
    from torch 2.13 ([pytorch#184703](https://github.com/pytorch/pytorch/pull/184703)).
- **Inductor deletes those nodes** in its post-grad passes (`_remove_profiler_ops` in
  `torch/_inductor/fx_passes/post_grad.py`), "to prevent them blocking fusion". A label boundary inside a graph would
  split fusion groups, so the generated kernels never carry your labels. The same PR added the removal; it is on in
  every release that has the flag.

Under Inductor, use these instead:
- **Labels around compiled calls**, in the training loop.
- **What the trace already marks:** each compiled graph appears as `Torch-Compiled Region: i/j`, and its kernels are
  named after the ops they fuse ([reading.md](reading.md) §3).
- **A debug run with `backend="aot_eager"` and the flag on,** to break time down by your labels. It runs the traced
  aten ops without Inductor's fusion, so the shares describe the unfused model and shift once Inductor fuses.
- **`torch.compiler.disable` around the labelled code.** The label survives, but that code runs eagerly with a graph
  break on each side, which costs fusion. Measure before keeping it.

### Gradient accumulation

- Call `prof.step()` once per **optimizer** step, so a window holds whole accumulation cycles.
- If it must run per micro-batch, make `active` ≥ the accumulation count and check the window holds an optimizer
  step. Measured: at 8 micro-batches per optimizer step, the window landed on 3 accumulation micro-batches and
  showed zero collectives: a correct answer, and a useless one.

## 3. Options and what they cost

| option | adds | turn on when |
|---|---|---|
| `activities=[CPU, CUDA]` | host ops and GPU kernels; the default when CUDA is present | always. On ROCm, `CUDA` covers HIP |
| `record_shapes=True` | input shapes per op; holds tensor references during profiling | identifying a kernel by its shapes, checking which input shape a window caught |
| `with_stack=True` | Python frames (`python_function` events) | finding the source line of a sync or a slow host op |
| `profile_memory=True` | allocation events | rarely; for memory use `torch.cuda.memory._record_memory_history` and snapshots |
| `with_flops=True` | FLOP estimates for matmul and 2D conv only | rough arithmetic intensity |
| `acc_events=True` | accumulates events across cycles | `repeat` > 1 with `key_averages()` over all cycles |
| `with_modules` | nothing in eager mode | never: deprecated, TorchScript only |

Measured cost:
- **Throughput:** 0.4% slower with a 3-micro-batch window (4×H100, compiled); ~0.7% with 8 micro-batches traced
  (4×H100, eager). The cost that doesn't shrink with the window is the export.
- **Trace size** (`use_gzip=True` shrank the JSON 13-21× on the traces measured here):
  - 13 MB, 53,657 events: 3 micro-batches, no stack or shapes, 4×H100;
  - ~140 MB per rank: 40 steps;
  - 595 MB: 2 eager steps with stack and shapes, MI300A.
- **Host memory:** three jobs on a unified-memory APU (MI300A) failed while profiling: a node hang, a node failure,
  and a host OOM kill. The logs did not establish that the profiler caused them. On an APU, host memory and GPU
  memory are one pool, so the profiler's buffers compete with the model. Keep `active` small and stack and shapes
  off there unless needed.

## 4. Catching a rare stall

A stall once every ~1,000 steps won't land in a fixed window. Profile in rolling windows and keep only those that
contain a slow step.

```python
import collections
import os
import time

import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile, schedule


class StallProfiler:
    """Rolling windows of `active` steps; exports only windows holding a step of `slow_s` or more."""

    def __init__(self, skip_first, active, slow_s=5.0, out_dir="stalls"):
        os.makedirs(out_dir, exist_ok=True)
        self.slow_s = slow_s
        self.out_dir = out_dir
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.step_times = collections.deque(maxlen=active)
        self.last = None
        self._prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(skip_first=skip_first, wait=0, warmup=1, active=active, repeat=0),
            on_trace_ready=self._keep_if_slow,
        )

    def _keep_if_slow(self, p):
        if max(self.step_times, default=0.0) >= self.slow_s:
            p.export_chrome_trace(f"{self.out_dir}/rank{self.rank}_step{p.step_num}.pt.trace.json.gz")

    def __enter__(self):
        self._prof.__enter__()
        self.last = time.perf_counter()
        return self

    def __exit__(self, *exc):
        return self._prof.__exit__(*exc)

    def step(self):
        """Call once per optimizer step."""
        now = time.perf_counter()
        self.step_times.append(now - self.last)
        self.last = now
        self._prof.step()


with StallProfiler(skip_first=50, active=3) as prof:
    for batch in loader:
        ...  # forward, backward, optimizer
        prof.step()
```

- Run it on **every rank**. The stalled rank and the ranks waiting for it all cross the threshold, so you get every
  view of the same stall.
- **Don't judge steps inside `skip_first`:** the compile step would be flagged.
- **With `warmup=1`, one step in every `active + 1` is unrecorded,** so a stall can land there. `warmup=0` records
  every step at the cost of the start-up skew described above.
- **Stop it at loop exit,** or teardown is reported as a stall.
- The profiler is on for the whole run, so measure throughput with it on before relying on it.

Measured: one stall over 5 s per ~1,000 steps (4×H100, about 0.35% of throughput). The captured window showed
4.0-4.9 s in the DataLoader `__next__` with every rank's GPU idle, and forward and backward normal. A watchdog
dumping worker stacks at that moment found them waiting on page faults.

## 5. ROCm

- The same API and the same trace format: `ProfilerActivity.CUDA` profiles the AMD GPU, and kernels map to their
  launches by correlation id.
- **Kernel names differ:**
  - hipBLASLt GEMMs are `Cijk_*`;
  - rocPRIM sorts appear where CUB would on NVIDIA;
  - Inductor's flex-attention kernels are `triton_tem_fused_*`, with a numeric suffix that changed between runs.
- **Not tested here:** `trace_report.py` matches `hip*` runtime names and the ROCm kernel names above, but has only
  been run on NVIDIA traces.
