---
name: pytorch-profiler
description: >-
  Use torch.profiler to decide where to spend optimization effort in PyTorch
  training and to find what slowed a run down: GPU idle gaps, host-device syncs
  (.item(), cudaStreamSynchronize, pageable copies), DataLoader waits seen in a
  trace, kernel hotspots, launch-bound steps, exposed NCCL all-reduce, a late
  DDP rank, rare multi-second stalls, a regression between two versions. Use
  when setting up torch.profiler.profile, schedule (skip_first past
  torch.compile and autotune warmup), record_function or
  tensorboard_trace_handler; when reading a .pt.trace.json without
  chrome://tracing; when attributing GPU time under torch.compile; or when a
  fix that looked large in a trace didn't move throughput.
---

# PyTorch profiler: where to spend optimization effort

Profile a few steady-state steps on one rank unless there are stalls or stragglers (§9). Decide first whether the
GPU is busy or idle: busy means the kernels are the target, idle means finding what it waits for. Then rank the
candidates by potential time savings. **A trace diagnoses; an unprofiled A/B on the same node validates.** A fix
that removes idle time may only improve throughput by ~1% when the trace caught warmup, or when the idle
wasn't what limited the step.

## 1. Profile the steady state

Run without the profiler and log each step's wall time. The step after which it stays within noise is `skip_first`,
and it is often dozens of steps. Early steps are slow, and misleading, because of:
- torch.compile: the first call, and a recompile at each new input shape (minutes under `max-autotune`);
- each new shape's first run, even without a recompile;
- GEMM libraries choosing kernels for new shapes: long `aten::mm` host ops while the GPU idles;
- the caching allocator still growing: `cudaMalloc` calls inside the window;
- clocks not yet at thermal steady state, so early kernels can run *faster*.

A window of compile warmup can show several times the idle of a steady one, full of stalls that are gone a few dozen
steps later.
- **Varying input shapes:** profile after every shape has compiled, or at the most common shape. Note which shapes
  the window held. A rare shape's first batch can arrive late in the run. `TORCH_LOGS=recompiles` logs each
  recompile, so set `skip_first` past the last one.
- **Periodic work** (logging, validation, checkpointing): set `skip_first` to avoid it, or to hold one on purpose.
- **When you can't measure first, set it high.** Too high costs only run time; too low traces warmup.
- **Check afterwards:** the traced step spans should match the unprofiled step time — its spread, not its mean,
  when the steps aren't one tight mode, since a short window catches whichever kinds it happened to land on.

## 2. Capture

Run with `MYPROJ_PROFILE=skip_first,active`, e.g. `MYPROJ_PROFILE=50,3`: skip 50 steps, record 3 optimizer steps
and write the trace; a single-rank run then leaves the loop. `warmup` keeps its default of 1. Unset, the profiler
is a `nullcontext` and costs nothing; any other unparsable value raises. §9 cycles this schedule instead, to catch
a rare stall. Prefix the variable with your project's name, since a bare `PROFILE` can collide with other tools.

```python
from __future__ import annotations

import contextlib
import os
import socket
import time
import warnings
from pathlib import Path

import torch.distributed as dist
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function, schedule


class StepProfiler:
    def __init__(
        self,
        sched_skip_first: int,  # steps before the window, past compile and autotune warmup
        sched_warmup: int = 1,  # steps traced but discarded before the window
        sched_active: int = 3,  # steps in the window; 0 disables the profiler on this rank
        record_shapes: bool = False,  # input shapes for §7; grows the trace and holds tensor refs, so off for an A/B
        worker_name: str | None = None,  # trace file prefix; None: {hostname}_{pid}, as the handler names it
        out_dir: str | Path = "traces",  # trace directory, fresh per run
    ) -> None:
        self.trace_path: Path | None = None  # set when the window is written; stays None on an unprofiled rank
        self.out_dir = Path(out_dir)
        self.worker_name = worker_name or f"{socket.gethostname()}_{os.getpid()}"
        self._prof_ctx: profile | contextlib.nullcontext[None] = contextlib.nullcontext()
        if sched_active <= 0:  # off on this rank: the nullcontext stays and costs nothing
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._prof_ctx = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=record_shapes,
            # with_stack=True adds Python frames, but names the enclosing function, not the line, and grows the trace
            # for memory, torch.cuda.memory._record_memory_history with snapshots beats profile_memory
            schedule=schedule(
                skip_first=sched_skip_first,
                wait=0,
                warmup=sched_warmup,
                active=sched_active,
                repeat=1,  # one window; §9 cycles instead
            ),
            on_trace_ready=self._write,
        )

    def __enter__(self) -> StepProfiler:
        self._prof_ctx.__enter__()
        return self

    def __exit__(self, *args: object) -> None:
        if isinstance(self._prof_ctx, profile) and not self.trace_path:
            warnings.warn("profiler window didn't close: drop the last ProfilerStep")
        self._prof_ctx.__exit__(*args)  # an exit inside the window still writes the steps it caught

    def _write(self, prof: profile) -> None:  # tensorboard_trace_handler's naming, but it returns no path
        self.trace_path = self.out_dir / f"{self.worker_name}.{time.time_ns()}.pt.trace.json"  # .gz costs seconds
        prof.export_chrome_trace(str(self.trace_path))

    def step(self) -> None:  # once per optimizer step, never per micro-batch
        if isinstance(self._prof_ctx, profile):
            self._prof_ctx.step()  # the window's last step writes the trace, so trace_path is set once it closes


@record_function("clip_grad")  # decorator form: labels every call
def clip_grad(model: nn.Module) -> None:
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)


if dist.is_initialized():  # true once init_process_group has run
    rank = dist.get_rank()
    world_size = dist.get_world_size()
else:
    rank = 0
    world_size = 1
skip_first, active = (int(v) for v in os.environ.get("MYPROJ_PROFILE", "0,0").split(","))

prof = StepProfiler(skip_first, sched_active=active if rank == 0 else 0, worker_name=f"rank{rank}")  # rank 0 only
with prof:
    for batch in loader:
        with record_function("forward"):
            loss = model(batch)
        with record_function("backward"):
            loss.backward()
        clip_grad(model)
        with record_function("optimizer"):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        prof.step()  # under gradient accumulation, only on the micro-batch that ends a cycle
        if prof.trace_path and world_size == 1:
            break  # with peers, one rank leaving here would deadlock them in the next collective
if prof.trace_path:
    print(f"profiler trace: {prof.trace_path}")
```

- **Labels go in the training loop, outside compiled code.**
  - Under Inductor, a `record_function` inside a compiled function never reaches the trace. Dynamo drops it by
    default.
  - `torch._dynamo.config.capture_profiler_record_function = True` (torch ≥ 2.11) puts the label into the graph, but
    Inductor then removes it so that it can't block fusion. Only backends that skip Inductor, such as `aot_eager`,
    keep it.
  - The profiler adds some labels on its own: DataLoader `__next__`, `Optimizer.step#...`, collectives, and
    `Torch-Compiled Region: i/j` around each compiled graph.
- **Trace one rank.** Ranks run the same code on the same shapes, so idle, syncs and hotspots read alike on any of
  them; trace every rank only to compare ranks, for a slow step (§9) or because §8's collective time reads as a
  minimum. Use a fresh directory per run, or a stale file reads as another rank.
- **Stop after the capture.** The step that closes the window also writes the trace, far longer than a step.
  Single-rank: break once `trace_path` is set. Multi-rank: cap at the launcher, since a rank that breaks deadlocks
  its peers — and drop the window and the next step from throughput stats, since the stall shows in the *peers'*.
- **Step once per optimizer step,** so the schedule counts optimizer steps and each window holds whole accumulation
  cycles; stepped per micro-batch, a window can land on accumulation micro-batches only and show no all-reduce. That
  means the micro-batch that *finishes* a cycle, not the one calling `optimizer.step()` — Accelerate's wrapper calls
  that every micro-batch. Use `accelerator.sync_gradients`, `(i + 1) % accum == 0`, or a framework's step-end hook.
- **A window the class warns didn't close** ended before or inside itself: run more steps, or lower `skip_first`.
  Drop its last `ProfilerStep#`, which spans post-loop code (a final eval or checkpoint), not a step.

## 3. Read the trace

Read the trace as text first: that works where `chrome://tracing` is blocked, and an agent can do it with a short
stdlib script. The file is Chrome trace JSON, gzipped when it ends in `.gz`: `{"traceEvents": [...]}`.
- **Fields:** each event has `name`, `cat`, `ph`, `ts` and `dur` (both µs), `pid`, `tid` and `args`.
- **Keep only `ph == "X"`.**
- **One rank per file.** Each file loads whole, so read files of hundreds of MB in a CPU job, not on a shared login
  node.
- **`ProfilerStep#N` is not the training step:** label each with `record_function(f"train_step_{global_step}")`.

| quantity | rule |
|---|---|
| window | `cat == "user_annotation"` and `name` starts with `ProfilerStep#`: from the first start to the last end |
| GPU work | `cat` in `kernel`, `gpu_memcpy`, `gpu_memset`, with `dur > 0` |
| GPU work of the window | its launch is inside the window: the event with `cat` in `cuda_runtime`, `cuda_driver` and the same `args.correlation` |
| busy, idle | busy is the union of GPU intervals; idle is the gaps between them, from the first GPU start to the last end |
| compute-only busy | the same union without collective kernels, `name` matching `(?i)[nr]ccl` |
| exposed comm | busy − compute-only busy |
| idle waiting for data | idle overlapping `user_annotation` events named `enumerate(DataLoader)#...` |
| syncs | `cat` in `cuda_runtime`, `cuda_driver`, `name` matching the sync pattern below, and `ts` inside the window |
| a sync's cost | the gap that opened during the call: from the call's return to the next GPU start (0 if the GPU stayed busy) |
| call site | the shortest `user_annotation` and `cpu_op` on the same `tid` that contain the event |
| gap context | the `cpu_op` and `user_annotation` events covering most of the gap; none means an untraced stall (§6) |
| hotspots | `dur` summed per kernel `name`, with launch counts |
| per step | for each `ProfilerStep#`: kernel count, GPU time, and host label time |

Sync pattern: `^(cuda|hip)(Stream|Device|Event)Synchronize$`.

**Report:**
- the step spans;
- busy %, compute-only busy %, and exposed comm;
- idle split by gap size: under 10 µs (kernels back to back, normal on a busy GPU), 10 µs to 1 ms (the host not
  launching fast enough), and 1 ms or more (stalls);
- the idle that overlaps DataLoader waits;
- syncs grouped by (name, label, op), with total wait and total idle after;
- the largest gaps with their context;
- the top kernels;
- the per-step figures, which say whether the window is one regime or a blend of unlike steps (§4).

**Why the rules:**
- **`gpu_user_annotation` is not work.** These spans are labels projected onto the GPU timeline, and they cover idle
  time. Counting them can turn a quarter-idle window into "nearly all busy". `ProfilerStep#N` has such a copy too,
  which is why the window comes from the host `user_annotation`.
- **Select by launch, not by run time.** Kernels run behind the host, and the profiler keeps recording after the
  last step.
- **Launches use two APIs.** Triton and Inductor kernels launch through `cuLaunchKernel` (`cuda_driver`); cuBLAS and
  aten kernels through `cudaLaunchKernel` (`cuda_runtime`). Scanning only one makes real launches look like host
  stalls.
- **Collective kernels spin while waiting for peers,** so they count as busy.
- **A sync's wait is the queue draining (§5).** Its cost is the idle after it returns.

**Per-op tables,** using the documented calls:

```python
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))
```

A table has no timeline, so it can't show gaps, overlap or syncs. Host time on an op includes any blocking, and
grouping by input shape is empty unless the capture set `record_shapes=True`.

**HTA** (`pip install HolisticTraceAnalysis`) answers two questions the rules above can only infer. It reads a
directory of traces, gzipped or not:

```python
from hta.trace_analysis import TraceAnalysis

analyzer = TraceAnalysis(trace_dir="traces")
idle = analyzer.get_idle_time_breakdown()  # idle split into host_wait, kernel_wait and other wait
queue = analyzer.get_queue_length_summary()  # launch queue depth; a minimum of 0 means the host ran dry
launches = analyzer.get_cuda_kernel_launch_stats()  # launch delay, and kernels shorter than their own launch
```

`host_wait` against `kernel_wait` is §4's host-or-GPU question answered directly, where the gap-size split above
is a best guess.
- **HTA's totals cover the whole trace, not the `ProfilerStep#` window,** so its idle share and the window's differ:
  22.9% against 17.2% on one trace here. Never compare a percentage from one tool against one from the other.
- `get_gpu_kernel_breakdown`'s mean/min/max/stddev columns are per rank, so a single-rank trace gives
  min = max = sum and stddev = 0, which reads like a single call.

**By eye:** `ui.perfetto.dev` opens `.json.gz` traces. It parses the trace in the browser, with a limit around 2 GB.
Use v57 or later, since earlier builds dropped PyTorch events that overlapped on one track. Check your site's policy
before opening traces from private code.

## 4. Where to spend effort

Start from the step's composition: compute-only busy, exposed communication, and idle. Then bound each candidate
before working on it:
- **Removing all idle** buys at most GPU window ÷ busy time.
- **Making a kernel family free** buys at most 1 ÷ (1 − its share of the step). A family at 20% caps at 1.25×.
- **Removing a sync** buys its idle after, not its wait.

| the trace shows | it is | go to |
|---|---|---|
| step spans, kernel counts or GPU time varying between steps; `cudaMalloc` or compile events | the wrong window | §1 |
| compute-only busy close to the whole window, few gaps | GPU-bound | §7 |
| idle while the host waited in DataLoader `__next__` | input-bound | the `pytorch-dataloader` skill |
| syncs with idle after them; gaps that start as a sync returns | sync drains | §5 |
| thousands of 10-100 µs gaps, host time far above GPU time, or a launch queue draining to 0 | the host can't keep up | §6 |
| long host ops launching nothing while the GPU idles | host work | §6 |
| a gap no host event covers, or one cheap op spanning tens of ms | an untraced host stall | §6 |
| exposed comm a large share of the window | communication-bound | §8 |
| one rank with the most data wait and the least collective time (all ranks traced) | a late rank | §9 |
| an all-reduce every micro-batch, or none in a multi-rank DDP window | `no_sync` not in effect, or no DDP wrapper | §8 |

## 5. Syncs

The host runs ahead of the GPU, filling a queue. A sync blocks the host until the GPU catches up.
- **During the wait, the GPU runs the queue.** A long wait means long queued work, not a slow sync. A huge `.item()`
  can be a slow kernel surfacing.
- **The cost is the idle after.** If the sync emptied the queue, the GPU idles until the host launches again. A sync
  that waited on older work while the GPU stayed busy is backpressure, and harmless. Inductor issues such
  `cudaEventSynchronize` calls itself.

| code | in the trace |
|---|---|
| `.item()`, `float(t)`, `.tolist()`, `print(t)`, `if t:` | `aten::_local_scalar_dense` → `cudaStreamSynchronize` |
| `.cpu()`, `.numpy()` | `aten::copy_` → `Memcpy DtoH` + `cudaStreamSynchronize` |
| `torch.tensor(x, device="cuda")` | `aten::copy_` → `Memcpy HtoD (Pageable -> Device)` + `cudaStreamSynchronize` |
| `.to(device)` without `non_blocking=True` | `aten::copy_` → `Memcpy HtoD` + `cudaStreamSynchronize` |
| `nonzero`, boolean masks, `masked_select`, `unique` | `cudaStreamSynchronize` under the op |
| `torch.cuda.synchronize()` | `cudaDeviceSynchronize` |

- **Find the line:** the call site (§3) gives the enclosing label and aten op. Narrow the labels until one
  statement is left, or use `torch.cuda.set_sync_debug_mode("warn")`, which warns with a traceback. It misses
  `torch.cuda.synchronize()` and non-blocking copies from pageable memory.
- **Fix:**
  - keep metrics on the device and `.item()` them at logging steps;
  - build device scalars with `torch.zeros((), device=...)` or `torch.full`, never `torch.tensor(value, device=...)`;
  - copy batches with `pin_memory=True` and `non_blocking=True`;
  - take data-dependent shapes out of the step.
- **Re-profile after each fix.** A drain hides any smaller one behind it.

## 6. Idle gaps and host overhead

- **A few big gaps** are discrete stalls: read what the host was doing in each one.
- **The host can't keep the GPU fed** when HTA shows idle dominated by `host_wait`, a launch queue draining
  towards 0, or launch delays near 0 (§3). The causes are below.
- **Launch-bound:** thousands of 10-100 µs gaps that add up, kernels of a few µs, each host op launching one
  kernel, and labels whose host time far exceeds their GPU time. Levers:
  - torch.compile, which fuses elementwise chains;
  - CUDA graphs (`mode="reduce-overhead"`), but only with fixed shapes;
  - larger batches;
  - moving bookkeeping off the step.
  - A `@torch.compiler.disable` region pays launch overhead for every op inside it, so shrink it to the call that
    needs it.
- **Long host ops that launch nothing while the GPU idles:** at new shapes this is usually library heuristics, i.e.
  warmup. Otherwise, move the host work off the critical path.
- **A gap no host event covers,** or a cheap op stretched to tens of ms, means the thread was blocked where the
  profiler can't see: the GIL held by another thread, garbage collection, an allocator lock, a page fault, or CPU
  contention.
  - A full GC pass in a large compiled process can take hundreds of ms, and on every rank at once.
  - Tools: `py-spy dump --pid <pid> --native` while it's stuck, `gc.callbacks` to time collections, and
    `faulthandler.dump_traceback_later(t, repeat=True)` as a watchdog. `gc.freeze()` after setup takes long-lived
    objects out of future collections.

## 7. Hotspots

- **Rank kernels by total time** (§3 hotspots), then group them by family: GEMM, attention, Triton
  (Inductor), elementwise, communication, memcpy.
- **Attribute by label through the launch,** not the host span. A kernel belongs to the label that was open when it
  was *launched*, on any thread: backward kernels launch from the autograd thread.
- **Under torch.compile, module scopes don't map to kernels.** Read Inductor's kernel names instead:
  - `triton_poi_*` is pointwise;
  - `triton_per_*` is a persistent reduction;
  - `triton_red_*` is a reduction;
  - `triton_tem_*` is a template (GEMM or attention);
  - the suffix lists the fused aten ops.
- **To identify one kernel,** use its input shapes, its grid, or a launch count equal to layers × steps. Shapes
  aren't in the triage capture: re-record a short window with `record_shapes=True` (§2).
- **Name-based families drift:**
  - some flash-attention kernels are CUTLASS-built (`kernel_cutlass_kernel_flash_attn...`), so test `flash` before
    `cutlass`;
  - `max-autotune` moves GEMMs from the vendor library into Triton templates.
- **A/B the compile mode before hand-writing a kernel.** `max-autotune` alone has made an attention backward kernel
  about twice as cheap, with no code change. See the `pytorch-compile` skill.
- **Check headroom:**
  - GEMMs: achieved FLOP/s ÷ the GPU's peak;
  - elementwise kernels: bytes moved ÷ time against memory bandwidth — `numel × element size` summed over inputs
    and outputs, read from the launching `cpu_op`, since an Inductor kernel hides its inputs in one fused op.
  - A kernel already near its roofline has little left to give. Look elsewhere.

## 8. Distributed

- **Exposed communication** is collective time with no compute running on any other stream. That's what it costs.
  Collective kernel duration includes spinning while waiting for late peers, so a late rank inflates it (§9). For
  the time the transfer itself takes, trace every rank and use each step's minimum collective time across ranks,
  then the median over steps; skew moves between ranks and between jobs.
- **Gradient accumulation:** stepping per optimizer step (§2) puts whole cycles in the window, so expect one set of
  all-reduces per step, on the cycle's last micro-batch only. If every micro-batch all-reduces, `no_sync()` isn't
  taking effect: check that the object you call it on is the DDP wrapper. A model-unwrapping helper can return the
  inner module, e.g. Accelerate's `unwrap_model()` with `keep_torch_compile=True`.
- **Multi-node:** confirm the transport before tuning buckets. Run with `NCCL_DEBUG=INFO`; a line like `GPU Direct
  RDMA Disabled` means every byte is staged through host memory. `NCCL_NET_GDR_LEVEL=SYS` can enable it when the GPU
  and NIC sit on different sockets.

## 9. Finding a slowdown

**A slow step** costs every rank, because the step waits for the slowest. A straggler and a rare stall are the same
failure, and it reads backwards: the late rank spends the *least* time in collectives, because the others spin
waiting for it, so its data wait or sync reads as all-reduce time on their traces, not as a wait. Three questions:
- **How often is it late?** Every step: the fixed window of §2, recorded on every rank, since a lone traced rank
  pays the profiler and would itself read as late. Rarely: no fixed window will land on it, so cycle windows for
  the whole run and keep the first one that caught a stall (below).
- **One rank, or all at once?** One rank is skew, and its own trace holds the cause; the peers show only all-reduce
  time. Every rank at once is a shared cause — storage, GC (§6), the scheduler — and every GPU idles together, so
  there is no late rank to find. A single-GPU run is always this case.
- **What was the late rank doing?** A DataLoader wait (the `pytorch-dataloader` skill), a sync (§5), or a gap no
  host event covers (§6).

**The rolling capture** varies §2's class. Add a `stall_threshold_s` argument in seconds; time each step in
`step()` into a `collections.deque(maxlen=sched_active)`; pass `repeat=0` to the schedule, which cycles windows
until the run ends instead of stopping after one; and hand `on_trace_ready` this in place of `_write`:

```python
def _write_if_stalled(self, prof: profile) -> None:
    if self.trace_path:  # the write stalls its own step, which would otherwise trigger the next capture
        return
    if max(self.step_times_s) >= self.stall_threshold_s:
        self._write(prof)
```

Run it on every rank, `StepProfiler(50, sched_active=3, worker_name=f"rank{rank}")` with a 5 s threshold.
`trace_path` is set once a stall is caught, so drop the early stop from the loop: a rolling run trains to the
end, and `trace_path` holds the trace.
- **Steps within `skip_first` and `warmup` aren't judged,** so the compile step isn't flagged.
- **The kept window is its own file.** Its `ProfilerStep#N` labels give the step.
- **Ignore §2's "window didn't close" warning here.** It assumes one fixed window; a rolling run that caught no
  stall writes nothing, which is the intended outcome.
- **Usually only the late rank crosses the threshold,** since a peer whose host never blocks on the GPU keeps
  launching while its GPU waits in the all-reduce, so its host step time stays normal. To keep every rank's view of
  the same window, all-reduce the max of `step_times_s` in `_write_if_stalled`, which costs one sync per window.

**A regression between two versions:**
- Profile both at the same steady step, with the same input shapes, on the same node, back to back.
- Compare per token (or per sample): compute-only busy, exposed comm, idle by cause, sync rows, and the top kernels.
- Diff the kernel lists: a new kernel, a changed launch count, or a GEMM that left the vendor library usually names
  the change.
- Check that neither window holds a recompile.
- Compare **median** step time and kernel time, not means: one DataLoader stall can double a short run's mean.

## 10. Measurement rules

- Confirm every fix with an unprofiled A/B, on the same node, back to back.
- Profile the steady state, and check the traced step spans against the unprofiled step time.
- A host span is not a cost, and a sync's wait is not its cost.
- Re-profile after each fix.
- Trace one rank; every rank to compare ranks.
- Compare medians and kernel time when the DataLoader stalls.
- Normalize per token when batch fill changes.
- Compare windows taken at the same point of the run, since early kernels can run faster.
- Log a line on the path under test, and read knobs from the resolved config.
- A label missing from the trace means the code didn't run, or it ran inside compiled code (§2).

*API facts checked against the torch 2.15 docs and source, and HTA against the 0.5.0 docs (2026-11).*
