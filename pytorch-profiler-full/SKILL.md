---
name: pytorch-profiler-full
description: >-
  Find where a PyTorch training step's time goes with torch.profiler: GPU idle
  gaps, host-device syncs (.item(), cudaStreamSynchronize, pageable copies),
  DataLoader waits seen in the trace, kernel hotspots, launch-bound steps,
  exposed NCCL/RCCL all-reduce, a late DDP rank, rare multi-second stalls. Use
  when setting up torch.profiler.profile, schedule (skip_first past
  torch.compile, autotune and allocator warmup), record_function labels or
  tensorboard_trace_handler; when reading a .pt.trace.json without
  chrome://tracing (text report, key_averages, Perfetto); when attributing GPU
  time under torch.compile or on ROCm; when gradient accumulation still
  all-reduces every micro-batch; or when a fix that looked large in a trace
  didn't move throughput.
---

# PyTorch profiler: where the step goes

A router. Capture a short steady-state window on every rank, read GPU idle before hotspots, and confirm every fix
without the profiler. Depth lives in `references/`.

**Hard rule:** a trace diagnoses, an unprofiled A/B decides. Measured: a host-sync fix cut trace idle from 24.5% to
10.9% and gained 1.01× tokens/s (4×H100), because the 3-step trace had sampled compile warmup. See
[measurement.md](references/measurement.md).

---

## 1. Know the steady step before profiling

Run without the profiler, log every step's wall time, and find the step after which it stays within noise. That
count is `skip_first`. It is often dozens of steps:
- torch.compile: 60-110 s for the first step;
- each new input shape: a recompile of 54-384 s depending on the mode;
- GEMM library heuristics at new shapes;
- the caching allocator still growing.

| same job (4×H100, compiled) | idle | largest gap |
|---|---|---|
| micro-batches 2-4 | 12.7% | 3.34 ms |
| micro-batches 61-63 | 1.8% | 7 µs |

With varying shapes, profile after every shape has compiled, or at the most common one, and log which shapes the
window held. [capture.md](references/capture.md) §1.

---

## 2. Capture

Run with `MYPROJ_PROFILE=skip_first,warmup,active`, e.g. `MYPROJ_PROFILE=50,1,3`: skip 50 steps until the unprofiled
step time is steady, discard 1, record 3 optimizer steps. Unset, the profiler is a `nullcontext` and costs nothing;
an empty or malformed value raises. Prefix the variable with your project's name: a bare `PROFILE` can collide with
other tools.

```python
from __future__ import annotations

import contextlib
import os
import warnings
from types import TracebackType

import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile, record_function, schedule, tensorboard_trace_handler


class StepProfiler:
    """torch.profiler over one window of optimizer steps, or a no-op when active is 0."""

    def __init__(
        self,
        skip_first: int,
        warmup: int,
        active: int,
        out_dir: str = "traces",
    ) -> None:
        self.enabled = active > 0
        self.steps = 0
        self.total_steps = skip_first + warmup + active
        self._prof: profile | contextlib.nullcontext[None] = contextlib.nullcontext()
        if not self.enabled:
            return
        rank = dist.get_rank() if dist.is_initialized() else 0
        self._prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(skip_first=skip_first, wait=0, warmup=warmup, active=active, repeat=1),
            on_trace_ready=tensorboard_trace_handler(out_dir, worker_name=f"rank{rank}", use_gzip=True),
        )

    @classmethod
    def from_env(cls, env_var: str, out_dir: str = "traces") -> StepProfiler:
        """Reads env_var=skip_first,warmup,active, e.g. MYPROJ_PROFILE=50,1,3. Unset disables; malformed raises."""
        spec = os.environ.get(env_var)
        if spec is None:
            return cls(0, 0, 0, out_dir)
        fields = spec.split(",")
        if len(fields) != 3:
            raise ValueError(f"{env_var}={spec!r}: expected skip_first,warmup,active, e.g. {env_var}=50,1,3")
        skip_first, warmup, active = (int(field) for field in fields)
        return cls(skip_first, warmup, active, out_dir)

    def __enter__(self) -> StepProfiler:
        self._prof.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self.enabled and self.steps < self.total_steps:
            warnings.warn(f"profiler stopped after {self.steps} of {self.total_steps} steps: no trace written")
        self._prof.__exit__(exc_type, exc, tb)

    def step(self) -> bool:
        """Call once per optimizer step. Returns True once the trace is written."""
        if not isinstance(self._prof, profile):
            return False
        self._prof.step()
        self.steps += 1
        return self.steps == self.total_steps


@record_function("clip_grad")  # decorator form: labels every call
def clip_grad(model: torch.nn.Module) -> None:
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)


# after init_process_group, so the rank is known
prof = StepProfiler.from_env("MYPROJ_PROFILE")
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
        if prof.step():
            break  # trace written; every rank stops at the same step
if prof.enabled:
    if dist.is_initialized():
        dist.destroy_process_group()
    raise SystemExit(0)  # skip eval, checkpointing and the rest of training
```

- **Labels go in the training loop, outside compiled code.** Under Inductor, a label inside a compiled function never
  reaches the trace. `torch._dynamo.config.capture_profiler_record_function=True` keeps it only for backends that skip
  Inductor's fusion, such as `aot_eager`; Inductor strips it so it can't block fusion (tested, torch 2.15). DataLoader
  `__next__`, `Optimizer.step#...`, collectives and `Torch-Compiled Region: i/j` are labelled automatically.
  [capture.md](references/capture.md) §2.
- **`@record_function("name")` labels a function** the loop calls, as `clip_grad` above. The same rule applies:
  only on functions that run outside compiled code.
- **Profile every rank.** Each writes `rank{N}.<ns>.pt.trace.json.gz`. Use a fresh directory per run.
- **Step the profiler per optimizer step.** Per micro-batch, a window landed on 3 accumulation micro-batches and
  showed zero collectives (8 micro-batches per step).
- **A run shorter than `skip_first + warmup + active` steps writes no trace.** The class warns at exit.
- **Exit once captured.** The step that closes the window also builds and writes the trace. It took 1160 ms against
  ~50 ms neighbours (MI300A).
- **Cost:** 0.4% throughput (4×H100, 3 steps). Size: 13 MB for 3 steps without stack or shapes, 595 MB for 2 eager
  steps with both; gzip shrank traces 13-21×.
- **Options:**
  - `with_stack=True` only to locate host code, and it names functions, not lines;
  - `record_shapes=True` only to identify kernels by shape.

Schedule semantics, options, and the rolling capture for rare stalls: [capture.md](references/capture.md).

---

## 3. Read the trace

```bash
python3 scripts/trace_report.py traces/              # full report for the first rank, plus a row per rank
python3 scripts/trace_report.py traces/ --rank 1     # full report for rank 1
```

- **Stdlib only.** It reads `.json` and `.json.gz`, and applies the traps in [reading.md](references/reading.md) §1:
  - only kernels, memcpy and memset count as GPU work;
  - GPU work is selected by its launch, matched by correlation id;
  - launches come from both the runtime and the driver API.
- **Read in order:**
  1. the `[warn]` lines (a wrong window makes everything below moot);
  2. Steps;
  3. Idle;
  4. Syncs;
  5. Regions;
  6. Host ops;
  7. Comm;
  8. Kernels;
  9. Ranks.
- **Per-op tables:** `prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1)`. A table has no
  timeline, so it shows neither gaps nor syncs.
- **By eye:** `ui.perfetto.dev` (v57+), if your site allows it. It parses the trace in the browser; about 2 GB is the
  limit. [reading.md](references/reading.md) §4-5.

---

## 4. Triage

| the report shows | it is | go to |
|---|---|---|
| `[warn]` step spans vary, `cudaMalloc` or `torch.compile ran inside the window` | the wrong window | §1 |
| compute busy near 100%, few gaps | GPU-bound: hotspots | §7 |
| idle "while the host waited in DataLoader `__next__`" | input-bound | the `pytorch-dataloader` skill |
| sync rows of kind "drains queue" with idle after; idle "that began as a sync returned" | sync drain | §5 |
| thousands of 10-100 µs gaps; labels with far more host time than GPU time | launch-bound | §6 |
| host ops with GPU idle meanwhile | host work blocking launches | §6 |
| a big gap that "no host event covers", or one cheap op spanning tens of ms | an untraced host stall: GIL, GC, allocator | §6 |
| comm exposed is a large share of the window | communication-bound | §8 |
| a rank with the most data wait and the least collective-kernel time | a straggler | §8 |
| collectives on every micro-batch, or none at all under DDP | `no_sync` broken, or a window without an optimizer step | §8 |

---

## 5. Syncs

- **The cost of a sync is the GPU idle after it returns, not the wait.** During the wait the GPU runs the queue. The
  report classes each call site:
  - **drains queue:** the GPU idles until the host launches again;
  - **queue stays full:** backpressure, harmless;
  - **GPU already idle:** the gap was already there.
- **Measured:**
  - Inductor's own `cudaEventSynchronize` held 38-43% of an L40S step inside `forward` with the GPU 99.8-100%
    busy: harmless.
  - `torch.tensor(0.0, device=...)` cost a 63-73 ms drain every step (4×H100). It still syncs in torch 2.15.
  - A 51.6 s `.item()` was a BLAS regression surfacing (MI300A).
- **Find the line** with the report's call site (label | aten op) or `torch.cuda.set_sync_debug_mode("warn")`.
- **Re-profile after each fix:** a drain hides smaller ones behind it.

[syncs-and-gaps.md](references/syncs-and-gaps.md) §1-4.

---

## 6. Idle gaps and host overhead

- **Few big gaps:** read what the host was doing in each one. **Thousands of tiny gaps:** launch overhead.
  - Cold window (4×H100): 3 gaps over 10 ms were 58% of the idle time, while 3,967 gaps under 100 µs added up to
    10.3 ms.
- **Host ops launching nothing while the GPU idles** are often warmup. GEMM heuristics at new shapes left the GPU
  idle 72.8% of their host time in a cold window and 2.2% at steady state.
- **Launch-bound levers:** torch.compile (fusion), CUDA graphs (fixed shapes only), larger batches.
  - Moving a `@torch.compiler.disable` boundary inward took a step from 236 to 65 ms (MI300A).
- **Nothing traced under a gap** means the thread was blocked outside the profiler's view. Use `py-spy dump`,
  `gc.callbacks` and `faulthandler`. Full GC passes in a compiled process took 160-200 ms on both ranks at once
  (2×L40S).

[syncs-and-gaps.md](references/syncs-and-gaps.md) §5-8.

---

## 7. Hotspots

- **Kernels by family and the top kernels** are in the report. Region GPU time comes from correlation ids, so
  backward kernels launched from the autograd thread still count toward `backward`.
- **Under torch.compile, module scopes cover only ~11% of kernel time.** Use loop-level labels plus Inductor kernel
  names:
  - `triton_poi`, `triton_per`, `triton_red`, `triton_tem`;
  - the suffix lists the fused ops.
  - Identify a specific kernel by shape, grid, or launch count = layers × steps.
- **Family rules drift:**
  - a `cutlass` match booked FA4 attention as GEMM (~85 ms in a 3-step window);
  - `max-autotune` moves GEMMs into Triton templates.
- **A/B the compile mode before hand-writing kernels.** `max-autotune-no-cudagraphs` made a flex-attention backward
  2.13× cheaper, and bf16 attention 2.23× more (MI300A). See the `pytorch-compile` skill.
- **ROCm:** hipBLASLt GEMMs are `Cijk_*`. The report's ROCm handling is untested on real ROCm traces.

[reading.md](references/reading.md) §3.

---

## 8. Distributed

- **Exposed = collective time with no compute on any other stream.** Collective kernels spin while waiting for
  peers, so their duration isn't cost, and they count as "busy":
  - 4×H100 on one node: 96.9% overlapped;
  - 2×2 A100: 80.6% of the window exposed, while total busy read ~100%.
- **Stragglers:**
  - The late rank spends the least time in collectives. Its DataLoader wait or sync shows up on the others as
    all-reduce time.
  - Two ranks of one job differed up to 5×.
  - Take the per-step min across ranks, then the median over steps.
- **Gradient accumulation:** accumulation micro-batches should show zero all-reduces. Accelerate's
  `unwrap_model()` at its default `keep_torch_compile=True` hid the DDP wrapper, so `no_sync()` did nothing. Fixing
  it took 2×2 A100 MFU from 11.1% to 51.8%.
- **Multi-node:** check that GPU Direct RDMA is on before tuning buckets. With `NCCL_DEBUG=INFO`, look for `GPU
  Direct RDMA Disabled`. `NCCL_NET_GDR_LEVEL=SYS` took comm from 772 to 140.5 ms per step (2 nodes × A100).

[distributed.md](references/distributed.md).

---

## 9. Rare stalls

- **Rolling windows** (`repeat=0`) on every rank, exporting only windows with a step over a threshold.
- **Don't judge steps inside `skip_first`,** and stop the capture at loop exit.
- **Measured:** one stall over 5 s per ~1,000 steps (4×H100). The window showed 4.0-4.9 s in the DataLoader
  `__next__` with every GPU idle.

[capture.md](references/capture.md) §4.

---

## 10. Measurement rules (short form)

- An unprofiled A/B on the same node decides every fix.
- Only kernels, memcpy and memset are GPU work.
- A host span is not a cost, and a sync's wait is not its cost.
- Read medians and kernel time when the loader stalls. A control run's mean was 138 ms against 62 ms while its median
  was 39.5 against 35.9.
- Re-profile after each fix.
- Profile every rank.
- Don't compare kernel times across points in a run: warmup kernels ran ~17% faster.
- Normalize per token.

Each rule with its incident: [measurement.md](references/measurement.md).

---

## 11. Reference map

| open | when |
|---|---|
| [capture.md](references/capture.md) | choosing the window, schedule semantics, labels, exit, options and their cost, rare stalls, ROCm |
| [reading.md](references/reading.md) | the report's sections, event categories, attribution by correlation id, kernel names, `key_averages`, Perfetto |
| [syncs-and-gaps.md](references/syncs-and-gaps.md) | syncs, idle gaps, host ops, launch-bound steps, stalls the profiler can't see |
| [distributed.md](references/distributed.md) | exposed communication, stragglers, gradient accumulation, GPU Direct RDMA, DDP knobs |
| [measurement.md](references/measurement.md) | before trusting any number from a trace |

*Numbers come from in-house runs: 4×H100 and 2×2 A100 with a 354M-parameter transformer, a 60M-parameter
transformer on MI300A, L40S runs, and a 2×L40S test loop on torch 2.15. Re-measure on yours; the mechanisms carry
over, the numbers don't. API facts checked against the torch 2.15 docs and source (2026-11).*
