---
name: pytorch-compile
description: >-
  Diagnose and fix torch.compile problems: graph breaks, recompilation, slow
  compile time, and compile-mode / runtime tradeoffs. Use when torch.compile is
  slow to compile, slower than eager, stalls for minutes on step 1 or on every
  new shape, or when you see graph break, "Recompiling function", guard failure,
  cache_size_limit / recompile_limit, dynamic shapes, mark_dynamic, tlparse,
  TORCH_TRACE, TORCH_LOGS=recompiles/graph_breaks, Dynamo, Inductor, Triton
  autotune, max-autotune, coordinate_descent_tuning, specialize_float, CUDA
  graphs / reduce-overhead, cudagraph_mark_step_begin, allow_in_graph,
  torch.compiler.disable, or "accessing tensor output of CUDAGraphs". Also for
  profiling a compiled model (kernels are fused and invisible to module hooks)
  and for measurement hygiene when benchmarking compiled vs eager.
---

# torch.compile diagnostics

A router. Identify which of four problems you have, run one command, open one
reference. Depth lives in `references/`; this file only gets you there.

**Hard rule:** never change a compile knob without a before/after measurement.
Compile effects are large, run-to-run variance is larger, and most of the
interesting quantities are not in any obvious log. See
[measurement.md](references/measurement.md).

---

## 0. Turn the logs on first

Almost every torch.compile question is unanswerable without these, and almost
nobody has them on.

```bash
TORCH_LOGS="graph_breaks,recompiles" python train.py 2> run.err   # near-free
TORCHDYNAMO_VERBOSE=1 TORCH_LOGS="graph_breaks,recompiles" ...    # + tracebacks
TORCH_TRACE=/abs/dir python train.py                              # ~10 s overhead
```

- **Output is on stderr**, not stdout. Capture `2> run.err` or `2>&1`.
- **Never export `TORCH_LOGS` as an empty string.** Present-but-empty has crashed
  `import torch` on some builds. Set it or unset it.
- **Zero log lines proves nothing** unless you can show the variable was live in
  that process. Print `os.environ.get("TORCH_LOGS")` at startup.
- In-process, when you cannot set env vars: `torch._dynamo.explain(fn)(*args)` and
  `torch._dynamo.utils.counters`.

---

## 1. Triage: which of four problems is this?

| you observe | signal that identifies it | first command | go to |
|---|---|---|---|
| compiled is slower than eager, or the profile shows many small compiled regions | `Graph break in user code at` lines; `explain()` reports >1 graph | `python3 scripts/log_triage.py run.err` | [graph-breaks.md](references/graph-breaks.md) |
| a multi-second stall recurs whenever input shapes change | `Recompiling function ... triggered by the following guard failure(s)` | `python3 scripts/log_triage.py run.err --recompiles` | [recompiles.md](references/recompiles.md) |
| one huge stall on step 1 (or the first step of each shape), then fast | no recompile lines after the first; step-1 wall time ≫ steady state | `python3 scripts/trace_summary.py $TORCH_TRACE` | [compile-time.md](references/compile-time.md) |
| steady state itself is slow, crashes, or memory grew | no compile-log activity at the slow steps | profile and attribute by correlation id | [mode-selection.md](references/mode-selection.md) + [tooling.md](references/tooling.md) |

Four notes that decide most ambiguous cases:

- **Breaks and recompiles are different mechanisms with different log channels.**
  `TORCH_LOGS=recompiles` alone shows zero graph breaks even when there are dozens.
- **N recompile lines ≠ N shape changes.** One shape change emits one recompile
  event *per graph-broken frame* — measured 23 events for a single transition on a
  fragmented model, 6 once the breaks were fixed. Fix breaks first.
- **If step 1 is enormous but shapes never change again**, it is a compile-time
  problem, not a recompile problem, and no caching knob makes the *first* compile
  cheap.
- **If the slow steps have no stderr activity of any kind**, it is not
  compilation. Check the input pipeline and machine variance before touching a
  compile knob.

---

## 2. Graph breaks

A break splits one compiled region into two with eager Python between. The cost is
fragmentation — lost fusion across the boundary, and one recompile set per
fragment per shape.

```bash
TORCH_LOGS="graph_breaks" ... 2> run.err && python3 scripts/log_triage.py run.err
grep -c 'Graph break in user code at' run.err
```

Reason strings you will actually see, each mapping to a fix pattern:
`torch.* op returned non-Tensor bool` → GB-1 · `Dynamic shape operator`
(`aten.nonzero.default`) → GB-2 · `Data-dependent branching` → GB-3 ·
`call_id with args (NestedUserFunctionVariable(),)` → GB-4 ·
`Attempted to inline function marked as skipped` → GB-5 (usually correct as-is) ·
`assert_size_stride` on an extern op → GB-7.

**Break count is a bad target.** A break at a boundary where the code was already
running eagerly costs approximately nothing — measured, removing the *trace
attempt* made throughput slightly better. Rank breaks by what they fragment: one
inside the repeated block is worth ten at an already-eager library boundary.
Details in [graph-breaks.md](references/graph-breaks.md).

---

## 3. Recompiles

A recompile is a **guard failure**. Ask three questions in order:

1. How many distinct shapes does the process actually see? (Log them per step.)
2. Is the varying dimension being promoted to dynamic, and after how many?
3. **What is the exact guard text?**

The guard text is the answer. Never infer the cause from the symptom — the decoder
table in [recompiles.md](references/recompiles.md) maps each guard form to its fix
(RC-1 content-dependent widths · RC-2 rank flips · RC-3 lazily grown buffers ·
RC-4 Python ints · RC-5 reduce the shape stream).

Two things that surprise people: under the default `dynamic=None` a fragmented
model produces **one burst** of recompiles at the first shape transition and then
nothing; and each new concrete shape still pays a smaller one-time cost with **no
log line at all** — that is Inductor kernel warmup, not a guard failure, so do not
chase it.

---

## 4. Compile time

```bash
TORCH_TRACE=/abs/dir python train.py
python3 scripts/trace_summary.py /abs/dir
```

Compile time can go into Dynamo tracing, AOT joint-graph tracing, Inductor
lowering, code generation, Triton kernel compilation, or autotune search. Read the
phase table, then read **`async_compile.wait` as a share of `code_gen`** — that
one ratio decides whether config levers can help at all:

- **~33%** — you are IR-bound. Levers work. Look at `ir_count / graph_op_count`
  (a 70× blowup is not unusual) and at `restart_reasons`.
- **~90%** — Triton is compiling a few very large kernels. **No Dynamo or Inductor
  flag reaches this**, and more compile workers will not help. Find the kernel in
  `triton_kernel_compile_times_us` — on one workload a single kernel was 94.8% of
  all Triton compilation — and take that work out of the graph (CT-1/CT-2).

The phase table is an **attribution, not a partition**: phases overlap, do not sum
to wall time, and one can exceed total wall time because it is clocked in autotune
workers. [compile-time.md](references/compile-time.md).

---

## 5. Runtime performance of a compiled model

**Module hooks do not work here.** They cover only ~11% of kernel time, because
compiled regions do not nest under Python module scopes. Attributing from
`nn.Module: <Class>_N` rows will mislead you.

Attribute by **correlation id** instead: each GPU `kernel` event's
`args.correlation` maps back to its host-side launch. Then read the generated
name — `triton_poi_fused_*` (pointwise), `triton_per_fused_*` (persistent
reduction), `triton_tem_fused_*` (template) — whose suffix lists the fused aten
ops and is your map back to source. Full procedure in
[tooling.md](references/tooling.md) §4.

Before hand-optimizing a kernel, **A/B the compile mode.** On one workload
`default` → `max-autotune-no-cudagraphs` made an attention backward 2.13× cheaper
with no code change at all. [mode-selection.md](references/mode-selection.md).

---

## 6. Invariants worth knowing before you start

1. Breaks and recompiles are separate mechanisms with separate log channels.
2. One shape change emits one recompile set *per graph-broken frame*; collapsing
   breaks collapses recompiles.
3. `TORCH_LOGS` writes to **stderr**.
4. Zero `Recompiling function` lines proves nothing unless the variable was
   provably set in that process.
5. Recompile **frequency** is a property of the shape stream and `dynamic=`, not
   the compile mode; recompile **cost** scales with the mode (measured 4–6.5× for
   max-autotune).
6. Warm caches buy compile time only — steady-state throughput was identical cold
   vs warm in every measured pair.
7. No Dynamo or Inductor config flag reaches Triton kernel compilation. If
   `async_compile.wait` is ~90% of `code_gen`, stop turning knobs.
8. More compile workers does not help when `async_compile.wait` already dominates
   — the work is already parallel; the lever is generating less code.
9. Exceeding the recompile limit is a **silent** fallback to eager for that frame
   — a performance cliff, not a crash.
10. An experiment whose active path has no positive confirmation in the log can
    silently run the baseline and be mistaken for a refutation.

---

## 7. Config levers, ranked

| lever | measured effect | cost | reach for it when |
|---|---|---|---|
| A/B the **compile mode** | 2.13× on one attention backward; 21.5% end-to-end | 4–6.5× recompile cost | always first — no code change |
| `specialize_float=True` (CT-3) | −18% to −34% compile, −2% to −4% step time | recompile if the float's *value* varies | `restart_reasons` is non-empty |
| `coordinate_descent_tuning=False` (CT-4) | −24% to −54% compile, incl. warm runs | +0.8% step time, +0.6 GiB peak | `*coordinate_descent_tuning` dominates the phase table |
| Compile preprocessing separately (CT-2) | −82% compile **and** faster runtime | a small refactor | one Triton kernel dominates compile |
| Reduce the shape stream (RC-5) | removes recompiles; flattens memory | padding waste | many distinct shapes |
| Regional compilation (CT-5) | recompiles 6→1 per shape, compile ~2× faster | **−13 to −17% throughput, forever** | compile time is the binding constraint |

Every row was measured on **one** workload at **one** problem size. Levers do not
transfer: `specialize_float` was worth −34% at one size and exactly zero at a
larger one. Re-measure on yours.

---

## 8. Scripts

Stdlib only, no install step.

```bash
python3 scripts/log_triage.py run.err            # breaks by reason, recompiles by guard,
                                                 # limit hits, burst timing, a verdict
python3 scripts/trace_summary.py /trace/dir      # phase table, IR blowup, guards, restarts,
                                                 # top Triton kernels, a verdict
python3 scripts/compile_watch.py --selftest      # imports torch; also a known-good fixture
```

`compile_watch.py` also exposes `CompileWatch()` for in-process **per-step**
compile-phase deltas — cumulative totals stop being informative by the sixth
distinct shape.

---

## 9. Measurement rules (short form)

Confirm the path under test actually ran · read knobs from the **resolved** config
· know your noise floor (matched baselines differed 10.8%) · prefer **medians and
kernel time** to means · exclude compiling steps by shape first-appearance, not a
fixed warmup · compare **total across all compiling steps**, never step 1 alone ·
`grep -c` before and after, expecting the targeted lines gone and no others.
Full versions with the incident behind each: [measurement.md](references/measurement.md).

---

## 10. Reference map

| open | when |
|---|---|
| [graph-breaks.md](references/graph-breaks.md) | finding, counting, and deciding whether to fix breaks |
| [recompiles.md](references/recompiles.md) | you have guard text, or shapes vary |
| [compile-time.md](references/compile-time.md) | one compile is enormous |
| [mode-selection.md](references/mode-selection.md) | choosing a mode, CUDA-graph errors, config levers |
| [measurement.md](references/measurement.md) | before trusting any number |
| [fix-patterns.md](references/fix-patterns.md) | the before/after catalogue, cited by ID everywhere |
| [tooling.md](references/tooling.md) | env vars, API index, tlparse, profiling a compiled model |

---

*Quantitative deltas throughout come from measurements on a single transformer
training workload. Treat them as orders of magnitude and re-measure on yours. The
mechanisms are general; the numbers are not.*
