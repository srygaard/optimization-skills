# Compile time

For when a single compile is enormous — minutes to hours — regardless of how many
times it happens. No caching knob makes the *first* compile cheap. You have to
find which phase is spending the time and generate less work for it.

---

## 1. Attribute first

```bash
TORCH_TRACE=/abs/path/to/tracedir python train.py 2> run.err
python3 scripts/trace_summary.py /abs/path/to/tracedir
```

Optionally render the interactive view: `uvx tlparse --no-browser -o out/ <tracefile>`
(or `pip install tlparse`). `trace_summary.py` is the non-interactive path and
prints the decision rules alongside the numbers.

`TORCH_TRACE` costs ~10 s of overhead, reported as `structured_logging_overhead_s`.
**A traced run is not comparable to an untraced baseline.**

---

## 2. Read the phase table

> The phase table is an **attribution, not a partition.** Phases overlap, do not
> sum to wall time, and one phase can exceed `total_wall_time` because it is
> clocked inside autotune worker processes rather than under the frame-compile
> timer.

Never compute percentages against the sum of the column. Compare each phase to
total wall time, and compare phases to each other.

A real table from one workload (a 324.8 s compile):

| phase | s | n |
|---|---|---|
| `CachingAutotuner.coordinate_descent_tuning` | 176.6 | 99 |
| `entire_frame_compile` | 127.3 | 1 |
| `backend_compile` | 93.2 | 3 |
| `create_aot_dispatcher_function` | 81.1 | 6 |
| `aot_trace_joint_graph` | 37.0 | 6 |
| `bytecode_tracing` | 31.9 | 3 |
| `inductor_compile` | 24.0 | 8 |
| `async_compile.wait` | 18.9 | 8 |

Note `coordinate_descent_tuning` (176.6) exceeding `entire_frame_compile` (127.3).
That is the overlap, not an error.

### Phase → action

| dominant phase | means | do |
|---|---|---|
| `entire_frame_compile` / `bytecode_tracing`, with non-empty `restart_reasons` | Dynamo is discarding and redoing work | §3 |
| `create_aot_dispatcher_function` / `aot_trace_joint_graph` | joint-graph tracing | graph size; reduce ops and graph inputs |
| `inductor_compile` / `code_gen` with **low** `async_compile.wait` share (~33%) | codegen bound by IR size | §4 |
| `code_gen` with `async_compile.wait` ~90% | Triton compiling a few very large kernels | §5 — no config flag reaches this |
| `*coordinate_descent_tuning`, `mm_template_autotuning`, `*_autotuning` | kernel search | [mode-selection.md](mode-selection.md) |
| `async_compile.wait` high **and** many kernels | genuinely parallel work | more workers will not help; generate less code |

**`async_compile.wait` as a share of `code_gen` is the single most useful ratio in
the table.** Low means you are IR-bound and config levers can help. High means
Triton kernel compilation dominates, and nothing in `torch._dynamo.config` or
`torch._inductor.config` reaches it.

---

## 3. Restarts: work traced and thrown away

```
restart_reasons              = [TensorifyScalarRestartAnalysis]
dynamo_time_before_restart_s = 63.60
entire_frame_compile_time_s  = 127.27
tensorify_float_attempt = True,  tensorify_float_success = True
```

Three tracing attempts — 21.8 s and 41.8 s discarded, 63.1 s kept. **50% of
frame-compile time was wasted work.** The pass finds float symbols that should be
specialized rather than passed as graph inputs, and restarts analysis to say so.
It succeeds, but only on the third attempt.

Lever: [CT-3](fix-patterns.md#ct-3--specialize_float--true), `torch._dynamo.config.specialize_float = True`.
Read that entry for the cost — it is conditional on *which* float this is.

---

## 4. The IR blowup

```
graph_op_count = 1269,  graph_node_count = 1428  ->  ir_count = 90081
graph_input_count = 144,  guard_count = 2495,  shape_env_guard_count = 82
```

A 1,269-op FX graph expanding to ~90,000 Inductor IR nodes is a **70× blowup**.
That ratio, not the op count, is what codegen time tracks. Nothing about caching
or recompilation touches it.

Watch `graph_input_count` too: 144 inputs is a lot, and each is a guard surface.

`specialize_float=True` cut `ir_count` 90,081 → 30,027 (3×) at an **unchanged**
`graph_op_count` — proof that the blowup ratio moves independently of how big your
model looks.

Rule of thumb: `ir_count / graph_op_count` above ~20× is worth investigating when
codegen dominates.

---

## 5. Superlinear scaling, and the one kernel that is the whole compile

**Diagnostic.** If compile time grows *faster than linearly* in some size
parameter, look for an op whose **generated code** — not its runtime — grows with
that parameter.

Measured on one workload: total compile 3.65× for a 2.13× sequence length, and
`code_gen` alone 10.3× for the same 2.13×, an exponent of ~3.1. Essentially cubic.
`code_gen` was 1,829 s of a 2,196 s step 1 — 83%.

**Find the kernel.** `trace_summary.py` reads `triton_kernel_compile_times_us`:

| kernel | s | share |
|---|---|---|
| `triton_per_fused__to_copy_bitwise_and_eq_gt_lt_sort_sum_12` | **1,506.1** | **94.8%** |
| (24 others) | 82.3 | 5.2% |

One kernel was 94.8% of 1,588 s of Triton compilation. **The fused name lists the
aten ops it contains** — that is your map back to source. Here: `sort`, from an
`argsort` inside a mask-preprocessing helper.

**The mechanism, generically:** `sort` / `argsort` / `topk` lower to an unrolled
bitonic network, so the *generated code* grows with the sorted length — here as
`(length/block)²`. Fused with a large elementwise mask evaluation, that one kernel
dominated everything else.

Do not rely on upstream annotations to save you: the offending op was already
tagged `fallback_to_eager` via `torch.fx.traceback.annotate` in PyTorch, and it
plainly had no effect.

**The fix, and the ranking that surprised people.** Take the expensive
preprocessing out of the model graph, then choose:

| approach | compile | runtime |
|---|---|---|
| [CT-1](fix-patterns.md#ct-1--move-expensive-to-codegen-preprocessing-out-of-the-model-graph) run it eagerly outside | −66.6% | **−8.1% worse** |
| [CT-2](fix-patterns.md#ct-2--compile-that-preprocessing-as-its-own-graph) compile it as its own graph | **−82.4%** | **faster than baseline** |

CT-2 dominates. The refined root cause explains why: the expensive kernel was
never the `argsort` alone, it was the `argsort` **fused with** the full
large-grid mask evaluation. Separated into two graphs, both halves are cheap.
"Compile it separately" is not merely relocating the cost.

---

## 6. Traps in this area

**A coarser block/tile size can look great on step 1 and cost more overall.**
Raising a block size made step 1 look better and total compile across all shapes
**+72% worse**, because inputs shorter than one block could no longer share a
graph with longer ones — producing a second large compile a few steps later.
Always compare *total time across all compiling steps*, never step 1 alone.

**A block size does not change numerics.** A block is skipped only when fully
masked; partial blocks still evaluate the mask element by element. If you thought
a block-size change altered your results, look elsewhere.

**Compile-time levers do not transfer across problem size.** See §7 and
[measurement.md](measurement.md).

---

## 7. Caching

```python
torch.compiler.save_cache_artifacts()      # portable bundle ("megacache")
torch.compiler.load_cache_artifacts(blob)
```
```bash
TORCHINDUCTOR_CACHE_DIR=/scoped/path       # default is keyed only by username
TORCHINDUCTOR_FX_GRAPH_CACHE=1
TORCHINDUCTOR_FORCE_DISABLE_CACHES=1       # to force a cold compile
```

- **Measured cold → warm:** 16–18× on a pathological compile, dropping to **4.5×**
  once the real fix landed. *The better your compile gets, the less caching
  matters.* Caching is not a substitute for §5.
- **Steady-state throughput is unchanged cold vs warm.** Warmth buys compile time
  only. If a "warm" run is faster in steady state, something else changed.
- **Autotune search is largely not cached.** A warm run still paid 191.4 s of
  coordinate-descent tuning (58% of a warm step 1) while `inductor_compile`
  collapsed to 24.9 s. That makes [CT-4](fix-patterns.md#ct-4--coordinate_descent_tuning--false-via-the-mode-preset)
  a saving on *every* run, not just cold ones.
- **Writing back the same byte count you read** is confirmation that nothing new
  compiled. A larger write means a shape or guard changed.
- **Cache-key hygiene.** Key artifacts on the resolved knob manifest — and know
  that adding, removing, or *renaming* a knob invalidates every existing artifact
  and silently forces a cold run. Budget for it or migrate the file deliberately.
- **Scope the Inductor cache directory per run.** The default is keyed only by OS
  username, so every concurrent job on the machine shares it.
- **Warm step 1 is no longer mostly compile.** Once compile time falls, input
  pipeline startup can be 17% of it. Subtract that before comparing.

---

## 8. Measured dead ends — do not re-test

- **Raising `TORCHINDUCTOR_COMPILE_THREADS`** when `async_compile.wait` already
  dominates. The work is already spread across the workers; the lever is
  generating less code.
- **Blaming graph fragmentation** for a slow compile without checking the break
  count first. It was 0 in both jobs where this was suspected.
- **Assuming cache erosion between runs.** Loads and saves were stable to ~21 KB.
- **Assuming a lazily built buffer honors the dtype argument** you passed at
  construction.
