# Tooling

---

## 1. Environment variables

| var | effect | cost |
|---|---|---|
| `TORCH_LOGS="graph_breaks,recompiles"` | the two diagnostics you want by default | near-free |
| `TORCH_LOGS="+dynamo"` | full Dynamo tracing log | very verbose; one repro run |
| `TORCH_LOGS="recompiles_verbose"` | full guard expressions, not the short form | moderate |
| `TORCH_LOGS="guards"` | every guard installed | very verbose |
| `TORCH_LOGS="output_code"` | the generated Triton/C++ | large |
| `TORCH_LOGS="perf_hints"` | Inductor's own suggestions | free |
| `TORCHDYNAMO_VERBOSE=1` | full tracebacks on breaks and errors | free |
| `TORCH_TRACE=/abs/dir` | structured JSONL trace for `tlparse` | ~10 s |
| `TORCH_COMPILE_DEBUG=1` | full debug dump directory | heavy |
| `TORCHINDUCTOR_CACHE_DIR=/path` | scope the codegen cache | — |
| `TORCHINDUCTOR_COMPILE_THREADS=N` | parallel kernel compilation | — |
| `TORCHINDUCTOR_FX_GRAPH_CACHE=1` | FX graph cache | — |
| `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1` | force a cold compile | — |

Values combine with commas; a `+` prefix raises verbosity (`+dynamo`).

### Gotchas

- **Everything goes to stderr**, not stdout. Capture `2> run.err` or `2>&1`.
- **Never export `TORCH_LOGS` as an empty string.** Present-but-empty has crashed
  `import torch` on some builds with
  `AttributeError: 'dict' object has no attribute 'get_log_level_pairs'`. Set it
  or unset it — do not forward an empty value.
- **If the process runs inside an environment-filtering wrapper** (container
  runtime, job launcher, curated subprocess `env`), a newly added diagnostic
  variable is silently dropped unless explicitly forwarded. Verify by printing it
  from inside the process.
- **`TORCHINDUCTOR_CACHE_DIR`'s default is keyed only by OS username**, so every
  concurrent job under the same account shares one cache directory. Scope it per
  run to rule out cross-job collisions by construction.
- **Each rank writes its own trace file.** Analyze rank 0 unless you are chasing a
  rank-divergence bug.

---

## 2. In-process API index

```python
# diagnosis
torch._dynamo.explain(fn)(*args)          # .graph_break_count, .break_reasons, .graphs
torch._dynamo.utils.counters               # {"graph_break": {reason: n}, "stats": {...}}
torch._dynamo.utils.compile_times(repr="csv", aggregate=True)
torch._dynamo.reset()                      # drop all compiled state and guards

# dynamo config
torch._dynamo.config.recompile_limit           # newer name
torch._dynamo.config.cache_size_limit          # older name; default 8
torch._dynamo.config.accumulated_recompile_limit
torch._dynamo.config.specialize_float           # default False -> CT-3
torch._dynamo.config.automatic_dynamic_shapes   # default True
torch._dynamo.config.assume_static_by_default
torch._dynamo.config.error_on_graph_break       # newer builds
torch._dynamo.config.capture_scalar_outputs     # allow .item() in-graph
torch._dynamo.config.suppress_errors, .verbose

# inductor config
torch._inductor.config.coordinate_descent_tuning   # -> CT-4
torch._inductor.config.max_autotune, .max_autotune_gemm
torch._inductor.config.compile_threads
torch._inductor.config.fx_graph_cache, .force_disable_caches
torch._inductor.config.triton.cudagraphs
torch._inductor.list_mode_options(mode)            # expand a preset -> options=

# shaping the compile
torch.compiler.disable                     # opaque eager boundary   -> GB-4/GB-5
torch.compiler.allow_in_graph              # opaque but IN graph     -> GB-7
torch.library.custom_op, .register_fake    # ditto, properly         -> GB-7
torch.compiler.cudagraph_mark_step_begin   #                         -> RC-6
torch.compiler.set_stance, .is_compiling
torch._dynamo.mark_dynamic, .maybe_mark_dynamic, .mark_static
torch.cond                                 # last resort for GB-3
module.compile(...)                        # IN PLACE                -> CT-5

# caching
torch.compiler.save_cache_artifacts(), .load_cache_artifacts(blob)
```

Names move between PyTorch versions — `cache_size_limit` became `recompile_limit`,
and metric field names in traces shift too. Check with `hasattr` rather than
assuming, and support both spellings in anything you commit.

---

## 3. tlparse

```bash
TORCH_TRACE=/abs/dir python train.py
uvx tlparse --no-browser -o out/ /abs/dir/dedicated_log_torch_trace_rank_0_*.log
# or: pip install tlparse
```

The output directory contains `index.html`, `chromium_events.json`,
`failures_and_restarts.html`, and one subdirectory per compile id with the FX
graph, the generated code, and the metrics.

`scripts/trace_summary.py` is the non-interactive path — it reads the raw
`TORCH_TRACE` log directly (or a tlparse dir with `--tlparse`) and prints the
phase table, the IR blowup ratio, guards, restarts, top Triton kernels, and a
verdict.

### Trace fields that matter

`restart_reasons` · `dynamo_time_before_restart_s` · `entire_frame_compile_time_s`
· `graph_op_count` vs `ir_count` (the blowup ratio) · `graph_input_count` ·
`guard_count` · `shape_env_guard_count` · `cache_size` ·
`triton_kernel_compile_times_us` · `structured_logging_overhead_s` · and the
chromium phases `code_gen`, `async_compile.wait`, `inductor_compile`,
`backend_compile`, `create_aot_dispatcher_function`, `bytecode_tracing`,
`*_autotuning`, `CachingAutotuner.coordinate_descent_tuning`.

---

## 4. Profiling a compiled model

**Module hooks do not work under `torch.compile`.** They cover only ~11% of kernel
time, because compiled regions do not nest under Python module scopes — Inductor
has fused everything else into opaque kernels with no relationship to your class
hierarchy. Do not attribute from `nn.Module: <Class>_N` rows.

The procedure that does work:

1. `torch.profiler` with `with_stack=True`. Capture 2 active steps after warmup,
   at **one fixed input shape**.
2. Export the Chrome trace and read `traceEvents` with stdlib `json` — the whole
   analysis is ~40 lines. Do not reach for a dependency.
3. **Attribute by correlation id.** Each GPU `kernel` event's `args.correlation`
   maps back to its host-side `cuda_runtime` launch event. That mapping is the
   only reliable link from a fused kernel to the code that launched it.
4. **Identify a specific kernel** by (a) its input tensor shapes — `[B,H,S,D]`
   patterns are recognizable; (b) a grid dimension containing a known product such
   as batch×heads; or (c) a launch count equal to `blocks × steps`. "No other
   kernel invoked exactly N times matches" is a valid identification argument.
5. **Read the generated name.** `triton_poi_fused_*` is pointwise,
   `triton_per_fused_*` a persistent reduction, `triton_tem_fused_*` a template.
   The suffix lists the fused aten ops — that is your map back to source.
6. **Separate vendor-library GEMM calls from Inductor's own Triton templates** in
   the kernel table. Under autotuning modes Inductor migrates some GEMMs from the
   vendor BLAS library to its own templates, so a mode change moves the
   *category*, not just the time. A table that lumps them together will hide it.
