# Compile mode and config levers

---

## 1. The four modes

| mode | adds | compile cost | when |
|---|---|---|---|
| `default` | Inductor codegen, minimal search | baseline | varying shapes, iterating, debugging |
| `reduce-overhead` | CUDA graphs | + capture per shape | small models, fixed shapes, launch-bound |
| `max-autotune-no-cudagraphs` | exhaustive template search + coordinate descent | 4–6.5× recompiles | fixed-ish shapes, throughput matters |
| `max-autotune` | the above **plus** CUDA graphs | + capture per shape | one shape, steady state |

**Measured on one workload:** `max-autotune` cost ~28% more on a cold compile than
`default` but ran **21.5% faster** in steady state (235.3 vs 299.7 ms/step). At the
kernel level, switching `default` → `max-autotune-no-cudagraphs` made one attention
backward **2.13× cheaper with no code change**.

That last number is the practical point: `default` barely tunes generated
templates, so **A/B-ing the compile mode is the cheapest possible kernel
optimization** — no rewrite, no new code. Try it before hand-routing anything.

The trade is entirely on the compile axis. Recompiles measured ~54–58 s per shape
under `default` and ~330–384 s under `max-autotune-no-cudagraphs`.

---

## 2. Setting Inductor options under a mode preset — the API trap

Setting the global config is **silently overwritten by the mode preset at compile
time**:

```python
torch._inductor.config.coordinate_descent_tuning = False   # DOES NOTHING
model = torch.compile(model, mode="max-autotune")          # preset wins
```

Expand the preset and pass it explicitly:

```python
opts = torch._inductor.list_mode_options(mode)   # dict for the preset
opts["coordinate_descent_tuning"] = False
model = torch.compile(model, options=opts)       # options=, NOT mode=
```

`mode=` and `options=` are mutually exclusive. This is why "I disabled it and
nothing changed" is such a common report — confirm the phase actually disappeared
from the trace, not just that you set the attribute.

---

## 3. The two levers, both axes measured

Full 2×2 on one workload at one problem size, `max-autotune-no-cudagraphs`. `cdt`
= `coordinate_descent_tuning`, `sf` = `specialize_float`:

| cdt | sf | cold step 1 | ms/step | peak GiB |
|---|---|---|---|---|
| on | off | 666,997 ms | 202.3 | 13.63 |
| on | on | 546,002 ms | 198.0 | 13.63 |
| off | off | 379,521 ms | 204.2 | 14.25 |
| off | **on** | **252,044 ms** | 199.5 | 14.25 |

Together **−62.2% compile**, better than independence would predict. Both edges
reproduced.

| lever | compile | runtime | memory | risk |
|---|---|---|---|---|
| `coordinate_descent_tuning=False` ([CT-4](fix-patterns.md#ct-4--coordinate_descent_tuning--false-via-the-mode-preset)) | −43% / −54% at a short shape, −24% at a long one; also −190 s off *warm* step 1 | +0.75–0.9% step time | +0.62 GiB peak | none structural — it selects kernels with smaller working sets |
| `specialize_float=True` ([CT-3](fix-patterns.md#ct-3--specialize_float--true)) | −18% / −34% at a short shape, **zero** at a long one | −2.1% to −3.9% step time (a *win*) | none | a recompile when the float's **value** changes |

### The lever that stopped working

`specialize_float` bought a large compile win at one problem size and **exactly
zero** at a 2.1× larger one. The reason is visible in the trace:
`async_compile.wait` as a share of `code_gen` went from 33% to 91%. At the larger
size the cost had moved into Triton kernel compilation, and **no Dynamo or
Inductor config flag reaches that.**

Meanwhile its *runtime* win got larger at the bigger size (−3.9% vs −2.1%).

**Always re-measure a lever at the size you actually run.** A projection from a
smaller shape is not evidence.

---

## 4. CUDA graphs

Two distinct failure modes, both worth recognizing on sight.

### Output buffers get reused across replays

```
RuntimeError: Error: accessing tensor output of CUDAGraphs that has been
overwritten by a subsequent run. ... clone the tensor outside of torch.compile()
or call torch.compiler.cudagraph_mark_step_begin() before each model invocation.
```

Anything you retain past the step that produced it must be a real copy. The trap
is that `.float()`, `.reshape()`, `.detach()` and `.to(dtype)` return **views**
when they change nothing. See [RC-6](fix-patterns.md#rc-6--clone-anything-you-retain-across-steps-under-cuda-graphs).

### Capture cost scales with the shape stream

Capture happens per shape. Measured on one workload: a single new shape stalled
for **844 s (~14 minutes)** while it recompiled, autotuned, and captured, with 94
autotune events and a `UserWarning: The CUDA Graph is empty` alongside.

**When shapes vary, `max-autotune-no-cudagraphs` is the better of the two.** Plain
`max-autotune` never reached steady state on that workload.

---

## 5. Crashes during autotuning of a new shape

A device memory-access fault raised **after** the autotuner has already filtered
to candidates that fit is a **codegen bug in a generated kernel**, not resource
exhaustion.

Distinguish it from the noise that precedes it. Lines like

```
OutOfResources: out of resource: shared memory, Required: N, Hardware limit: 65536
PassManager::run failed
```

are the autotuner **correctly rejecting** oversized or invalid candidates and
moving on. That is normal — a healthy run logged 478 of them; a crashing run
logged 12. Their count is not a signal.

Practical response: the fault reproduces on the first compile of specific shapes,
so shrink or drop that shape, or run that one configuration under `mode="default"`
for a clean comparison. Not comparable in absolute terms, fine for relative
shape-to-shape work.

---

## 6. Regional compilation

Compile the repeated block instead of the whole model. See
[CT-5](fix-patterns.md#ct-5--regional-compilation-via-modulecompile) for the code
and the `state_dict` hazard.

**Measured on one workload:** recompiling frames per shape transition 6 → 1, graph
breaks → 0, first compile ~245–445 s → ~178 s, per-shape recompile ~200–260 s → ~127 s.

**And it was reverted.** A consistent ~13–17% relative steady-state throughput
loss across three independently compiled runs with non-overlapping ranges,
attributed to lost cross-block fusion.

> Recompiling is one-time; the throughput loss is forever.

Adopt it when compile time is the binding constraint (short jobs, many shapes,
interactive iteration) and you have measured the throughput cost on your model.
Not as a default.

---

## 7. Isolating compile cost in a microbenchmark

```python
for shape in shapes:
    torch._dynamo.reset()          # each shape gets an isolated COLD compile
    fn = torch.compile(f, mode=mode)
    warm_s = time_it(fn, shape)    # report separately
    steady_ms = median(...)        # never the mean
```

Without the reset, guards and recompiles accumulate across unrelated shapes and
every shape after the first is measuring something different.

**A microbenchmark is necessary but not sufficient for choosing a mode.** A bare
`torch.compile(torch.mm)` exercises almost none of Dynamo. Two builds that rated
fine on an isolated GEMM sweep both died with an internal Dynamo error on a real
model. Validate the mode end to end before committing to it.
