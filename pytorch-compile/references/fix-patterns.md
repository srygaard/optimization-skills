# Fix patterns

Numbered catalogue. Every other file in this skill cites these IDs. Each entry:
symptom → mechanism → before/after → how to verify → known cost.

Snippets are minimal and generic. Adapt names; the shape of the fix is the point.

---

## Graph breaks

### GB-1 — Hoist constant capability/config queries out of `forward`

**Symptom.** `Graph Break Reason: torch.* op returned non-Tensor bool call_function <name>`,
often at many call sites at once.

**Mechanism.** A device/environment query returns a plain Python `bool` or `int`.
Dynamo cannot put a non-Tensor result of a `torch.*` op into the graph, so it
breaks — once per call site, every call.

```python
# before  (8 call sites -> 8 graph breaks)
def forward(self, x):
    use_bf16 = self.cfg.bf16 and torch.cuda.is_bf16_supported()

# after
def __init__(self, ...):
    self.bf16_supported = torch.cuda.is_bf16_supported()   # once, at build time
def forward(self, x):
    use_bf16 = self.cfg.bf16 and self.bf16_supported
```

**Verify.** `grep -c is_bf16_supported run.err` → `0`, and the total
`Graph break in user code at` count drops by exactly the number of call sites.

**Cost.** None. The value cannot change during a run.

**Generalize.** Any function returning a Python scalar from a device, environment,
or version query: `torch.cuda.get_device_capability`, `torch.cuda.is_available`,
`is_bf16_supported`, `torch.__version__` comparisons, `dist.get_rank()`.

---

### GB-2 — Replace boolean-mask scatter with a dense mask + `torch.where`

**Symptom.** `Graph Break Reason: Dynamic shape operator` naming `aten.nonzero.default`.

**Mechanism.** A **boolean-mask fancy-index scatter** lowers to `aten.nonzero`,
whose output size is data-dependent. This is the key insight most people miss:
*the values being indexed being data-dependent was never the problem.* The
indexing form alone produces `nonzero`, no matter what is stored.

```python
# before
dst[batch_idx[has_t], pos_idx[has_t]] = src[has_t].float()

# after — dense per-position mask, no size depends on data
pos_mask = (seq_idx == pos_idx.unsqueeze(1)) & has_t.unsqueeze(1)
dst = torch.where(pos_mask, src.unsqueeze(1).expand_as(dst), dst)
```

**Verify.** `grep -c nonzero run.err` → `0`. Check numerical equivalence
separately, covering the empty-mask and boundary-index cases — this rewrite is
easy to get subtly wrong.

**Cost.** Computes over the full dense grid instead of only selected positions.
Usually irrelevant next to the fragmentation it removes; measure if the grid is
large.

---

### GB-3 — Delete data-dependent branches that only skip compute

**Symptom.** `Graph Break Reason: Data-dependent branching`, with Dynamo's own
hint: *"this graph break is fundamental — it is unlikely that Dynamo will ever
be able to trace through your code… use `torch.cond`"*.

**Mechanism.** `if some_tensor.any():` forces a host sync and a Python branch.
Dynamo's hint is accurate about the *branch* — and irrelevant if the branch is
unnecessary. Very often the `.any()` is a compute-skipping optimization guarding
code whose result is already masked downstream.

```python
# before
if property_mask.any():
    embeds = compute_property_embeds(values)
    x = torch.where(property_mask, embeds, x)

# after — the where already handles "nothing matched" correctly
embeds = compute_property_embeds(values)
x = torch.where(property_mask, embeds, x)
```

**Verify.** `grep -c 'Data-dependent branching' run.err` → `0`.

**Cost.** The guarded block now runs unconditionally. Check first that it is safe
on an all-false mask (no division by a zero count, no `None` inputs) and that its
cost is small.

**When this does not apply.** If the skipped compute is genuinely expensive or
unsafe, reach for `torch.cond`. Do that second, not first.

---

### GB-4 — Give callables a stable identity, or stop tracing into the caller

**Symptom.** `Graph Break Reason: call_id with args (NestedUserFunctionVariable(),)`
inside `inspect.py`, reached from a library function you called.

**Mechanism.** The library runs reflection — `inspect.signature(fn)`, which calls
`inspect.unwrap`, which calls `id(f)` — on a callable you passed. A **fresh nested
closure** is a `NestedUserFunctionVariable` with no stable identity, and Dynamo
cannot trace `id()` on it.

```python
# before — fresh closure every forward call
def build(self, idx_a, idx_b):
    def mask_mod(b, h, q, kv):
        return ...idx_a...idx_b...
    return library_fn(mask_mod, ...)

# step 1 — module-level function bound with functools.partial
def _mask_mod(idx_a, idx_b, b, h, q, kv):
    return ...
def build(self, idx_a, idx_b):
    return library_fn(functools.partial(_mask_mod, idx_a, idx_b), ...)
```

**Step 1 is often not enough** — record this honestly, it saves the next person a
day. The reflection then trips on the *type* of the partial instead:

```
Graph break in user code at .../inspect.py:328
Graph Break Reason: hasattr UserDefinedClassVariable __get__
```

…now failing in `_signature_from_callable` → `_signature_is_builtin` →
`ismethoddescriptor` → `hasattr(type(obj), "__get__")`. The library's reflection
is not traceable for *any* callable shape. Chasing the next one is diminishing
returns.

```python
# step 2 — make the whole wrapper an opaque eager boundary
@torch.compiler.disable
def build(self, idx_a, idx_b):
    return library_fn(...)
```

**Verify.** The N messy library-internal breaks collapse to exactly 1, at your
call site, with reason `Attempted to inline function marked as skipped`.

**Cost.** Measured: **none, and slightly negative** — throughput *improved*. The
wrapper was running eagerly either way; Dynamo's failed trace attempt on every
call was pure overhead. See GB-5.

---

### GB-5 — Accept the break

**Symptom.** `Attempted to inline function marked as skipped`, plus
*"Dynamo developers have intentionally marked that the function…"*.

One expected break at one known call site, resuming immediately after, is a fine
outcome. A break at a boundary where the code was already running eagerly costs
approximately nothing.

**Measured:** the presence or absence of one such break did not move steady-state
throughput beyond run-to-run variance in three matched pairs. Removing the *trace
attempt* was slightly faster than leaving Dynamo to fail at it repeatedly.

**Decide by measurement, not by break count.** A log with 16 breaks that all sit
at one already-eager library boundary is healthier than a log with 3 breaks
splitting your transformer block into 4 fragments.

**Corollary — the deliberate non-fix.** When the break is an upstream limitation
rather than your bug, consider leaving the workaround *off* so a future PyTorch
release fixes it for you. Only after measuring that the break is free.

---

### GB-6 — Hoist host–device syncs out of per-block code

**Symptom.** Kernel time looks fine; wall clock does not. Long blocking gaps in a
profile at `.item()`, `.cpu()`, `.tolist()`, `bool(tensor)`, or `nonzero`.

**Mechanism.** These are data-dependent syncs: they drain the entire queued
device stream. Measured on one workload: a single `all_reduce(...).item()` in a
per-step callback blocked for 51.6 s and 22.9 s on individual calls — not because
the reduce was slow, but because it waited on a backlog of async work. Running
such a call once per transformer block instead of once per step multiplies it.

**Fix.** Compute the layer-invariant plan **once per step** and thread the result
through the blocks. Accumulate metrics as tensors and sync at most once per
logging interval, never per block.

---

### GB-7 — Keep an opaque kernel *in* the graph

**Symptom.** A hand-written or vendor kernel graph-breaks once per call site,
fragmenting the surrounding pack/cast/`where` glue into eager code.

**Mechanism.** Dynamo cannot trace the kernel's internals, so it breaks around
the whole region. But you do not need it traced — you need it *opaque and
in-graph*, so everything around it still compiles.

```python
torch.compiler.allow_in_graph(my_kernel)          # simplest

# or, properly, with a fake/meta implementation
@torch.library.custom_op("mylib::my_kernel", mutates_args=())
def my_kernel(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor: ...

@my_kernel.register_fake
def _(q, k):
    return q.new_empty(q.shape)      # sizes AND strides must match reality
```

**Precondition, and the common failure.** The fake/meta kernel's declared sizes
**and strides** must match what the real kernel returns. A widely-shipped kernel
was found declaring a non-transposed layout for a value the real kernel returns
transposed; in-graph tracing aborted with `assert_size_stride`. A wrong fake
kernel is a real, frequent bug — when tracing fails on an extern op, suspect the
meta implementation before your own code.

**Payoff, measured on one workload.** Restructuring so that *only* the extern
call was opaque, with the surrounding glue compiled in, took a per-block break
worth 236 ms/step down to 65.1 ms/step — 3.6×.

---

## Recompiles

### RC-1 — Pad content-dependent widths to a fixed size

**Guard.** `tensor 'indices' size mismatch at index 1. expected 2, actual 4`

**Mechanism.** A dimension computed from batch *contents* — `max(len(d["x"]) for d
in batch)` — is neither the batch nor the sequence dimension, so automatic dynamic
shapes will not absorb it. If the model branches on that width, it is a traced
constant and every distinct value is a separate compile.

**Fix.** Pad every row to a fixed width, and make the padding **inert** rather
than special-cased: emit a degenerate range (`low == high`) that every consumer
already gates on (`if high > low`). No new branch, no numerical change.

**Measured on one workload** (before → after, both warm): recompiles 2 → 0, run
mean 520.6 → 313.1 ms/step, std 4503.5 → 53.1 ms, the worst step 101,417 ms → 161 ms.

---

### RC-2 — Emit a constant rank

**Guard.** `tensor 'loss_mask' rank mismatch. expected 3, actual 2`

**Mechanism.** A flag like `is_vector = any(...)` flips a tensor between two ranks
depending on batch contents. Rank is always specialized — dynamic shapes never
help here.

**Fix.** Always emit the higher rank. Verify it is numerically a no-op: typically
the scalar producer emits zeros and every consumer already broadcast the lower
rank.

---

### RC-3 — Pre-warm lazily grown internal buffers

**Guard.** `tensor '...rotary_emb._buffers['cos_cached'] ' size mismatch at index 0. expected 2048, actual 4356`

**Mechanism.** Positional/rotary caches are built at a config default
(`max_position_embeddings`) and **re-registered at a larger shape** the first time
a longer input arrives. Those buffers are read inside the compiled graph, so the
size is guarded, and regrowing discards every graph traced before it.

**Fix.** Size the buffer to the maximum you will ever feed, and force it there
**immediately before `torch.compile`**. Unit-test the pre-warm — it is a silent
failure if the attribute name changes.

```python
prewarm_positional_cache(model, max_len=max(all_shape_plans))
model = torch.compile(model, ...)
```

**Two properties make this worse than one wasted recompile**, and are why it is
worth fixing rather than tolerating:

- **Order-dependent.** The buffer only grows to the longest input seen *so far*.
  A run that starts short compiles everything against the small size and loses it
  all at once when the first long input arrives.
- **It poisons the on-disk cache.** Artifacts keyed on a graph guarding the small
  size can never be reused. One run spent 105 minutes compiling and saved nothing.

**Measured on one workload:** the same new-shape event went from *never completed
in 54 minutes* to *344 ms*.

---

### RC-4 — Quantize Python ints that cross the compile boundary

**Guard.** `plan.max_rows == 154` — a bare value guard on a Python attribute.

**Mechanism.** A plain Python `int` reaching the compiled frame is guarded **by
value**, not by symbol. `mark_dynamic` cannot touch it; it is not a tensor.

**Fix.** Round int fields to the same small bucket set you round shapes to. If
the field can take any of 500 values, quantize it to 6.

---

### RC-5 — Reduce the shape stream

The option nobody tries first. Round every varying dimension to a small fixed set
of buckets so the compiler sees N shapes instead of hundreds. See
[recompiles.md](recompiles.md) for the options table and its padding trade.

Note that recompiles are a **one-time** cost and are frequently the wrong thing to
optimize for. Reduce the shape stream when it also flattens per-step memory and
latency; do not rewrite your input pipeline solely to remove a burst you pay once.

---

### RC-6 — Clone anything you retain across steps under CUDA graphs

**Error.**
```
RuntimeError: Error: accessing tensor output of CUDAGraphs that has been
overwritten by a subsequent run.
```

**Mechanism.** CUDA graphs reuse output buffers across replays. Anything you keep
past the step that produced it must be a real copy. The trap:

```python
# before — a NO-OP VIEW when value is already float32; the buffer gets reused
stats.append(value.detach().float().reshape(()))

# after
stats.append(value.detach().float().reshape(()).clone())
```

`.float()`, `.reshape()`, `.detach()`, `.to(dtype)` are all no-ops when they
change nothing, and a no-op returns a view.

**Make it unconditional**, not gated on the compile mode: every cudagraph-enabling
mode hits this, including plain `max-autotune`. The cost is one scalar copy per
step. Alternative: `torch.compiler.cudagraph_mark_step_begin()` before each model
invocation.

---

## Compile time

### CT-1 — Move expensive-to-codegen preprocessing out of the model graph

**Symptom.** One generated kernel dominates compile time (see
[compile-time.md](compile-time.md)), and its fused name names ops belonging to a
mask/index/plan structure rather than to your model's math.

**Fix.** Build the auxiliary structure *before* entering the model and pass it in.
Keep the model able to build one when the caller supplies none — no graph break
either way.

```python
aux = build_aux(...)                 # outside the compiled region
out = model(x, aux=aux)
```

**Measured on one workload:** step-1 compile 2,196,726 → 734,343 ms (−66.6%);
`async_compile.wait` 1,530.7 → 34.2 s (−98%). **But runtime got 8.1% worse** —
eager rebuild materialized a large dense grid every step. Which is why CT-2 exists.

---

### CT-2 — Compile that preprocessing as its own graph

Better than CT-1 on **both** axes, and it contradicted the prediction that
separate compilation would merely relocate the cost.

```python
build_aux_compiled = torch.compile(build_aux)   # its own small graph
aux = build_aux_compiled(...)
out = model(x, aux=aux)
```

**Measured on one workload:** step-1 compile 2,196,726 → 386,096 ms (−82.4%),
*and* steady state faster than the unmodified baseline (227.2 vs 235.3 ms/step),
with byte-identical peak memory.

**Why it works.** The expensive kernel was never the costly op alone — it was that
op **fused with** the full large-grid mask evaluation. Separated into two graphs,
both halves are cheap. Prefer CT-2; CT-1 is dominated by it.

---

### CT-3 — `specialize_float = True`

```python
torch._dynamo.config.specialize_float = True    # default False
```

Removes the `TensorifyScalarRestartAnalysis` restart — on one workload, ~50% of
Dynamo frame-compile time that was traced and thrown away. Also shrank `ir_count`
90,081 → 30,027 (3×) at an unchanged `graph_op_count`.

**Cost.** A recompile whenever the specialized float's **value** changes.
**Identify which float it is before enabling.** A config constant (a loss weight,
an embedding scale) makes this free. A per-step value (a sampled timestep) makes
it a bad trade.

**Do not over-generalize this one.** It bought −18% and −34% compile at one
problem size and **exactly zero** at a larger one, where the cost had moved into
Triton kernel compilation, which no Dynamo flag reaches.

---

### CT-4 — `coordinate_descent_tuning = False`, via the mode preset

**The API trap:** setting `torch._inductor.config.coordinate_descent_tuning = False`
globally is **overwritten by the mode preset at compile time**. Expand the preset
and pass it explicitly:

```python
opts = torch._inductor.list_mode_options(mode)   # dict for "max-autotune" etc.
opts["coordinate_descent_tuning"] = False
model = torch.compile(model, options=opts)       # options=, NOT mode=
```

**Measured on one workload:** −43% and −54% cold compile at a short shape, −24% at
a long one, and it removes ~190 s from *warm* step 1 as well, because this search
is largely not cached. Cost: +0.75–0.9% step time, +0.62 GiB peak memory
(coordinate descent picks smaller-working-set kernels). Template autotuning still
runs — the kernel search is intact.

---

### CT-5 — Regional compilation via `module.compile()`

Compile the repeated block rather than the whole model.

```python
for layer in model.layers:
    layer.compile(mode=mode)        # IN PLACE

# NEVER: layer = torch.compile(layer)
# that wraps it in OptimizedModule and inserts "_orig_mod." mid-key into
# state_dict, breaking every checkpoint save/load path you have.
```

**Measured on one workload:** recompiling frames per shape transition 6 → 1, graph
breaks → 0, first compile ~245–445 s → ~178 s, per-shape recompile ~200–260 s → ~127 s.

**And it was reverted.** A consistent ~13–17% relative steady-state throughput
loss across three independently compiled runs (non-overlapping ranges), attributed
to lost cross-block fusion. Recompiling is one-time; the throughput loss is
forever. **Measure both axes before adopting.**
