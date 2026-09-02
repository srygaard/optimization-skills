# Recompiles

A recompile is a **guard failure**: the compiled code carried an assumption, the
assumption stopped holding, and Dynamo traced the frame again. The guard text
names the assumption. Read it — never infer the cause.

```bash
TORCH_LOGS="recompiles" python train.py 2> run.err
python3 scripts/log_triage.py run.err --recompiles
```

`recompiles_verbose` gives the full guard expression when the short form is
ambiguous.

---

## 1. Mechanics: three regimes of `dynamic=`

### `dynamic=None` (the default: automatic dynamic shapes)

Dynamo compiles **statically** for the first shape it sees. On the *second*
distinct value of a dimension it marks that dimension symbolic and recompiles —
then generalizes across every later value of it.

Measured on one workload: 23 `Recompiling function` lines in a single ~73 s window
at the first shape transition, then **zero** across four more genuinely new
shapes. Per-transition wall time collapsed from ~120,000 ms to under 1,000 ms.

Why 23 and not 1: **one recompile event per graph-broken subgraph frame.** A
graph-broken model produces `forward`, several
`torch_dynamo_resume_in_forward_at_*`, plus any disabled helpers — each recompiles
independently. With graph breaks removed, the same transition produced 6 events;
with regional compilation, 1. **N recompile lines ≠ N shape changes.**

### `dynamic=False`

A recompile for every distinct shape, forever. Correct when the shape set is tiny
and fixed, and you want maximum specialization.

### `dynamic=True`

Every dimension of every input symbolic from the first trace, whether or not it
ever varies. Structurally different and strictly riskier: it defeats
shape-dependent specialization *and* reaches lowering paths that mixed
concrete/symbolic tracing never touches. Prefer `None` — it promotes only the
dimension that actually varied.

### The cost below Dynamo's visibility

Even after promotion, the **first appearance of each new concrete shape** pays a
smaller one-time tax with **no log line at all** — measured ~350–450 ms under
`default`, seconds under autotuning modes. This is Inductor/Triton per-size kernel
warmup, not a recompile.

Do not chase it as a guard failure. You will find nothing, because
`TORCH_LOGS=recompiles` is silent by design here.

---

## 2. Guard-text decoder

| guard text | means | fix |
|---|---|---|
| `tensor 'X' size mismatch at index N. expected A, actual B` | a traced dim changed. If `N` is **not** the batch or sequence dim, automatic dynamic will not absorb it | [RC-1](fix-patterns.md#rc-1--pad-content-dependent-widths-to-a-fixed-size) |
| `tensor 'X' rank mismatch. expected A, actual B` | rank varies with batch *contents*. Rank is always specialized | [RC-2](fix-patterns.md#rc-2--emit-a-constant-rank) |
| `tensor '...._buffers['name'] ' size mismatch at index 0` | a lazily grown / re-registered internal buffer read inside the graph | [RC-3](fix-patterns.md#rc-3--pre-warm-lazily-grown-internal-buffers) |
| `<obj>.<attr> == N` | a **Python int** reaching the frame, guarded by value | [RC-4](fix-patterns.md#rc-4--quantize-python-ints-that-cross-the-compile-boundary) |
| `___check_obj_id(...)`, closure identity | a fresh object or closure constructed per call | [GB-4](fix-patterns.md#gb-4--give-callables-a-stable-identity-or-stop-tracing-into-the-caller) |
| `tensor 'X' dtype mismatch` | mixed-precision path taken conditionally | make the cast unconditional |
| `tensor 'X' requires_grad mismatch` | eval/train or frozen-parameter switch | expected at the switch; ignore if once |

**The critical distinction for `size mismatch`:** batch and sequence dimensions
get absorbed by automatic dynamic shapes after one recompile. Any *other*
dimension — a width from `max(len(...) for ... in batch)`, a count of matched
tokens, a number of spans — does not, because it varies with batch composition
rather than batch length, and the model usually branches on it. Those recompile
forever. RC-1.

---

## 3. Count cache slots

Recompile lines carry `[frame/slot]` markers — `[0/1]`, `[0/2]`. The slot is how
many compiled versions that frame is holding.

```bash
grep -o '\[[0-9]\+/[0-9]\+\]' run.err | sort -u
```

Measured on one workload: per-frame slot usage topped out at **3** against a
default limit of 8 — once a frame goes dynamic it never recompiles again, so the
limit was never binding.

```python
torch._dynamo.config.recompile_limit               # newer name
torch._dynamo.config.cache_size_limit              # older name; default 8
torch._dynamo.config.accumulated_recompile_limit   # across all frames
```

Both spellings exist in the wild — check which your build has, and set the one it
recognizes.

**The failure mode when you exceed it is a silent fallback to eager for that
frame.** A performance cliff, not a crash, and nothing in a normal log says so
except one line:

```
torch._dynamo hit config.cache_size_limit (8)
   function: 'forward' ...
   last reason: ...
```

`log_triage.py` flags these loudly. Raising the limit to 16 is cheap insurance for
a configuration you have not measured — but if you are hitting 8, the shape stream
is the actual problem (RC-5).

---

## 4. Dynamic shapes: the dead ends

Each of these was tried and measured. They do not work.

**`mark_dynamic` / `mark_static` only affect tensors passed as actual arguments
to the compiled function, before tracing starts.** Intermediates derived *inside*
the traced region — a slice of an already-dynamic input, a mask built in `forward`
— inherit their symbols from that input and cannot be pinned. Verified by dumping
the compiled graph: shapes stayed fully symbolic after the calls.

**Hard `mark_dynamic` on a dimension something downstream specializes** raises
`ConstraintViolationError` with `inferred to be a constant (N)`. That is Dynamo
telling you a real specialization exists, not a bug to work around.

**Soft `maybe_mark_dynamic` does not crash — and does not work either.** On one
workload it silently re-specialized at the same boundary (recompiles unchanged)
*and* measurably regressed throughput. Neither form reaches a Python-int value
guard (RC-4); those are not tensors.

**Marking several tensors as sharing one symbol** because their sizes happen to
match numerically produces `ConstraintViolationError` from the shape-guard pass.
That is a harness bug, not a finding about dynamism.

**Symbolic shapes can fail only under autotuning modes.** The same input that
compiles fine under `mode="default"` can hard-fail during lowering under
`max-autotune*` once a size becomes symbolic — some of these are upstream bugs
fixed in later PyTorch releases. If a symbolic-shape error appears, A/B against
`default` and against a newer PyTorch **before** assuming your model is at fault.

---

## 5. Reduce the shape stream instead

Usually the right answer, and rarely the first one tried.

| approach | shapes | padding waste | effort |
|---|---|---|---|
| one global shape | 1 | pays the max on every step | trivial |
| length buckets | N (pick N ≈ 5–8) | moderate | small |
| token-budgeted batches (vary rows, fix bucket lengths) | still N | lower, and flattens memory and step time | small |
| packing (concatenate items, block-diagonal masking) | 1 | ~none | large rewrite |
| `dynamic=True` | — | none | removes one-time recompiles only |

Two notes that decide between these:

- **Varying the row count within a bucket introduces no new shapes** if you keep
  the bucket lengths fixed — token-budgeted batching bought +22.5% throughput on
  one workload at an unchanged compile budget.
- **`dynamic=True` addresses recompiles only.** It does nothing for per-step time
  or memory variance, and recompiles are a one-time cost. It is the option that
  looks most like a fix and delivers least.

Also quantize any Python-int field that crosses the boundary to the *same* bucket
set (RC-4), or you have moved the problem, not solved it.

**Guard the bucket invariant.** If anything downstream can produce an item longer
than your top bucket — a tokenizer that truncates a prefix then appends suffix
tokens, say — you get an unbounded extra shape and a mystery recompile. Assert the
bound where the buckets are computed, not where they are used.

---

## 6. Related levers

- **Regional compilation** cuts recompiling frames per transition (6 → 1 measured)
  but cost ~13–17% steady-state throughput. See
  [CT-5](fix-patterns.md#ct-5--regional-compilation-via-modulecompile) and
  [mode-selection.md](mode-selection.md).
- **Compile mode changes recompile *cost*, not frequency** — 4–6.5× more expensive
  per recompile under `max-autotune` on one workload, same trigger and same count.
  (Caveat: a mode change can *prevent* dynamic promotion from succeeding at all,
  which changes frequency indirectly. Verify with the log rather than assuming.)
- **Fix graph breaks first.** Fewer frames means fewer recompile events per shape.
