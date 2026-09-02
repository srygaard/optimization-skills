# Graph breaks

A graph break splits one compiled region into two, with eager Python in between.
The cost is not the break itself — it is fragmentation: fewer fusion
opportunities across the boundary, and **one recompile set per fragment per
shape** instead of one for the whole model.

---

## 1. See them

```bash
TORCH_LOGS="graph_breaks" python train.py 2> run.err      # near-free
TORCHDYNAMO_VERBOSE=1 TORCH_LOGS="graph_breaks" ...       # full tracebacks
python3 scripts/log_triage.py run.err
```

Output is on **stderr**. `TORCH_LOGS=recompiles` alone shows **zero** breaks even
when there are dozens — they are separate log channels.

In-process, when you cannot set environment variables:

```python
expl = torch._dynamo.explain(fn)(*args)
expl.graph_break_count, expl.break_reasons, expl.graphs
torch._dynamo.utils.counters["graph_break"]     # {reason: count}
```

To turn the *next* break into an exception instead of a silent split:
`torch.compile(..., fullgraph=True)`, or
`torch._dynamo.config.error_on_graph_break = True` on newer builds.

**The traceback is the whole point.** A break reported inside `inspect.py` or a
library file is usually reached from *your* call site — and only the full
traceback shows which one. One break assumed for weeks to be library plumbing
turned out to originate three frames up in the model's own code, and was fixable.
Turn on `TORCHDYNAMO_VERBOSE` before concluding a break is out of scope.

---

## 2. Count them reproducibly

This protocol is what makes a fix defensible:

```bash
grep -c 'Graph break in user code at' before.err        # total
grep -c '<the specific reason string>' before.err       # the ones you target
# ... apply fix, rerun with the SAME TORCH_LOGS ...
grep -c '<the specific reason string>' after.err        # expect 0
grep -c 'Graph break in user code at' after.err         # expect before - N
```

A good fix removes the targeted lines **and no others**. Two traps:

- Adding lines to a file shifts every later break's reported line number by the
  same delta. That is not a new break.
- Fixing one noisy break *unmasks* others that were always there. They are
  pre-existing, not introduced.

---

## 3. The catalogue

| reason text in the log | mechanism | fix |
|---|---|---|
| `torch.* op returned non-Tensor bool call_function <name>` | device/capability query returning a Python scalar inside `forward` | [GB-1](fix-patterns.md#gb-1--hoist-constant-capabilityconfig-queries-out-of-forward) |
| `Dynamic shape operator` on `aten.nonzero.default` | boolean-mask fancy-index **scatter** — lowers to `nonzero` regardless of the values | [GB-2](fix-patterns.md#gb-2--replace-boolean-mask-scatter-with-a-dense-mask--torchwhere) |
| `Data-dependent branching` (+ the *"fundamental … use `torch.cond`"* hint) | `if some_tensor.any():` | [GB-3](fix-patterns.md#gb-3--delete-data-dependent-branches-that-only-skip-compute) |
| `call_id with args (NestedUserFunctionVariable(),)` in `inspect.py` | a library calls `inspect.signature`/`id()` on a fresh nested closure | [GB-4](fix-patterns.md#gb-4--give-callables-a-stable-identity-or-stop-tracing-into-the-caller) |
| `hasattr UserDefinedClassVariable __get__` in `inspect.py` | the same reflection, now tripping on a `functools.partial`'s *type* | GB-4, step 2 |
| `Attempted to inline function marked as skipped` | `@torch.compiler.disable`, or Dynamo's skip list | [GB-5](fix-patterns.md#gb-5--accept-the-break) — usually correct |
| long blocking gaps at `.item()` / `.cpu()` / `bool(t)` | host–device sync draining the queued stream | [GB-6](fix-patterns.md#gb-6--hoist-hostdevice-syncs-out-of-per-block-code) |
| `assert_size_stride` while tracing an extern op | the op's fake/meta kernel disagrees with the real kernel | [GB-7](fix-patterns.md#gb-7--keep-an-opaque-kernel-in-the-graph) |

---

## 4. Decide whether to fix

**Not every break is worth removing, and break count is a bad target.**

A break at a boundary where the code was *already running eagerly* costs
approximately nothing. Measured on one workload: replacing 11 messy
library-internal breaks with 1 clean break at the call site made throughput
slightly **better** (84.4–84.7 vs 90.7–99.3 ms/step) — because Dynamo's repeated
failed attempt to trace into the library was pure wasted overhead. The work
itself ran eagerly either way.

Rank breaks by what they fragment, not by how many there are:

1. **Inside the repeated block** (per layer, per head) — worst. Splits the model
   into N fragments, multiplies recompiles by N, kills cross-layer fusion. Fix.
2. **Once per forward, mid-model** — splits the graph in two. Worth fixing if the
   halves would fuse.
3. **At an already-eager boundary** (a mask builder, a metrics callback, a
   `@torch.compiler.disable`d helper) — free. Accept.
4. **Before the first real op or after the last** — free.

A log with 16 breaks all at one already-eager library boundary is healthier than
a log with 3 breaks splitting a transformer block into 4 fragments.

---

## 5. Converting a messy break into a clean one

When the break is inside library reflection, chasing "which callable shape does
the library tolerate" is diminishing returns — each fix just moves the failure to
the next `inspect` call. Put the boundary where you want it instead:

```python
@torch.compiler.disable
def build_the_thing(...):     # Dynamo never looks inside
    return library_fn(...)
```

Dynamo breaks cleanly at the call site and resumes immediately after. N confusing
breaks become 1 expected one.

**The deliberate non-fix.** When the break is an upstream limitation, consider
leaving the workaround off so a future PyTorch release removes it for you — but
only after measuring that the break is free. That bet was taken on one workload
and cost nothing measurable.

---

## 6. Verify

Re-run with the same `TORCH_LOGS`. Then:

- `grep -c '<targeted reason>'` → `0`
- `grep -c 'Graph break in user code at'` → dropped by exactly the number targeted
- no *new* reason strings appear
- if you were also chasing recompiles: re-count those too. Removing breaks reduces
  the number of frames, which reduces recompile events per shape change — often
  by more than the break fix itself seemed worth.
