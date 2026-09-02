# Measurement hygiene

Twelve rules. Each one exists because a real measurement was wrong first.

Compile numbers are unusually easy to fool yourself with: the effects are large,
the variance is larger, and half the interesting quantities are not logged
anywhere obvious.

---

### 1. Confirm the path under test actually ran

One experiment reported a configuration that was in fact the **baseline path** —
a method was defined on an inner class and looked up on the outer wrapper,
returned `None`, and the code returned quietly. It produced a clean number that
matched the baseline within noise and was nearly recorded as a refutation of the
whole hypothesis.

> Any experiment whose active path has no positive confirmation in the log can
> produce a baseline number and be mistaken for a refutation.

Log a distinctive line on the path under test. Make the setup **raise** rather
than no-op when it cannot find what it needs.

---

### 2. Read knob values from the resolved config, not the command line

A value set in a config file never appears as a command-line override, so scraping
overrides silently mislabels those runs. Dump the resolved config at startup and
read *that*.

Corollary for compile specifically: a `torch._inductor.config` attribute you set
may be overwritten by a mode preset. Confirm the *effect* in the trace — the phase
gone from the table — not the attribute.

---

### 3. Know your noise floor before believing a delta

Measured on one workload: two matched no-lever baselines differed by **10.8%**;
five matched runs of a single commit spanned 72.7–95.8 ms/step.

Treat anything under ~15% as unresolved unless it reproduces. And note the shape
of the delta: **a difference that is uniform across every input shape is usually
machine variance; a real effect is normally shaped** — bigger where the mechanism
says it should be bigger.

---

### 4. Prefer medians and kernel time to means

One run's **mean** was 2.2× worse than another's (138.0 vs 62.1 ms/step) while its
**median** step was healthy (39.5 ms). The mean was wrecked by a fat tail: 12% of
steps at 300–1400 ms, 97% of which coincided with an input-pipeline stall
recurring on a period equal to the dataloader worker count.

Comparing means across such runs is invalid.

The same trap in reverse: a "std dropped 201 → 46 ms" claim can be a statement
about how much I/O stall each run happened to absorb, not about kernel
consistency. When in doubt, compare **GPU kernel time**, which no host-side stall
can contaminate.

---

### 5. Exclude compiling steps from steady-state stats, and say so

Detect them by **shape first-appearance**, not by a fixed warmup count:

```python
if (batch, seq) not in seen:
    seen.add((batch, seq)); is_recompile = True    # exclude from mean/std
```

This is strictly better than raising a warmup threshold until the spikes fall
inside it — you never have to know in advance which step number the last new shape
lands on. Log the shape inline (`step 109: 244,081 ms (recompile (72, 576))`) so
transitions correlate with spikes instead of being inferred from token counts.

---

### 6. Check your sample size

With a warmup plus recompile-step exclusions, a 20-step run measures ~13 steps —
and the shape mix cannot reach its stationary proportions in 13 draws. A 2-step
run measures nothing at all, and is useful only for compile cost and cache
population.

Use 60+ measured steps for a number comparable to a several-hundred-step baseline.

---

### 7. Subtract input-pipeline startup from step 1

Once compile time falls, startup stops being negligible: on one workload it went
from 3% to 17% of step 1 without changing at all. Instrument batch-load time
separately and subtract it before calling step 1 a compile metric.

---

### 8. Compare total time across all compiling steps, not step 1

A change can make step 1 look excellent and pay a second large compile three steps
later — one block-size change improved step 1 and made total compile **+72%**
worse.

Also: frame-compile timers omit autotuning clocked outside them, so **wall time
across the compiling steps is the honest number**, not the sum of the phase table.

---

### 9. Design two-lever questions as a 2×2

Require **each edge** to reproduce, and report the interaction. Two levers
measured this way came out −62.2% together — better than independence predicted,
which is information you cannot get from two one-at-a-time runs.

---

### 10. Instrumentation changes the measurement

`TORCH_TRACE` costs ~10 s (`structured_logging_overhead_s = 9.96`). A traced run is
not comparable to an untraced baseline. Profiling with stacks costs more.

Trace the run you are *diagnosing*; baseline the run you are *reporting*.

---

### 11. Get positive proof that compilation happened at all

The absence of a config key from a dump is weak evidence. Inductor's own internal
warnings appear on stderr **only during an actual codegen pass** — e.g. the
`TensorFloat32 tensor cores ... available but not enabled` notice from
`torch/_inductor/compile_fx.py`, or `DeviceCopy in input program` from
`torch/_inductor/utils.py`. Their presence proves a compile ran.

Same principle as rule 1, applied to the compiler itself.

---

### 12. Fix one thing and re-count exactly

The before/after `grep -c` protocol:

```bash
grep -c '<specific reason or guard>' before.err
grep -c '<specific reason or guard>' after.err     # expect 0
grep -c 'Graph break in user code at' after.err    # expect before - N, no more
```

A good fix removes the lines it targeted, **no more and no less**. Line-number
shifts from added code are not new findings, and breaks unmasked by removing a
noisy one were always there.

---

## The zero-lines trap

Stated separately because it invalidates entire investigations:

> `TORCH_LOGS` emits a line only when something actually happens. **Zero
> `Recompiling function` lines with the variable unset looks exactly like zero
> with it set.**

A clean stderr is meaningless unless you can prove the variable was live in the
process that wrote it. Print `os.environ.get("TORCH_LOGS")` at startup, or have
`log_triage.py` tell you (it warns when a file contains no Dynamo anchors of any
kind).

If the process runs inside any wrapper that filters its environment — a container
runtime, a job launcher, a subprocess with a curated `env` — a newly added
diagnostic variable is **silently dropped** unless explicitly forwarded. This has
cost people days.
