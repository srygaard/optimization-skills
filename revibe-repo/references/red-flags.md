# Red flags (Ousterhout, *A Philosophy of Software Design*)

Each red flag is a symptom of complexity. Record one only if fixing it would make the code meaningfully
easier to understand, use or change. A flag on code that is about to be deleted is not worth recording.

**Found automatically** by `measure` (via scb-check): **Repetition** (duplicated blocks of 10+ lines),
**Pass-Through Method** (`trivial-wrapper`), **Shallow Module** (`low-use-short-function`), and
**Nonobvious Code** as comparisons against unnamed numbers (ruff PLR2004).
**The rest come from your review** of the exposed code and the files holding medium issues.

| flag | how it shows up in ML research code | typical fix | severity |
|---|---|---|---|
| Repetition | copy-pasted training loops, samplers (DDIM/PLMS) and eval scripts; one model file per experiment | one function with the real differences as a few parameters; delete dead variants | medium |
| Pass-Through Method | `def get_model(cfg): return build_model(cfg)`; wrappers that forward every argument | inline into callers or remove | medium |
| Shallow Module | a class per tiny step; helpers called once whose interface is as big as their body | merge into the caller or a deeper module | medium |
| Information Leakage | the same batch-dict keys, file layout or tensor-shape convention hardcoded in several modules | one module owns the format; the others call it | medium |
| Overexposure | `train(...)` with 40 keyword arguments, most never changed; config objects every caller must fill in | sensible defaults; group rarely used options | medium |
| Special-General Mixture | `if dataset == "imagenet":` or `if args.exp == "v3":` inside a generic layer or loop | move the special case to the caller, a config or a subclass | medium |
| Nonobvious Code | unnamed thresholds (`if n_atoms > 38:`), monkeypatching (`model.train = disabled_train`), `getattr` with built strings, side effects on import | name it, explain the why in one comment, or simplify | medium |
| Temporal Decomposition | `preprocess.py` → `stage2.py` → `postprocess.py` split by execution order, each knowing the same file format | group by what each piece hides, not by when it runs | low |
| Conjoined Methods | `forward` and `_forward_impl` (or train/eval twins) that can't be read apart | merge, or split along a real boundary | low |
| Hard to Describe | a docstring can't say what the function does in one sentence without "and/also/unless" | split along the "and", or narrow what it does | low |
| Hard to Pick Name | only names like `process`, `handle` or `do_step_and_log` fit | same: the function does too much | low |
| Implementation Documentation Contaminates Interface | a public docstring that narrates the algorithm's steps instead of inputs, outputs and effects | move the steps to comments inside | low |
| Vague Name | `data`, `tmp`, `x2` or `res` in a public signature | **public API only**; leave locals and math notation (x, h, q, k, v, B, T, C) alone | low |
| Comment Repeats Code | `# increment i` | **public API only**; delete the comment | low |

## Two judgment calls, made the same way each time

- **Hard to Describe:** from the signature and docstring alone, write ≤ 25 words without
  "and/also/unless". Then read the body. Record the flag if the sentence misses an effect or needs a
  conjunction.
- **Hard to Pick Name:** propose a name of ≤ 4 words that covers the whole body. Record the flag if that
  is impossible without "and" or "or".

## Recording a red flag

Append to `issues` in `.revibe/issues.json`. Use IDs `RF-A1`, `RF-A2`, and so on; `measure` never
touches issues with `"source": "agent"`.

```json
{"id": "RF-A1", "source": "agent", "category": "red-flag", "rule": "special-general-mixture",
 "severity": "medium", "file": "ldm/models/diffusion/ddpm.py", "line": 812, "end_line": 860,
 "symbol": "LatentDiffusion.get_input", "summary": "Dataset-specific branches inside the generic input path",
 "evidence": {}, "fix": "Move the per-dataset key handling into the datasets' collate functions.",
 "done_when": "get_input has no dataset names; the safety net passes.", "effort": "M",
 "status": "open", "note": ""}
```

**Keep the count comparable.** After the phase, review the same files again: mark fixed flags
`fixed`, and add any new ones. The open red-flag count before and after is the measure. Run `report`
after each review: it recounts HEAD's metrics, so the baseline includes the flags you added.
