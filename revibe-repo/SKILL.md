---
name: revibe-repo
description: >-
  Measurable, prioritized cleanup of a Python ML research repo before open-sourcing it. Use when
  preparing a lab or research codebase for public release, auditing code quality, or planning and
  tracking a refactor with before/after numbers. Also for checking a repo for leaked secrets and API
  keys (W&B, Hugging Face, OpenAI, Anthropic, AWS, GitHub) anywhere in git history, internal
  cluster paths, hostnames and SLURM accounts (Mila, DRAC), large checkpoints or data in the repo, a
  missing LICENSE or README, undeclared or vulnerable dependencies
  (deptry, pip-audit), dead code (vulture), cyclomatic complexity, duplication, code sloppiness
  metrics (verbosity, erosion, clone ratio, scb-check, SlopCodeBench), or the red flags
  from Ousterhout's "A Philosophy of Software Design". Produces .revibe/report.md, which another
  agent can work from, .revibe/issues.json, with every finding and its severity, and the metrics of
  each measured commit, so progress is tracked across the work.
---

# Revibe a repo before open-sourcing

A router. One script measures, tracks and reports; depth lives in `references/`.

**Outputs**, all in `<repo>/.revibe/` (it ignores itself in git, so nothing there is published):
- `report.md`: a self-contained work plan another agent can execute without this skill.
- `issues.json`: every finding with its severity (critical, high, medium, low) and status. It is the
  source of truth for the issues; the report is generated from it.
- `<commit time>-<commit>/metrics.json`: the scorecard of each measured commit, so the numbers can be
  followed across the work. The commit time (Unix epoch) leads the folder name, so the folders list in
  commit order. Re-measuring a commit replaces its metrics.

**Hard rules**
1. **Run the leak scan first; it is mandatory.** `measure` and `report` refuse to run until `leaks` has
   scanned the current commit. Never print, echo or `cat` a secret. Rotating keys is the user's job.
2. **Measure before changing anything,** and re-measure with the same scope and tool versions.
3. **Build the safety net before refactoring,** in proportion to the risk: a characterization test for
   each function you change that is somewhat complex (roughly CC above 5) or has side effects (files,
   global state, randomness, devices). A short or simple pure function needs none: the import test and
   its callers' tests cover it. See [safety-net.md](references/safety-net.md).
4. **The user approves each phase** before work on it starts.
5. **No nitpicks.** Only record or change something that makes the code meaningfully easier to
   understand, use or change. A refactor that adds lines needs a reason. Never pack code to save lines
   (`a, b = f(), g()`, several statements on one line): LOC should fall because code is gone.

---

## 1. Workflow

```bash
S=<this skill>/scripts
$S/cleanup.py leaks   <repo> [--internal REGEX]   # 1. secret scan of every commit (mandatory)
$S/cleanup.py measure <repo> --label baseline      # 2. this commit's scorecard + mechanical issues
#                                                    3. review issues.json (section 3)
$S/cleanup.py report  <repo>                       # 4. render .revibe/report.md
#                                                    5. the user picks a phase, or hands the report on
#                                                    6. fix loop: safety net, one issue per commit
$S/cleanup.py leaks <repo> && $S/cleanup.py measure <repo>   # 7. after each commit
$S/cleanup.py compare <repo> && $S/cleanup.py report <repo>  #    the baseline against this commit
```

- The safety net goes under `tests/`, in its own commit, before the first refactor.
- Each fix commit has a Conventional Commits message (`refactor:`, `fix:`, `test:`, ...) and a
  `Cleanup-Issue: <ID>` footer (see [safety-net.md](references/safety-net.md)). Run the safety net
  after each commit; revert rather than debug forward. Delete dead code before refactoring what is left.
- Every new commit needs a new `leaks` run before `measure` or `report`. Only the first reads every
  commit; later runs read the commits added since and the files at HEAD, which takes seconds. A
  rewritten history is read again in full.
- `--internal REGEX` adds a repo-specific internal pattern (a lab hostname, `/checkpoint/`). Patterns
  are remembered for later scans.
- `--exclude GLOB` on `measure` leaves paths out of the metrics and the dependency check. Tests and docs
  are excluded by default, so adding a safety net (and its `import pytest`) doesn't move the numbers.
  Keep the same excludes for every commit.
- `--label NAME` names a commit's metrics (`baseline` is the one `compare` and the report start from;
  without it, the first measured commit). `compare` takes a label, a commit or any git ref, and by
  default compares the baseline with the latest measured commit. The report's scorecard shows the
  baseline and the last six commits measured after it.

**While refactoring** (report.md repeats these):
- If a file holds many related pure functions, consider organizing them into a class.
- If a function returns a tuple, consider making it a named tuple, so callers read fields by name.
- If a function or method you touch is somewhat complex (several branches or loops, roughly CC above 5)
  and has no docstring, add one sentence explaining its purpose. Simple ones need none.
- Avoid introducing new globals: keep a constant next to its only user, and pass state as arguments.
  The scorecard counts module-level globals, and `compare` warns when the count goes up.
- Avoid catch-all modules such as `common.py` or `utils.py`. Put helpers in a `utils/` package with
  one module per purpose, e.g. `utils/paths.py`, `utils/timing.py`, `utils/git.py`. Never give a
  module the name of a standard-library module (`time.py`, `json.py`, `logging.py`, `types.py`): it
  shadows the library whenever its folder is on `sys.path`, as it is when a script there runs.
- Prefer `pathlib` to string paths: build paths with `/` (`root / "data" / name`) and use `.read_text()`,
  `.exists()` and `.glob()`, instead of `os.path.join`, f-string paths or `open(dir + "/" + name)`.

## 2. Severity decides the phase

| severity | meaning | for example | phase |
|---|---|---|---|
| **critical** | can't be undone once published | credentials anywhere in history | High, first |
| **high** | fix before publishing | possible secrets; internal paths and big files at HEAD; large dead code and unimported modules; generated outputs and scratch copies; no LICENSE; fixable vulnerabilities; no pyproject.toml dependencies or requirements.txt at the root; no README | High |
| **medium** | the biggest quality wins | complex functions, duplicate groups, trivial wrappers, source files over 1000 code lines, functions comparing against many unnamed numbers, working notes at the root, red flags you add | Medium |
| **low** | optional | smaller complexity and duplication, small dead code, module-level globals, functions with fewer magic comparisons, string paths, other history-only findings, unused deps, no lockfile | Low |

The exact rules are set in one place, `scripts/tools/__init__.py`, which creates each tool with the
severity of every kind of finding it reports and the thresholds they use; the tool's class says what each
kind covers. Every report.md ends with them, so an agent working from the report sees them too.

**Release status** is **blocked** while any critical or high issue is open, or while the leak scan is
older than HEAD.

**High phase checklist:** the user rotates leaked keys, then remove them from HEAD and decide on history
(a fresh, squashed history is recommended: [leaks.md](references/leaks.md)) · delete what shouldn't ship
· add the safety net · add a LICENSE if missing (the user picks it) · bump vulnerable pins · check the
README quickstart works from a fresh checkout.

**Skip, never record:** renaming locals, polishing comments, driving lint to zero, type annotations
everywhere, splitting long but simple functions (it creates shallow wrappers), Halstead volume or
maintainability index, refactoring code that is about to be deleted, and touching scripts that reproduce
published results (freeze or tag them).

## 3. Reviewing issues.json (step 3)

Edit `.revibe/issues.json` directly. Each call is a yes/no decision with a reason, never a 1–10 score.

- **Dead code:** vulture is guessing (60% confidence). Confirm nothing names the symbol: yaml
  `_target_`/`target:`, registries, `getattr`, notebooks, shell scripts. `measure` already dismisses
  candidates named with their module (`module.Name`, `module:name`, `from module import Name`) in configs,
  notebooks and tests, or in a string in the code (see `note`). Bare mentions (`model.train()`,
  `split="train"`) don't dismiss; they are listed in `evidence.mentioned_in` for you to check. Methods are not listed: framework hooks (`training_step`,
  `forward`) make them unreliable. A module whose name no other tracked file mentions, and that isn't a
  script (no `__main__` guard, no work at the top level), is one candidate for the whole file.
- **Possible secrets** (high, "Possible secret"): look at the location, never the value. Dismiss false
  positives with a `note`. The scan ignores `gitleaks:allow` comments and the repo's own ignore files, so
  a key someone waved past a hook still shows up.
- **Red flags:** review the exposed code (entry points, public API, core model and training modules) and
  the files that hold the medium issues. Add an issue for each red flag worth fixing:
  [red-flags.md](references/red-flags.md). After the phase, review the same files again.
- **Medium and above:** make `fix` and `done_when` concrete when the default text is generic.
- **To dismiss:** `"status": "dismissed"` plus a `note` with the reason. A false positive, a deliberate
  design (an override like `model.train = disabled_train`), not worth it, or a rotated key that stays in
  the local history are all valid. `fixed` only sticks once the tools stop detecting the issue:
  re-measuring reopens a fixed issue they still see. Your edits
  survive re-runs; `grep <ID> .revibe/issues.json` shows one issue per line.
- **Bugs found on the way** get their own issue (`"id": "BUG-A1"`, `"category": "bug"`, a `severity`
  and a `summary`; the rest is optional) and their own commit, never inside a refactor.
- **Typos stop the script:** every command checks issues.json first and names any issue with an unknown
  severity, status or category, a missing field, or a dismissal without a note.

## 4. Reading the scorecard

| metric | source | meaning |
|---|---|---|
| LOC, Verbosity, Erosion, Cognitive erosion, Violation %, Clone Ratio | `scb-check` 0.2.0 | Verbosity = share of lines that are duplicated or flagged by slop rules. Erosion = share of complexity mass (CC × √SLOC) in functions with CC > 10 |
| Functions CC > 10 | scb-check | the work list behind Erosion: one issue per function, with its cognitive complexity as evidence. CC counts `and`/`or` and conditional expressions too |
| Dead code LOC | vulture | unused functions, classes and modules not named anywhere else |
| Red flags open, Leaks open | issues.json | counts of open issues |
| Suppressions | grep | `# noqa` and `# scbc ignore`: hiding findings is not fixing them |
| Avg code lines per file, Files > 500 / > 1000 code lines | `tokenize`, `ast` | code lines per source file: comments, docstrings, imports and blank lines don't count, so documenting code never makes a file "too long". A file over 1000 gets a medium issue: split it into modules that each own one concept. A file that doesn't parse has no count (it is listed under not measured) |
| Globals | `ast` | names a module binds outside its functions and classes, constants included, plus names declared `global`. One low issue holds the count and the files with the most |
| Magic comparisons | ruff PLR2004 | comparisons against unnamed numbers (`if n_atoms > 38:`). Positional arguments and indices don't count: they are normal in scientific code. A function comparing against 3+ distinct ones gets a medium issue; every other function, and each module's top-level code, gets a low one |
| String paths | ruff PTH | `os.path`, `open()` and `glob` calls on string paths, where pathlib reads better. One low issue counts them: convert the code you touch anyway |

- **Reference averages** from SlopCodeBench, quoted in the report's metric descriptions, from the latest
  version of the paper that reports each. Verbosity and Erosion (arXiv 2603.24755v2, Table 2: 473
  maintained human Python repos): human code averages 0.19 and 0.34, agent-written code 0.44 and 0.68.
  Violation % and Clone Ratio (v1, Table 2; v2 dropped them): human 0.11 and 0.06, agent 0.32 and 0.08.
  They are for orientation, not targets. scb-check 0.2.0 measures Violation % and Clone Ratio with its own
  rules and clone detection, so compare their changes, not their levels. The paper has no reference for
  Cognitive erosion.
- **Ratios can rise when you delete clean code.** `compare` warns on absolute amounts instead: verbosity-
  flagged LOC, mass in high-CC functions, suppressions, files over 1000 code lines, module-level globals.
  A warning after a refactor means it added slop.
- **Erosion predicts maintenance cost, not failures** (paper v2, Appendix B.3): it correlates with the
  cost of the next change, not with whether that change passes its tests, and plain size (LOC) predicts
  cost better still. Net LOC is the simplest honest signal: most cleanup commits should delete more than
  they add.
- **"not measured" is not zero,** and a tool that measured nothing never closes its earlier issues.
  deptry needs `pyproject.toml` dependencies or a `requirements.txt`; pip-audit needs a pinned source
  (`uv.lock`, `pylock.toml`, pinned requirements, `.venv`). It never resolves unpinned requirements,
  which can mean multi-GB torch downloads.

## 5. The script

`scripts/cleanup.py` is the only entry point, a PEP 723 script whose header pins the analyzers:
scb-check 0.2.0, vulture 2.16, deptry 0.25.1, pip-audit 2.10.1, ruff 0.15.7. Each tool's class names the same
version (a test keeps them equal), and `measure` stops if another one is installed. The modules it
imports sit next to it: `state` (issues.json, its severity scale and the snapshots), `scope` (the files
in scope), `leaks`, `measure`, `report`, `tools/` (one module per tool) and `utils/` (helpers, one module
per purpose).

Each tool module defines a class: the tool's name, PyPI package and pinned version, what each kind of
finding covers (`policy`), its scorecard metrics, and the methods that run it and turn what it finds into
issues. `tools/__init__.py` creates every tool with its severities and settings, and a tool refuses
severities that don't match its policy. `tools/base.py` defines `Tool`, which builds every issue, the
`Result` every tool returns, and `Step`, the base of a measure step, `measure(scope) -> Result`; the
`Scope` (`scope.py`) holds the repo, the files in scope and what they define. Betterleaks, the leak
scan, has `scan` instead.

scb-check, vulture and pip-audit run in process (pip-audit through its private API, so check
`tools/pip_audit.py` when bumping it); ruff runs as `python -m ruff`, and deptry as `python -m deptry`, since
deptry's API exits the process. Betterleaks 1.9.0 is not on PyPI: the script downloads its release binary once, refuses it
unless the SHA-256 matches the pinned value, and caches it in `~/.cache/revibe-repo/`. Set
`BETTERLEAKS=/path/to/betterleaks` to use your own copy (offline machines).

- `scripts/test_cleanup.py` checks the bookkeeping offline (`--offline`, a second), then
  builds a throwaway repo with one known instance of each finding and checks all of them, including
  that no secret value reaches any output.
- The shebang needs `env -S`. On old Linux (coreutils < 8.30) run `uv run --script cleanup.py ...`.
- `report` copies `cleanup.py`, its modules, `tools/` (which holds the leak rules, `betterleaks.toml`)
  and `utils/` into `.revibe/scripts/`, so another agent can re-measure with
  `.revibe/scripts/cleanup.py` without this skill.
- Coverage is optional and never required, because it usually means running a real training job. If a
  `coverage.json` (coverage.py JSON report) sits at the repo root, dead-code candidates whose bodies ran
  are dismissed (def lines run on import, so they don't count).

## 6. Reference map

| open | when |
|---|---|
| [leaks.md](references/leaks.md) | triaging leak findings, rotation, history, adding patterns, bumping Betterleaks |
| [safety-net.md](references/safety-net.md) | before the first refactor |
| [red-flags.md](references/red-flags.md) | step 3, adding red-flag issues |

---

*The thresholds in the tools' settings are defaults chosen to keep the work list short.
The reference ranges come from one paper's panel of Python repos measured with its own tooling.*
