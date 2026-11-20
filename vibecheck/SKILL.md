---
name: vibecheck
description: >-
  Measurable, prioritized cleanup of a Python ML research repo before open-sourcing it. Use when
  preparing a lab or research codebase for public release, auditing code quality, or planning and
  tracking a refactor with before/after numbers. Also for checking a repo for leaked secrets and API
  keys (W&B, Hugging Face, OpenAI, Anthropic, AWS, GitHub) anywhere in git history, internal
  cluster paths, hostnames and SLURM accounts (Mila, DRAC), large checkpoints or data in the repo, a
  missing LICENSE or README, undeclared or vulnerable dependencies
  (deptry, pip-audit), dead code (vulture), cyclomatic complexity, duplication, code sloppiness
  metrics (verbosity, erosion, clone ratio, scb-check, SlopCodeBench), or the red flags
  from Ousterhout's "A Philosophy of Software Design". Produces .vibecheck/report.md, which another
  agent can work from, .vibecheck/issues.json, with every finding and its severity, and the metrics of
  each measured commit, so progress is tracked across the work.
metadata:
  version: 1.1.0
---

# Vibecheck a repo before open-sourcing

Vibecheck steers where to spend cleanup time. Its scores, metrics and rankings point at code worth a
look; they are not targets. A change is worth making when it makes the code easier to understand, use
or change, whether or not a number moves.

One script measures, tracks and reports; depth lives in `references/`.

**Outputs**, all in `<repo>/.vibecheck/` (it ignores itself in git, so nothing there is published):
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
3. **Build the safety net before refactoring,** in proportion to the risk:
   [safety-net.md](references/safety-net.md).
4. **The user approves each phase** before work on it starts.
5. **No nitpicks.** Only record or change something that makes the code meaningfully easier to
   understand, use or change. A refactor that adds lines needs a reason. Never pack code to save lines
   (`a, b = f(), g()`, several statements on one line): LOC should fall because code is gone.

---

## 1. Workflow

```bash
# 1. secret scan of every commit (mandatory)
./scripts/vibecheck.py leaks   <repo> [--internal REGEX]
# 2. this commit's scorecard + mechanical issues
./scripts/vibecheck.py measure <repo> --label baseline
# 3. review issues.json (section 3)
# 4. render .vibecheck/report.md
./scripts/vibecheck.py report <repo>
# 5. the user picks a phase, or hands the report on
# 6. fix loop: follow "How to work this report" at the top of report.md
./scripts/vibecheck.py leaks <repo> && ./scripts/vibecheck.py measure <repo>   # 7. after each commit
./scripts/vibecheck.py compare <repo> && ./scripts/vibecheck.py report <repo>  #    the baseline against this commit
```

- The fix loop's rules (one commit per family of fix, `Cleanup-Issue: <ID>` footers, what to do while
  refactoring) are written once, in report.md, so an agent without this skill gets them too. Revert
  rather than debug forward.
- Only the first `leaks` run reads every commit; later runs read the commits added since and the files
  at HEAD, which takes seconds. A rewritten history is read again in full.
- `--internal REGEX` adds a repo-specific internal pattern (a lab hostname, `/checkpoint/`). Patterns
  are remembered for later scans.
- `--exclude GLOB` on `measure` leaves paths out of the metrics and the dependency check. Tests and docs
  are excluded by default, so adding a safety net (and its `import pytest`) doesn't move the numbers.
  Keep the same excludes for every commit.
- `--label NAME` names a commit's metrics (`baseline` is the one `compare` and the report start from;
  without it, the first measured commit). `compare` takes a label, a commit or any git ref, and by
  default compares the baseline with the latest measured commit. The report's scorecard shows the
  baseline and the last six commits measured after it.

## 2. Each severity is a phase

Work the phases in order: critical, high, medium, low.

| severity | meaning | for example |
|---|---|---|
| **critical** | can't be undone once published, or exposes everyone who installs it | credentials in history, malicious dependency |
| **high** | fix before publishing | internal paths, big files, large dead code, no LICENSE |
| **medium** | the biggest quality wins | complex functions, duplicates, red flags you add |
| **low** | optional | small dead code, globals, string paths |

A vulnerable dependency takes the CVSS band of its worst advisory (critical 9.0+, high 7.0+, medium
4.0+, low); an unrated one is high, and one with no fixed release is low.

The exact rules are set in one place, `scripts/tools/__init__.py`, which creates each tool with the
severity of every kind of finding it reports and the thresholds they use; the tool's class says what each
kind covers. Every report.md ends with them, so an agent working from the report sees them too.

**Release status** is **blocked** while any critical or high issue is open, or while the leak scan is
older than HEAD.

**Critical phase checklist:** the user rotates leaked keys, then remove them from HEAD and decide on
history (a fresh, squashed history is recommended: [leaks.md](references/leaks.md)) · drop any
dependency whose advisory reports malicious code, and rotate what its build could reach · bump critical
pins.

**High phase checklist:** delete what shouldn't ship · add the safety net · add a LICENSE if missing
(the user picks it) · bump high-rated vulnerable pins · check the README quickstart works from a fresh
checkout.

**Skip, never record:** renaming locals, polishing comments, driving lint to zero, type annotations
everywhere, splitting long but simple functions (it creates shallow wrappers), Halstead volume or
maintainability index, refactoring code that is about to be deleted, and touching scripts that reproduce
published results (freeze or tag them).

## 3. Reviewing issues.json (step 3)

Edit `.vibecheck/issues.json` directly (`grep <ID>` finds an issue; edits survive re-runs). Each call is
a yes/no decision with a reason, never a 1–10 score: dismiss with `"status": "dismissed"` and a `note`.

- **Dead code:** `vulture` is guessing (60% confidence). Confirm nothing names the symbol: yaml
  `_target_`/`target:`, registries, `getattr`, notebooks, shell scripts. `measure` already dismisses
  candidates named with their module (`module.Name`, `module:name`, `from module import Name`) in configs,
  notebooks and tests, or in a string in the code (see `note`). Bare mentions (`model.train()`,
  `split="train"`) don't dismiss; they are listed in `evidence.mentioned_in` for you to check. Methods are not listed: framework hooks (`training_step`,
  `forward`) make them unreliable. A module whose name no other tracked file mentions, and that isn't a
  script (no `__main__` guard, no work at the top level), is one candidate for the whole file.
- **Possible secrets** (high, "Possible secret"): look at the location, never the value. Dismiss false
  positives. The scan ignores `gitleaks:allow` comments and the repo's own ignore files, so
  a key someone waved past a hook still shows up.
- **Red flags:** review the exposed code (entry points, public API, core model and training modules) and
  the files that hold the medium issues. Add an issue for each red flag worth fixing:
  [red-flags.md](references/red-flags.md). After the phase, review the same files again.
- **Medium and above:** make `fix` and `done_when` concrete when the default text is generic.
- **Typos stop the script:** every command checks issues.json first and names any issue with an unknown
  severity, status or category, a missing field, or a dismissal without a note.

## 4. Reading the scorecard

| metric | source | meaning |
|---|---|---|
| LOC, Verbosity, Erosion, Cognitive erosion, Violation %, Clone Ratio | `scb-check` 0.2.0 | Verbosity = share of lines that are duplicated or flagged by slop rules. Erosion = share of complexity mass (CC × √SLOC) in functions with CC > 10 |
| Functions CC > 10 | `scb-check` | the work list behind Erosion: one issue per function, with its cognitive complexity as evidence. CC counts `and`/`or` and conditional expressions too |
| Dead code LOC | `vulture` | unused functions, classes and modules not named anywhere else |
| Red flags open, Leaks open | issues.json | counts of open issues |
| Suppressions | grep | `# noqa` and `# scbc ignore`: hiding findings is not fixing them |
| Avg code lines per file, Files > 500 / > 1000 code lines | `tokenize`, `ast` | code lines per source file: comments, docstrings, imports and blank lines don't count, so documenting code never makes a file "too long". A file over 1000 gets a medium issue: split it into modules that each own one concept. A file that doesn't parse has no count (it is listed under not measured) |
| Globals | `ast` | names a module binds outside its functions and classes, constants included, plus names declared `global`. One low issue holds the count and the files with the most |
| Magic comparisons | `ruff` PLR2004 | comparisons against unnamed numbers (`if n_atoms > 38:`). Positional arguments and indices don't count: they are normal in scientific code. A function comparing against 3+ distinct ones gets a medium issue; every other function, and each module's top-level code, gets a low one |
| String paths | `ruff` PTH | `os.path`, `open()` and `glob` calls on string paths, where `pathlib` reads better. One low issue counts them: convert the code you touch anyway |

- **Reference averages** from SlopCodeBench, quoted in the report's metric descriptions, from the latest
  version of the paper that reports each. Verbosity and Erosion (arXiv 2603.24755v2, Table 2: 473
  maintained human Python repos): human code averages 0.19 and 0.34, agent-written code 0.44 and 0.68.
  Violation % and Clone Ratio (v1, Table 2; v2 dropped them): human 0.11 and 0.06, agent 0.32 and 0.08.
  `scb-check` 0.2.0 measures Violation % and Clone Ratio with its own
  rules and clone detection, so compare their changes, not their levels. The paper has no reference for
  Cognitive erosion.
- **Ratios can rise when you delete clean code.** `compare` warns on absolute amounts instead: verbosity-
  flagged LOC, mass in high-CC functions, suppressions, files over 1000 code lines, module-level globals.
  A warning after a refactor says where to look at what it added.
- **Erosion predicts maintenance cost, not failures** (paper v2, Appendix B.3): it correlates with the
  cost of the next change, not with whether that change passes its tests, and plain size (LOC) predicts
  cost better still. Net LOC is the simplest signal: most cleanup commits delete more than they add.
- **"not measured" is not zero,** and a tool that measured nothing never closes its earlier issues.
  `deptry` needs `pyproject.toml` dependencies or a `requirements.txt`; `pip-audit` needs a pinned source
  (`uv.lock`, `pylock.toml`, pinned requirements, `.venv`). It never resolves unpinned requirements,
  which can mean multi-GB `torch` downloads.

## 5. The script

`scripts/vibecheck.py` is the only entry point, a PEP 723 script whose header pins the analyzers. Each
tool's class names the same version (a test keeps them equal), and `measure` stops if another one is
installed. The modules it
imports sit next to it: `state` (issues.json, its severity scale and the snapshots), `scope` (the files
in scope), `leaks`, `measure`, `report`, `tools/` (one module per tool) and `utils/` (helpers, one module
per purpose).

`scb-check`, `vulture` and `pip-audit` run in process (`pip-audit` through its private API, so check
`tools/pip_audit.py` when bumping it). `pip-audit` reports no severity, so one OSV query per vulnerable
dependency supplies the rating; an unreachable OSV leaves it high. `ruff` and `deptry` run as `python -m`,
since `deptry`'s API exits the process. `betterleaks` is not on PyPI: the script downloads its release binary once, refuses it
unless the SHA-256 matches the pinned value, and caches it in `~/.cache/vibecheck/`. Set
`BETTERLEAKS=/path/to/betterleaks` to use your own copy (offline machines).

- `scripts/test_vibecheck.py` checks the bookkeeping offline (`--offline`, a second), then
  builds a throwaway repo with one known instance of each finding and checks all of them, including
  that no secret value reaches any output.
- The shebang needs `env -S`. On old Linux (coreutils < 8.30) run `uv run --script vibecheck.py ...`.
- `report` copies `vibecheck.py`, its modules, `tools/` (which holds the leak rules, `betterleaks.toml`)
  and `utils/` into `.vibecheck/scripts/`, so another agent can re-measure with
  `.vibecheck/scripts/vibecheck.py` without this skill.
- Coverage is optional and never required, because it usually means running a real training job. If a
  `coverage.json` (`coverage.py` JSON report) sits at the repo root, dead-code candidates whose bodies ran
  are dismissed (def lines run on import, so they don't count).

## 6. Reference map

| open | when |
|---|---|
| [leaks.md](references/leaks.md) | triaging leak findings, rotation, history, adding patterns, bumping `betterleaks` |
| [safety-net.md](references/safety-net.md) | before the first refactor |
| [red-flags.md](references/red-flags.md) | step 3, adding red-flag issues |
