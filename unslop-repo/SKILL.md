---
name: unslop-repo
description: >-
  Measurable, prioritized cleanup of a Python ML research repo before open-sourcing it. Use when
  preparing a lab or research codebase for public release, auditing code quality, or planning and
  tracking a refactor with before/after numbers. Also for checking a repo for leaked secrets and API
  keys (W&B, Hugging Face, OpenAI, Anthropic, AWS, GitHub) anywhere in git history, internal
  cluster paths, hostnames and SLURM accounts (Mila, DRAC), large checkpoints or data in history, a
  missing LICENSE, vendored code under another license, undeclared or vulnerable dependencies
  (deptry, pip-audit), dead code (vulture), cyclomatic complexity, duplication, code sloppiness
  metrics (verbosity, erosion, clone ratio, scb-check, SlopCodeBench), or the red flags
  from Ousterhout's "A Philosophy of Software Design". Produces .cleanup/report.md, which another
  agent can work from, and .cleanup/issues.json, with every finding and its severity.
---

# Unslop a repo before open-sourcing

A router. One script measures, tracks and reports; depth lives in `references/`.

**Outputs**, both in `<repo>/.cleanup/` (it ignores itself in git, so nothing there is published):
- `report.md`: a self-contained work plan another agent can execute without this skill.
- `issues.json`: every finding with its severity (critical, high, medium, low) and status. It is the
  source of truth; the report is generated from it.

**Hard rules**
1. **Run the leak scan first; it is mandatory.** `measure` and `report` refuse to run until `leaks` has
   scanned the current commit. Never print, echo or `cat` a secret. Rotating keys is the user's job.
2. **Measure before changing anything,** and re-measure with the same scope and tool versions.
3. **Build the safety net before refactoring:** every function you change needs a test that calls it.
   See [safety-net.md](references/safety-net.md).
4. **The user approves each phase** before work on it starts.
5. **No nitpicks.** Only record or change something that makes the code meaningfully easier to
   understand, use or change. A refactor that adds lines needs a reason.

---

## 1. Workflow

```bash
S=<this skill>/scripts
$S/cleanup.py leaks   <repo> [--internal REGEX]   # 1. full-history secret scan (mandatory)
$S/cleanup.py measure <repo> --label baseline      # 2. scorecard + mechanical issues
#                                                    3. review issues.json (section 3)
$S/cleanup.py report  <repo>                       # 4. render .cleanup/report.md
#                                                    5. the user picks a phase, or hands the report on
#                                                    6. fix loop: safety net, one issue per commit
$S/cleanup.py leaks <repo> && $S/cleanup.py measure <repo> --label after-high   # 7. after the phase
$S/cleanup.py compare <repo> baseline && $S/cleanup.py report <repo>
```

- The safety net goes under `tests/`, in its own commit, before the first refactor.
- Each fix commit carries a `Cleanup-Issue: <ID>` trailer. Run the safety net after each commit;
  revert rather than debug forward. Delete dead code before refactoring what is left.
- Every new commit needs a new `leaks` run before `measure` or `report`. It takes seconds.
- `--internal REGEX` adds a repo-specific internal pattern (a lab hostname, `/checkpoint/`). Patterns
  are remembered for later scans.
- `--exclude GLOB` on `measure` leaves paths out of the metrics. Tests and docs are excluded by default,
  so adding a safety net doesn't move the numbers. Keep the same excludes for every snapshot.

## 2. Severity decides the phase

| severity | meaning | default for | phase |
|---|---|---|---|
| **critical** | can't be undone once published | credentials anywhere in history | High, first |
| **high** | fix before publishing | possible secrets to review; internal paths, hosts, emails at HEAD; files > 5 MB at HEAD; unused functions and classes of 20+ lines; missing LICENSE; vendored code under another license; vulnerable dependency with a fix; undeclared dependency; missing README | High |
| **medium** | the biggest quality wins | functions with CC > 10 and mass (CC × √SLOC) ≥ 100, e.g. CC 15 over 45 lines; duplicate groups with lines × (copies − 1) ≥ 20; trivial wrappers; red flags you add | Medium |
| **low** | optional | smaller complexity and duplication; anything inside a dead-code candidate (delete it instead); dead code under 20 lines; unused deps; vulnerabilities without a fix; no lockfile; findings only in history (except credentials) | Low |

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

Edit `.cleanup/issues.json` directly. Each call is a yes/no decision with a reason, never a 1–10 score.

- **Dead code:** vulture is guessing (60% confidence). Confirm nothing names the symbol: yaml
  `_target_`/`target:`, registries, `getattr`, notebooks, shell scripts. `measure` already dismisses
  candidates named with their module (`module.Name`, `module:name`, `from module import Name`) in configs,
  notebooks and tests, or in a string in the code (see `note`). Bare mentions (`model.train()`,
  `split="train"`) don't dismiss; they are listed in `evidence.mentioned_in` for you to check. Methods are not listed: framework hooks (`training_step`,
  `forward`) make them unreliable.
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
  survive re-runs; `grep <ID> .cleanup/issues.json` shows one issue per line.
- **Bugs found on the way** get their own issue (`"id": "BUG-A1"`, `"category": "bug"`, a `severity`
  and a `summary`; the rest is optional) and their own commit, never inside a refactor.
- **Typos stop the script:** every command checks issues.json first and names any issue with an unknown
  severity, status or category, a missing field, or a dismissal without a note.

## 4. Reading the scorecard

| metric | source | meaning |
|---|---|---|
| LOC, Verbosity, Erosion, Cognitive erosion, Violation %, Clone Ratio | `scb-check` 0.2.0 | Verbosity = share of lines that are duplicated or flagged by slop rules. Erosion = share of complexity mass (CC × √SLOC) in functions with CC > 10 |
| Functions CC > 10 | scb-check | the work list behind Erosion: one issue per function, with its cognitive complexity as evidence. CC counts `and`/`or` and conditional expressions too |
| Dead code LOC | vulture | unused functions and classes not named anywhere else |
| Red flags open, Leaks open | issues.json | counts of open issues |
| Suppressions | grep | `# noqa` and `# scbc ignore`: hiding findings is not fixing them |

- **Reference ranges** from SlopCodeBench (arXiv 2603.24755): maintained human Python repos average
  Verbosity 0.15 and Erosion 0.31, agent-written code 0.33 and 0.68. They are for orientation, not targets.
  scb-check 0.2.0 measures Violation % and Clone Ratio differently from the paper, so those two have no
  reference.
- **Ratios can rise when you delete clean code.** `compare` warns on absolute amounts instead: verbosity-
  flagged LOC, mass in high-CC functions, suppressions. A warning after a refactor means it added slop.
- **Erosion and verbosity predict maintenance cost, not bugs** (paper, Appendix G). Net LOC is the
  simplest honest signal: most cleanup commits should delete more than they add.
- **"not measured" is not zero,** and a tool that measured nothing never closes its earlier issues. deptry needs `pyproject.toml` dependencies or a `requirements.txt`;
  pip-audit needs a pinned source (`uv.lock`, `pylock.toml`, pinned requirements, `.venv`). It never
  resolves unpinned requirements, which can mean multi-GB torch downloads.

## 5. The script

`scripts/cleanup.py` is a single PEP 723 file. It uses only the stdlib, and runs the analyzers through
pinned `uvx`: scb-check 0.2.0, vulture 2.16, deptry 0.25.1, pip-audit 2.10.1.
Betterleaks 1.9.0 is not on PyPI. The script downloads its release binary once, refuses it unless the
SHA-256 matches the pinned value, and caches it in `~/.cache/unslop-repo/`. Set
`BETTERLEAKS=/path/to/betterleaks` to use your own copy (offline machines).

- `scripts/test_cleanup.py` checks the parsers and bookkeeping offline (`--offline`, a second), then
  builds a throwaway repo with one known instance of each finding and checks all of them, including
  that no secret value reaches any output.
- The shebang needs `env -S`. On old Linux (coreutils < 8.30) run `uv run --script cleanup.py ...`.
- `report` copies `cleanup.py` and `leak_rules.toml` into `.cleanup/`, so another agent can re-measure
  with `.cleanup/cleanup.py` without this skill.
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

*Thresholds (CC 10, mass 100, 20 duplicated lines, 20-line dead code) are defaults chosen to keep the
work list short.
The reference ranges come from one paper's panel of Python repos measured with its own tooling.*
