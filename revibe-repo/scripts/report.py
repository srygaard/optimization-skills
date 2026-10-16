"""Scorecards, `cleanup.py compare` and `cleanup.py report`."""

import shutil
from pathlib import Path

from state import Cleanup, Severity
from tools import policies, scorecard_metrics


# --- scorecard -------------------------------------------------------------------------------------

def scorecard(snaps: list) -> list:
    """Rows of [metric, value per snapshot, delta (last minus first, with two or more), reference] for every
    metric that any of the snapshots measured: the measure steps' in step order, then the open-issue counts.
    measure, compare and report all print these rows."""
    rows = []
    for m in (*scorecard_metrics(), *Cleanup.metrics):
        vals = [s["metrics"].get(m.key) for s in snaps]
        if not m.label or all(v is None for v in vals):
            continue
        row = [m.label, *("-" if v is None else m.form.format(v) for v in vals)]
        if len(snaps) >= 2:
            row.append(change(vals[0], vals[-1]))
        ref = m.reference
        rows.append(row + [f"{ref[0]} / {ref[1]}" if ref else ""])
    return rows


def change(before, after) -> str:
    """How a metric moved between two snapshots, signed; "" when either didn't measure it."""
    if before is None or after is None:
        return ""
    return f"{after - before:+,}" if isinstance(before, int) else f"{after - before:+.3f}"


def column(snap: dict) -> str:
    """What a snapshot is called in a table: its label, or its commit."""
    return snap.get("label") or snap["commit"][:8]


def print_scorecard(snap: dict):
    print(f"scorecard  {column(snap)}  @ {snap['commit'][:8]}{' (dirty)' if snap['dirty'] else ''}  "
          f"release: {snap['release_status'].upper()}")
    for label, value, ref in scorecard([snap]):
        print(f"  {label:<22}{value:>10}" + (f"   (paper, human / agent code: {ref})" if ref else ""))
    o = snap["open_issues"]
    print("  Open issues           " + " · ".join(f"{o[s.name]} {s.name}" for s in Severity))
    for tool, why in snap["not_measured"].items():
        print(f"  not measured: {tool} ({why[:100]})")


# --- compare ---------------------------------------------------------------------------------------

def cmd_compare(a):
    """`cleanup.py compare`: two measured commits side by side, with the regressions between them. By default
    the baseline and the latest."""
    cleanup = Cleanup(a.repo)
    cleanup.save()  # counts issues edited by hand into HEAD's metrics
    base = cleanup.snapshot(a.base) if a.base else cleanup.baseline()
    after = cleanup.snapshot(a.to) if a.to else (cleanup.snapshots() or [None])[-1]
    if base is None or after is None or after["commit"] == base["commit"]:
        raise SystemExit("nothing to compare: commit, then run `cleanup.py measure` on the new commit")
    rows = scorecard([base, after])
    for s in Severity:
        b = base["open_issues"][s.name]
        n = after["open_issues"][s.name]
        rows.append([f"open {s.name}", b, n, f"{n - b:+}", ""])
    print(f"{'metric':<22}{column(base):>14}{column(after):>14}{'delta':>10}")
    for label, b, n, delta, _ in rows:
        print(f"{label:<22}{b:>14}{n:>14}{delta:>10}")
    for w in comparison_warnings(base, after):
        print(f"warning: {w}")


def comparison_warnings(base: dict, after: dict) -> list:
    """Regressions in absolute terms. Ratios are not used: deleting clean code raises them on its own."""
    out = []
    for m in scorecard_metrics():
        b = base["metrics"].get(m.key)
        n = after["metrics"].get(m.key)
        if m.warn and b is not None and n is not None and n > b:
            out.append(f"{m.warn} went up ({b:,} -> {n:,}): cleanup should not add slop or hide findings")
    if base["tools"] != after["tools"]:
        out.append("tool versions differ between the snapshots: numbers may not be comparable")
    if base["scope"] != after["scope"]:
        out.append("scope (excludes) differs between the snapshots: numbers are not comparable")
    return out


# --- report ----------------------------------------------------------------------------------------

HOW_TO = """\
1. Work in phase order: High, then Medium, then Low (critical and high issues are the High phase). Bigger items come first within a category.
2. Critical leaks: the user must revoke/rotate the key first. Never print, echo or `cat` a secret; the previews here are enough.
3. Safety net before refactoring, in its own commit under `tests/` (outside the metrics): a test that imports every module, `--help` for each entry point, and a characterization test (small synthetic inputs, pinned output) for each function you change. The tests must run in the project's real environment; if it isn't available (e.g. no torch, no GPU), ask the user before refactoring, and don't commit stubs.
4. One issue per commit, with a `Cleanup-Issue: <ID>` trailer. Run the safety net after each commit. Delete dead code before refactoring: issues marked "Inside dead-code candidate" go away with it.
5. Record each outcome in `.revibe/issues.json` (one issue per line; `grep <ID>` finds it): `status` `fixed`, or `dismissed` with a `note`. Re-measuring reopens a `fixed` issue the tools still detect. A false positive, a deliberate design, not worth the change, or a rotated key that stays in the local history are all valid reasons to dismiss. Bugs you find get their own issue (`"id": "BUG-A1"`, `"category": "bug"`, a `severity` and a `summary`; the other fields are optional) and their own commit, never inside a refactor. Don't edit this report; regenerate it.
6. No nitpicks: only change what makes the code meaningfully easier to understand, use or change. A refactor that adds lines needs a reason. Never pack code to save lines (`a, b = f(), g()`, several statements on one line): LOC should fall because code is gone.
7. While refactoring: if a file holds many pure functions, consider organizing them into classes; if a function returns a tuple, consider a named tuple; give each function or method you touch a one-sentence docstring on its purpose if it has none; don't introduce new globals (keep a constant next to its only user, pass state as arguments); and instead of a catch-all `common.py` or `utils.py`, put helpers in a `utils/` package with one module per purpose (`utils/paths.py`, `utils/timing.py`, `utils/git.py`), never named after a standard-library module such as `time.py` or `json.py`, which it would shadow; and use `pathlib` (`root / "data" / name`, `.read_text()`, `.exists()`) instead of string paths built with `os.path.join` or f-strings.
8. Re-measure after each commit, so `.revibe/<commit>/metrics.json` tracks the work: `.revibe/scripts/cleanup.py leaks .`, `.revibe/scripts/cleanup.py measure .`, then `.revibe/scripts/cleanup.py compare .` (the baseline against this commit) and `.revibe/scripts/cleanup.py report .` (needs uv)."""

def cell(text) -> str:
    return str(text if text is not None else "").replace("|", "\\|").replace("\n", " ")


def md_table(header: list, rows: list) -> list:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header),
            *("| " + " | ".join(cell(x) for x in row) + " |" for row in rows)]


def where(i: dict) -> str:
    loc = f"{i['file']}:{i['line']}" if i.get("file") and i.get("line") else i.get("file") or ""
    sym = i.get("symbol") or ""
    return f"`{loc}`" + (f" ({sym})" if sym and sym != "<module>" and len(sym) < 60 else "") if loc else sym


def report_header(cleanup: Cleanup) -> list:
    """Title, commit, release status and what blocks it, then the instructions."""
    data = cleanup.data
    status = data["release_status"].upper()
    reason = "" if cleanup.leak_scan_current() else " (the leak scan is not current: run `leaks` again)"
    critical = len(cleanup.open_issues(Severity.critical))
    high = len(cleanup.open_issues(Severity.high))
    blocking = (f" · {critical} critical and {high} high issues open: the High phase is what blocks the release"
                if critical or high else "")
    return [f"# Cleanup report: {cleanup.repo.name}", "",
            f"Commit `{data['commit'][:8]}` · generated {data['generated'][:10]} · "
            f"release status: **{status}**{reason}{blocking}", "",
            "Source of truth: `.revibe/issues.json`, and `.revibe/<commit>/metrics.json` for each measured commit. "
            "This file is generated by `cleanup.py report`.", "",
            "## How to work this report", "", HOW_TO, "", "## Scorecard", ""]


def report_columns(cleanup: Cleanup) -> list:
    """The snapshots the scorecard shows, oldest first: the baseline and the last six commits measured after
    it, so the table tracks the work."""
    base = cleanup.baseline()
    if base is None:
        return []
    return [base, *[s for s in cleanup.snapshots() if s["date"] > base["date"]][-6:]]


def scorecard_section(cols: list) -> list:
    """The scorecard table, with deltas and regression warnings (baseline against latest) when it has several
    snapshots."""
    if not cols:
        return ["No snapshot yet: run `cleanup.py measure`."]
    delta = ["Δ"] if len(cols) >= 2 else []
    lines = md_table(["metric", *(column(s) for s in cols), *delta, "reference (paper: human / agent)"],
                     scorecard(cols))
    if not_measured := cols[-1].get("not_measured"):
        lines += ["", "Not measured: " + "; ".join(f"{k} ({v[:80]})" for k, v in not_measured.items())]
    if len(cols) >= 2:
        lines += [f"\n> warning: {w}" for w in comparison_warnings(cols[0], cols[-1])]
    return lines


def phase_section(phase: str, items: list) -> list:
    """A phase's open issues as a table; a fix or done-when equal to the row above reads "same as above"."""
    lines = ["", f"## {phase} phase ({len(items)} open)", ""]
    if not items:
        return lines + ["Nothing open."]
    done = phase != "Low"  # Low issues are optional: no done-when column
    rows = []
    prev = {}
    for i in items:
        same = {k: "same as above" if i[k] and i[k] == prev.get(k) else i[k] for k in ("fix", "done_when")}
        rows.append([i["id"], i["severity"], where(i), i["summary"], same["fix"],
                     *([same["done_when"]] if done else []), i["effort"]])
        prev = i
    return lines + md_table(["ID", "severity", "where", "problem", "fix", *(["done when"] if done else []), "effort"],
                            rows)


def dismissed_section(data: dict) -> list:
    """Dismissed issues with their reasons, then how many are fixed."""
    dismissed = [i for i in data["issues"] if i["status"] == "dismissed"]
    fixed = sum(i["status"] == "fixed" for i in data["issues"])
    lines = ["", f"## Dismissed ({len(dismissed)})", ""]
    if dismissed:
        lines += md_table(["ID", "where", "problem", "reason"],
                          [[i["id"], where(i), i["summary"], i["note"]] for i in dismissed])
    else:
        lines.append("None.")
    return lines + ["", f"Fixed so far: {fixed} issue(s)."]


def severity_rules() -> list:
    """Every tool's policy as a list, so an agent working from the report sees why each issue has its severity."""
    rows = policies().values()
    return ["", "## Severity rules", "",
            "Set where tools/__init__.py creates each tool; red flags and bugs you add keep the severity you "
            "give them.", "",
            *(f"- **{sev.name}:** " + "; ".join(when for s, when in rows if s == sev.name) for sev in Severity)]


def render_report(cleanup: Cleanup) -> str:
    """report.md: the header, the scorecard, a table per phase, the dismissed issues and the severity rules."""
    phases = (("High", (Severity.critical, Severity.high)), ("Medium", (Severity.medium,)), ("Low", (Severity.low,)))
    lines = report_header(cleanup) + scorecard_section(report_columns(cleanup))
    for phase, severities in phases:
        lines += phase_section(phase, cleanup.open_issues(*severities))
    lines += dismissed_section(cleanup.data) + severity_rules()
    return "\n".join(lines) + "\n"


def cmd_report(a):
    """`cleanup.py report`: render .revibe/report.md, and copy the scripts to .revibe/scripts/."""
    cleanup = Cleanup(a.repo)
    cleanup.require_leak_scan()
    cleanup.save()  # refreshes the release status and the snapshot taken at HEAD
    d = cleanup.dir
    (d / "report.md").write_text(render_report(cleanup))
    scripts = Path(__file__).resolve().parent  # cleanup.py, the modules it imports, tools/ and utils/
    copy = d / "scripts"
    if scripts != copy.resolve():  # lets another agent re-measure without the skill
        shutil.rmtree(copy, ignore_errors=True)  # a module renamed since the last copy must not linger
        shutil.copytree(scripts, copy, ignore=shutil.ignore_patterns("__pycache__", "test_*.py", ".*"))
    counts = " · ".join(f"{len(cleanup.open_issues(s))} {s.name}" for s in Severity)
    print(f"release: {cleanup.data['release_status'].upper()}  ·  open {counts}")
    print(f"-> {d / 'report.md'}")
