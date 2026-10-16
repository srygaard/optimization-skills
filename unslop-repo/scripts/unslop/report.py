"""Scorecards, `cleanup.py compare` and `cleanup.py report`."""

import shutil
from pathlib import Path

from .common import (
    CC_THRESHOLD, LONG_FILE_LINES, POLICY, REFERENCE, RULES, SCRIPTS, SEVERITIES, WARN_FILE_LINES, die, repo_root,
)
from .state import (
    latest_snapshot, leak_scan_current, load_issues, load_snapshot, open_issues, recount_head_snapshot,
    require_leak_scan, save_issues, state_dir,
)


# --- scorecard -------------------------------------------------------------------------------------

ROWS = (  # key, label, format
    ("loc", "LOC", "{:,}"), ("verbosity", "Verbosity", "{:.3f}"), ("erosion", "Erosion", "{:.3f}"),
    ("cog_erosion", "Cognitive erosion", "{:.3f}"), ("violation_pct", "Violation %", "{:.3f}"),
    ("clone_ratio", "Clone Ratio", "{:.3f}"), ("cc_over_10", f"Functions CC > {CC_THRESHOLD}", "{:,}"),
    ("dead_code_loc", "Dead code LOC", "{:,}"), ("red_flags_open", "Red flags open", "{:,}"),
    ("leaks_open", "Leaks open", "{:,}"), ("vulnerable_deps", "Vulnerable deps", "{:,}"),
    ("deps_missing", "Dependency problems", "{:,}"), ("suppressions", "Suppressions", "{:,}"),
    ("avg_file_lines", "Avg file length", "{:,}"), ("files_over_500", f"Files > {WARN_FILE_LINES} lines", "{:,}"),
    ("files_over_1000", f"Files > {LONG_FILE_LINES} lines", "{:,}"),
)


def fmt(key: str, value) -> str:
    fmt_ = next((f for k, _, f in ROWS if k == key), "{}")
    return "-" if value is None else fmt_.format(value)


def scorecard(snaps: list) -> list:
    """Rows of [metric, value per snapshot, delta (with two snapshots), reference] for every metric
    that any of the snapshots measured. measure, compare and report all print these rows."""
    rows = []
    for key, label, _ in ROWS:
        vals = [s["metrics"].get(key) for s in snaps]
        if all(v is None for v in vals):
            continue
        row = [label, *(fmt(key, v) for v in vals)]
        if len(snaps) == 2:
            b, n = vals
            row.append("" if b is None or n is None else f"{n - b:+,}" if isinstance(b, int) else f"{n - b:+.3f}")
        ref = REFERENCE.get(key)
        rows.append(row + [f"{ref[0]} / {ref[1]}" if ref else ""])
    return rows


def print_scorecard(snap: dict):
    print(f"scorecard  {snap['label']}  @ {snap['commit'][:8]}{' (dirty)' if snap['dirty'] else ''}  "
          f"release: {snap['release_status'].upper()}")
    for label, value, ref in scorecard([snap]):
        print(f"  {label:<22}{value:>10}" + (f"   (paper, human / agent code: {ref})" if ref else ""))
    o = snap["open_issues"]
    print("  Open issues           " + " · ".join(f"{o[s]} {s}" for s in SEVERITIES))
    for tool, why in snap["not_measured"].items():
        print(f"  not measured: {tool} ({why[:100]})")


# --- compare ---------------------------------------------------------------------------------------

def cmd_compare(a):
    repo = repo_root(a.repo)
    d = state_dir(repo)
    recount_head_snapshot(d, load_issues(d), repo)
    base = load_snapshot(d, a.base)
    after = load_snapshot(d, a.to) if a.to else latest_snapshot(d, exclude=a.base)
    if after is None:
        die("nothing to compare: run `cleanup.py measure --label <after>` first")
    rows = scorecard([base, after]) + [
        [f"open {s}", base["open_issues"][s], after["open_issues"][s],
         f"{after['open_issues'][s] - base['open_issues'][s]:+}", ""] for s in SEVERITIES]
    print(f"{'metric':<22}{base['label']:>14}{after['label']:>14}{'delta':>10}")
    for label, b, n, delta, _ in rows:
        print(f"{label:<22}{b:>14}{n:>14}{delta:>10}")
    for w in comparison_warnings(base, after):
        print(f"warning: {w}")


def comparison_warnings(base: dict, after: dict) -> list:
    """Regressions in absolute terms. Ratios are not used: deleting clean code raises them on its own."""
    out = []
    for key, label in (("verbosity_flagged_loc", "verbosity-flagged LOC"),
                       ("high_cc_mass", "mass in functions with CC > 10"), ("suppressions", "suppressions"),
                       ("files_over_1000", f"files over {LONG_FILE_LINES} lines")):
        b, n = base["metrics"].get(key), after["metrics"].get(key)
        if b is not None and n is not None and n > b:
            out.append(f"{label} went up ({b:,} -> {n:,}): cleanup should not add slop or hide findings")
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
5. Record each outcome in `.cleanup/issues.json` (one issue per line; `grep <ID>` finds it): `status` `fixed`, or `dismissed` with a `note`. Re-measuring reopens a `fixed` issue the tools still detect. A false positive, a deliberate design, not worth the change, or a rotated key that stays in the local history are all valid reasons to dismiss. Bugs you find get their own issue (`"id": "BUG-A1"`, `"category": "bug"`, a `severity` and a `summary`; the other fields are optional) and their own commit, never inside a refactor. Don't edit this report; regenerate it.
6. No nitpicks: only change what makes the code meaningfully easier to understand, use or change. A refactor that adds lines needs a reason.
7. Re-measure when you stop (the label is any name): `.cleanup/cleanup.py leaks .`, `.cleanup/cleanup.py measure . --label after-<what>`, `.cleanup/cleanup.py compare . baseline`, `.cleanup/cleanup.py report .` (needs uv)."""

PHASES = (("High", ("critical", "high")), ("Medium", ("medium",)), ("Low", ("low",)))


def cell(text) -> str:
    return str(text if text is not None else "").replace("|", "\\|").replace("\n", " ")


def md_table(header: list, rows: list) -> list:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header),
            *("| " + " | ".join(cell(x) for x in row) + " |" for row in rows)]


def where(i: dict) -> str:
    loc = f"{i['file']}:{i['line']}" if i.get("file") and i.get("line") else i.get("file") or ""
    sym = i.get("symbol") or ""
    return f"`{loc}`" + (f" ({sym})" if sym and sym != "<module>" and len(sym) < 60 else "") if loc else sym


def render_report(repo: Path, data: dict, d: Path) -> str:
    base = load_snapshot(d, "baseline") if (d / "snapshots" / "baseline.json").exists() else None
    latest = latest_snapshot(d)
    status = data["release_status"].upper()
    reason = "" if leak_scan_current(data, repo) else " (the leak scan is not current: run `leaks` again)"
    blockers = open_issues(data, "critical", "high")
    lines = [f"# Cleanup report: {repo.name}", "",
             f"Commit `{data['commit'][:8]}` · generated {data['generated'][:10]} · "
             f"release status: **{status}**{reason}"
             + (f" · {len(open_issues(data, 'critical'))} critical and {len(open_issues(data, 'high'))} high "
                "issues open: the High phase is what blocks the release" if blockers else ""), "",
             "Source of truth: `.cleanup/issues.json`. This file is generated by `cleanup.py report`.", "",
             "## How to work this report", "", HOW_TO, "", "## Scorecard", ""]
    if latest:
        cols = [base] if base else []
        if not base or latest["label"] != base["label"]:
            cols.append(latest)
        header = ["metric", *(s["label"] for s in cols), *(["Δ"] if len(cols) == 2 else []),
                  "reference (paper: human / agent)"]
        lines += md_table(header, scorecard(cols))
        if not_measured := latest.get("not_measured"):
            lines += ["", "Not measured: " + "; ".join(f"{k} ({v[:80]})" for k, v in not_measured.items())]
        if len(cols) == 2:
            lines += [f"\n> warning: {w}" for w in comparison_warnings(cols[0], cols[1])]
    else:
        lines.append("No snapshot yet: run `cleanup.py measure`.")

    for phase, sevs in PHASES:
        items = open_issues(data, *sevs)
        lines += ["", f"## {phase} phase ({len(items)} open)", ""]
        if not items:
            lines.append("Nothing open.")
            continue
        done = phase != "Low"  # Low issues are optional: no done-when column
        rows, prev = [], {}
        for i in items:
            same = {k: "same as above" if i[k] and i[k] == prev.get(k) else i[k] for k in ("fix", "done_when")}
            rows.append([i["id"], i["severity"], where(i), i["summary"], same["fix"],
                         *([same["done_when"]] if done else []), i["effort"]])
            prev = i
        lines += md_table(["ID", "severity", "where", "problem", "fix", *(["done when"] if done else []), "effort"],
                          rows)
    dismissed = [i for i in data["issues"] if i["status"] == "dismissed"]
    fixed = [i for i in data["issues"] if i["status"] == "fixed"]
    lines += ["", f"## Dismissed ({len(dismissed)})", ""]
    lines += (md_table(["ID", "where", "problem", "reason"],
                       [[i["id"], where(i), i["summary"], i["note"]] for i in dismissed]) if dismissed else ["None."])
    lines += ["", f"Fixed so far: {len(fixed)} issue(s).", "", "## Severity rules", "",
              "Set by `POLICY` in unslop/common.py; red flags and bugs you add keep the severity you give them.", ""]
    lines += [f"- **{sev}:** " + "; ".join(when for s, when in POLICY.values() if s == sev) for sev in SEVERITIES]
    return "\n".join(lines) + "\n"


def cmd_report(a):
    repo = repo_root(a.repo)
    d = state_dir(repo)
    data = load_issues(d)
    require_leak_scan(data, repo)
    save_issues(d, data, repo)  # refresh release status
    recount_head_snapshot(d, data, repo)
    (d / "report.md").write_text(render_report(repo, data, d))
    if SCRIPTS != d.resolve():  # lets another agent re-measure without the skill
        shutil.copy2(SCRIPTS / "cleanup.py", d)
        shutil.copy2(RULES, d)
        shutil.copytree(SCRIPTS / "unslop", d / "unslop", dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__"))
    o = {s: len(open_issues(data, s)) for s in SEVERITIES}
    print(f"release: {data['release_status'].upper()}  ·  open " + " · ".join(f"{o[s]} {s}" for s in SEVERITIES))
    print(f"-> {d / 'report.md'}")
