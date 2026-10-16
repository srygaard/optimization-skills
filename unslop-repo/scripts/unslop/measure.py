"""`cleanup.py measure`: run the steps, merge their issues, write a snapshot."""

import sys

from .code import CodeIndex, scb_check, shipped_files, vulture_dead
from .common import (
    CleanupError, DEFAULT_EXCLUDES, POLICY, RETIRED_SOURCES, TOOLS, ToolError, git, head, now, repo_root,
    uvx,
)
from .project import count_suppressions, deptry_deps, file_lengths, license_readme, pip_audit_vulns
from .report import print_scorecard
from .state import (
    count_issues, load_issues, merge, open_issues, require_leak_scan, save_issues, state_dir,
    write_snapshot,
)


# --- measure ---------------------------------------------------------------------------------------

STEPS = (("scb-check", scb_check), ("vulture", vulture_dead),
         ("deptry", deptry_deps), ("pip-audit", pip_audit_vulns), ("license", license_readme),
         ("file-length", file_lengths))


def run_steps(repo, files, index, steps=None) -> tuple:
    """Run the measure steps: (issues, metrics, not_measured, measured). A step that fails or has nothing
    to read is not measured, and one broken step doesn't lose the others' results."""
    found = []
    metrics = {}
    not_measured = {}
    measured = set()
    for name, step in steps or STEPS:
        try:
            r = step(repo, files, index)
        except Exception as e:
            not_measured[name] = str(e) if isinstance(e, (ToolError, CleanupError)) else f"error: {e!r}"
            continue
        for i in r.issues:
            i["source"] = name  # merge closes issues by source
        found += r.issues
        metrics.update(r.metrics)
        not_measured.update(r.notes or {})
        if r.skipped:
            not_measured[name] = r.skipped
        else:
            measured.add(name)
    return found, metrics, not_measured, measured


def mark_inside_dead_code(found: list):
    """Complexity and red-flag issues inside a dead-code candidate drop to low: delete before refactoring."""
    dead = [i for i in found if i["category"] == "dead-code" and i["status"] == "open"]
    for i in found:
        if i["category"] not in ("complexity", "red-flag") or not i.get("line"):
            continue
        host = next((d for d in dead if d["file"] == i["file"] and d["line"] <= i["line"] <= d["end_line"]), None)
        if host:
            i["severity"] = POLICY["inside-dead-code"][0]
            i["evidence"]["inside_dead_code"] = host["id"]
            i["fix"] = f"Inside dead-code candidate {host['id']}: delete that instead, unless it turns out to be used. " + i["fix"]


def cmd_measure(a):
    repo = repo_root(a.repo)
    d = state_dir(repo)
    data = load_issues(d)
    require_leak_scan(data, repo)
    uvx()  # fail before any step runs
    excludes = list(dict.fromkeys([*DEFAULT_EXCLUDES, *a.exclude]))
    files = shipped_files(repo, excludes)
    index = CodeIndex(repo, files)
    dirty = bool(git(repo, "status", "--porcelain", "--untracked-files=no").strip())
    snap = {"label": a.label, "date": now(), "commit": head(repo), "dirty": dirty, "scope": excludes,
            "tools": dict(TOOLS), "files": len(files), "metrics": {}, "not_measured": {}}
    found, metrics, not_measured, measured = run_steps(repo, files, index)
    snap["metrics"].update(metrics)
    snap["not_measured"].update(not_measured)
    snap["metrics"]["suppressions"] = count_suppressions(repo, files)
    if index.syntax_errors:
        snap["not_measured"]["parse"] = f"{len(index.syntax_errors)} files don't parse: {index.syntax_errors[:3]}"
    mark_inside_dead_code(found)
    merge(data, found, measured)
    for i in open_issues(data):
        if i.get("source") in RETIRED_SOURCES:
            i["status"], i["note"] = "dismissed", f"no longer checked: {RETIRED_SOURCES[i['source']]}"
    data["tools"].update(TOOLS)
    save_issues(d, data, repo)
    count_issues(snap, data, repo)
    if (d / "snapshots" / f"{a.label}.json").exists():
        print(f"note: overwriting snapshot {a.label}", file=sys.stderr)
    path = write_snapshot(d, snap)
    print_scorecard(snap)
    print(f"-> {d / 'issues.json'}  ·  {path}")
