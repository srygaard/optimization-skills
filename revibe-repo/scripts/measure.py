"""`cleanup.py measure`: run the tools, merge their issues, record a snapshot in issues.json."""

import sys
from importlib.metadata import PackageNotFoundError, version

from report import print_scorecard
from scope import Scope, shipped_files
from state import Cleanup, merge, now
from tools import ANALYZERS, MEASURE, vulture
from tools.base import ToolError
from utils.git import git, head


# --- measure ---------------------------------------------------------------------------------------


def tool_versions() -> dict:
    """{tool: version} of the analyzers that cleanup.py's script header pins. Stops when one is missing, as
    it is when the script runs without uv."""
    try:
        return {m.TOOL.name: version(m.TOOL.package) for m in ANALYZERS}
    except PackageNotFoundError as e:
        raise SystemExit(f"{e.name} is not installed: run `uv run --script cleanup.py ...`, which installs the analyzers "
            "pinned in its header")


def run_steps(scope, steps=MEASURE) -> tuple:
    """Run the measure steps: (issues, metrics, not_measured, measured). A step that fails or has nothing
    to read is not measured, and one broken step doesn't lose the others' results."""
    found = []
    metrics = {}
    not_measured = {}
    measured = set()
    for step in steps:
        try:
            r = step.measure(scope)
        except Exception as e:
            not_measured[step.TOOL.name] = str(e) if isinstance(e, ToolError) else f"error: {e!r}"
            continue
        found += r.issues
        metrics.update(r.metrics)
        not_measured.update(r.notes or {})
        if r.skipped:
            not_measured[step.TOOL.name] = r.skipped
        else:
            measured.add(step.TOOL.name)
    return found, metrics, not_measured, measured


def cmd_measure(a):
    """`cleanup.py measure`: run every measure step, fold their issues into issues.json, record a snapshot."""
    cleanup = Cleanup(a.repo)
    cleanup.require_leak_scan()
    repo = cleanup.repo
    data = cleanup.data
    tools = tool_versions()  # fail before any step runs
    default_excludes = ("tests/**", "test/**", "**/test_*.py", "**/*_test.py", "**/conftest.py", "docs/**",
                        ".cleanup/**")
    excludes = list(dict.fromkeys([*default_excludes, *a.exclude]))
    scope = Scope(repo, shipped_files(repo, excludes))
    dirty = bool(git(repo, "status", "--porcelain", "--untracked-files=no").strip())
    snap = {"label": a.label, "date": now(), "commit": head(repo), "dirty": dirty, "scope": excludes,
            "tools": tools, "files": len(scope.files), "metrics": {}, "not_measured": {}}
    found, metrics, not_measured, measured = run_steps(scope)
    snap["metrics"].update(metrics)
    snap["not_measured"].update(not_measured)
    if scope.syntax_errors:
        snap["not_measured"]["parse"] = f"{len(scope.syntax_errors)} files don't parse: {scope.syntax_errors[:3]}"
    vulture.mark_inside_dead_code(found)
    merge(data, found, measured)
    retired = {"ruff": "scb-check reports complexity now"}  # sources no longer run: their open issues are dismissed
    for i in cleanup.open_issues():
        if i.get("source") in retired:
            i["status"] = "dismissed"
            i["note"] = f"no longer checked: {retired[i['source']]}"
    data["tools"].update(tools)
    if a.label in data["snapshots"]:
        print(f"note: overwriting snapshot {a.label}", file=sys.stderr)
    data["snapshots"][a.label] = snap
    cleanup.save()  # also counts the issues into the snapshot
    print_scorecard(snap)
    print(f"-> {cleanup.path}")
