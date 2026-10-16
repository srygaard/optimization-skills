"""`cleanup.py measure`: run the tools, merge their issues, record the metrics of this commit."""

import sys
from importlib.metadata import PackageNotFoundError, version

from report import print_scorecard
from scope import Scope, shipped_files
from state import Cleanup, merge, now
from tools import ANALYZERS, MEASURE
from tools.base import ToolError
from utils.git import git


# --- measure ---------------------------------------------------------------------------------------


def tool_versions() -> dict:
    """{tool: version} of the analyzers. Stops unless each is installed at the version its class pins, as it
    is when the script runs through uv: numbers from other versions don't compare."""
    for m in ANALYZERS:
        try:
            installed = version(m.package)
        except PackageNotFoundError:
            installed = ""
        if installed != m.version:
            found = f"{installed} is installed" if installed else "it is not installed"
            raise SystemExit(f"{m.package} {m.version} is pinned but {found}: run `uv run --script cleanup.py ...`, "
                             "which installs the versions its header pins")
    return {m.name: m.version for m in ANALYZERS}


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
            not_measured[step.name] = str(e) if isinstance(e, ToolError) else f"error: {e!r}"
            continue
        found += r.issues
        metrics.update(r.metrics)
        not_measured.update(r.notes or {})
        if r.skipped:
            not_measured[step.name] = r.skipped
        else:
            measured.add(step.name)
    for step in steps:
        step.adjust(found)
    return found, metrics, not_measured, measured


def cmd_measure(a):
    """`cleanup.py measure`: run every measure step, fold their issues into issues.json, and write this commit's
    metrics to .revibe/<commit>/metrics.json."""
    cleanup = Cleanup(a.repo)
    cleanup.require_leak_scan()
    repo = cleanup.repo
    data = cleanup.data
    tools = tool_versions()  # fail before any step runs
    default_excludes = ("tests/**", "test/**", "**/test_*.py", "**/*_test.py", "**/conftest.py", "docs/**",
                        ".revibe/**")
    excludes = list(dict.fromkeys([*default_excludes, *a.exclude]))
    scope = Scope(repo, shipped_files(repo, excludes))
    dirty = bool(git(repo, "status", "--porcelain", "--untracked-files=no").strip())
    snap = {"label": a.label, "date": now(), "commit": cleanup.commit, "dirty": dirty, "scope": excludes,
            "tools": tools, "files": len(scope.files), "metrics": {}, "not_measured": {}}
    found, metrics, not_measured, measured = run_steps(scope)
    snap["metrics"].update(metrics)
    snap["not_measured"].update(not_measured)
    if scope.syntax_errors:
        snap["not_measured"]["parse"] = f"{len(scope.syntax_errors)} files don't parse: {scope.syntax_errors[:3]}"
    merge(data, found, measured)
    retired = {  # sources no longer run: their open issues are dismissed
        "ruff": "scb-check reports complexity now",
        "git-history": "large files are checked at HEAD only now, by measure",
    }
    for i in cleanup.open_issues():
        if i.get("source") in retired:
            i["status"] = "dismissed"
            i["note"] = f"no longer checked: {retired[i['source']]}"
    data["tools"].update(tools)
    if cleanup.snapshot_path(cleanup.commit).exists():
        print(f"note: replacing the metrics of {cleanup.commit[:8]}, measured before", file=sys.stderr)
    path = cleanup.write_snapshot(snap)
    cleanup.save()
    print_scorecard(snap)
    print(f"-> {cleanup.path}  ·  {path}")
