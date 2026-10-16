"""`cleanup.py measure`: run the tools, merge their issues, write a snapshot."""

import re
import sys
from importlib.metadata import PackageNotFoundError, version

from common import git, head, now, repo_root
from report import print_scorecard
from scope import CodeIndex, shipped_files
from state import (
    count_issues, load_issues, merge, open_issues, require_leak_scan, save_issues, state_dir,
    write_snapshot,
)
from tools import ANALYZERS, MEASURE, vulture
from tools.base import ToolError


# --- measure ---------------------------------------------------------------------------------------


def tool_versions() -> dict:
    """{tool: version} of the analyzers that cleanup.py's script header pins. Stops when one is missing, as
    it is when the script runs without uv."""
    try:
        return {m.TOOL.name: version(m.TOOL.package) for m in ANALYZERS}
    except PackageNotFoundError as e:
        raise SystemExit(f"{e.name} is not installed: run `uv run --script cleanup.py ...`, which installs the analyzers "
            "pinned in its header")


def run_steps(repo, files, index, steps=MEASURE) -> tuple:
    """Run the measure steps: (issues, metrics, not_measured, measured). A step that fails or has nothing
    to read is not measured, and one broken step doesn't lose the others' results."""
    found = []
    metrics = {}
    not_measured = {}
    measured = set()
    for step in steps:
        try:
            r = step.measure(repo, files, index)
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


def count_suppressions(repo, files: list) -> int:
    pattern = re.compile(r"#\s*(?:noqa|scbc\s+ignore)", re.I)
    return sum(len(pattern.findall((repo / f).read_text(errors="replace"))) for f in files)


def cmd_measure(a):
    """`cleanup.py measure`: run every measure step, fold their issues into issues.json, write a snapshot."""
    repo = repo_root(a.repo)
    d = state_dir(repo)
    data = load_issues(d)
    require_leak_scan(data, repo)
    tools = tool_versions()  # fail before any step runs
    default_excludes = ("tests/**", "test/**", "**/test_*.py", "**/*_test.py", "**/conftest.py", "docs/**",
                        ".cleanup/**")
    excludes = list(dict.fromkeys([*default_excludes, *a.exclude]))
    files = shipped_files(repo, excludes)
    index = CodeIndex(repo, files)
    dirty = bool(git(repo, "status", "--porcelain", "--untracked-files=no").strip())
    snap = {"label": a.label, "date": now(), "commit": head(repo), "dirty": dirty, "scope": excludes,
            "tools": tools, "files": len(files), "metrics": {}, "not_measured": {}}
    found, metrics, not_measured, measured = run_steps(repo, files, index)
    snap["metrics"].update(metrics)
    snap["not_measured"].update(not_measured)
    snap["metrics"]["suppressions"] = count_suppressions(repo, files)
    if index.syntax_errors:
        snap["not_measured"]["parse"] = f"{len(index.syntax_errors)} files don't parse: {index.syntax_errors[:3]}"
    vulture.mark_inside_dead_code(found)
    merge(data, found, measured)
    retired = {"ruff": "scb-check reports complexity now"}  # sources no longer run: their open issues are dismissed
    for i in open_issues(data):
        if i.get("source") in retired:
            i["status"] = "dismissed"
            i["note"] = f"no longer checked: {retired[i['source']]}"
    data["tools"].update(tools)
    save_issues(d, data, repo)
    count_issues(snap, data, repo)
    if (d / "snapshots" / f"{a.label}.json").exists():
        print(f"note: overwriting snapshot {a.label}", file=sys.stderr)
    path = write_snapshot(d, snap)
    print_scorecard(snap)
    print(f"-> {d / 'issues.json'}  ·  {path}")
