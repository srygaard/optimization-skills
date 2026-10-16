"""`cleanup.py leaks`: the mandatory leak scan, over the full history once and over HEAD after that."""

from pathlib import Path

from state import Cleanup, Severity, merge, now
from tools import LEAK_SCAN
from utils.git import git, head


def print_leaks(repo: Path, data: dict, found: list, scanners: str):
    """Counts by severity, then the first 15 issues as issues.json now has them. `scanners` names the tools
    and their versions."""
    counts = " · ".join(f"{n} {s.name}" for s in Severity if (n := sum(i["severity"] == s.name for i in found)))
    print(f"leaks @ {head(repo)[:8]}  {scanners}  {counts or 'nothing found'}")
    ids = {i["id"] for i in found}
    for i in [i for i in data["issues"] if i["id"] in ids][:15]:
        print(f"  {i['id']}  {i['severity']:<8} {i['status']:<9} {i['summary'][:90]}")
    if len(found) > 15:
        print(f"  ... {len(found) - 15} more in .revibe/issues.json")


def cmd_leaks(a):
    """`cleanup.py leaks`: the leak scan, folded into issues.json. It reads the full history the first time,
    with `--history`, or when its rules changed, and only the files at HEAD otherwise."""
    cleanup = Cleanup(a.repo)
    repo = cleanup.repo
    data = cleanup.data
    tool = LEAK_SCAN
    last = data.get("leak_scan", {})
    # --internal patterns are remembered, so a later scan without them doesn't mark their issues fixed
    internal = list(dict.fromkeys([*last.get("internal", []), *a.internal]))
    known = [i for i in data["issues"] if i["source"] == tool.name]
    result = tool.scan(repo, internal, {} if a.history else last, known)
    merge(data, result.issues, {tool.name})
    scanned = dict(result.metrics)  # its version, by name; the rules' hash; whether it read the history
    read_history = scanned.pop("history")
    data["tools"][tool.name] = scanned[tool.name]
    data["leak_scan"] = {"commit": cleanup.commit, "date": now(), **scanned, "internal": internal,
                         "history": cleanup.commit if read_history else last["history"]}  # the commit it last read
    cleanup.save()
    scope = "full history" if read_history else "HEAD"
    print_leaks(repo, data, result.issues, f"{tool.name} {scanned[tool.name]}, {scope}")
    untracked = git(repo, "ls-files", "--others", "--exclude-standard").split()
    if untracked:
        print(f"warning: {len(untracked)} untracked files were not scanned (e.g. {untracked[0]}); "
              "commit or ignore them and re-run")
