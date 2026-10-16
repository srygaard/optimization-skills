"""`cleanup.py leaks`: the mandatory leak scan, Betterleaks over the full history."""

from pathlib import Path

from state import Cleanup, Severity, merge, now
from tools import betterleaks
from utils.git import git, head


def print_leaks(repo: Path, data: dict, found: list, version: str):
    """Counts by severity, then the first 15 issues as issues.json now has them."""
    counts = " · ".join(f"{n} {s.name}" for s in Severity if (n := sum(i["severity"] == s.name for i in found)))
    print(f"leaks @ {head(repo)[:8]}  betterleaks {version}  {counts or 'nothing found'}")
    ids = {i["id"] for i in found}
    for i in [i for i in data["issues"] if i["id"] in ids][:15]:
        print(f"  {i['id']}  {i['severity']:<8} {i['status']:<9} {i['summary'][:90]}")
    if len(found) > 15:
        print(f"  ... {len(found) - 15} more in .cleanup/issues.json")


def cmd_leaks(a):
    """`cleanup.py leaks`: Betterleaks over the full history, folded into issues.json."""
    cleanup = Cleanup(a.repo)
    repo = cleanup.repo
    data = cleanup.data
    # --internal patterns are remembered, so a later scan without them doesn't mark their issues fixed
    internal = list(dict.fromkeys([*data.get("leak_scan", {}).get("internal", []), *a.internal]))
    leaks = betterleaks.scan(repo, internal)
    found = leaks.issues
    merge(data, found, {betterleaks.TOOL.name})
    version = leaks.metrics["version"]
    data["tools"][betterleaks.TOOL.name] = version
    data["leak_scan"] = {
        "commit": head(repo), "date": now(), "betterleaks": version, "internal": internal,
        "rules_sha256": leaks.metrics["rules_sha256"],
    }
    cleanup.save()
    print_leaks(repo, data, found, version)
    untracked = git(repo, "ls-files", "--others", "--exclude-standard").split()
    if untracked:
        print(f"warning: {len(untracked)} untracked files were not scanned (e.g. {untracked[0]}); "
              "commit or ignore them and re-run")
