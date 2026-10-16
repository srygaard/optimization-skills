"""`cleanup.py leaks`: the mandatory leak scan, Betterleaks and large files over the full history."""

import hashlib
from pathlib import Path

from common import Severity, git, head, now, repo_root, run
from state import load_issues, merge, save_issues, state_dir
from tools import betterleaks, large_files


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
    repo = repo_root(a.repo)
    d = state_dir(repo)
    data = load_issues(d)
    exe = betterleaks.executable()
    version = run([exe, "version"]).stdout.strip() or "?"
    # --internal patterns are remembered, so a later scan without them doesn't mark their issues fixed
    internal = list(dict.fromkeys([*data.get("leak_scan", {}).get("internal", []), *a.internal]))
    rules = betterleaks.rule_set(internal)
    found = betterleaks.scan(repo, exe, rules) + large_files.scan(repo)
    merge(data, found, {betterleaks.TOOL.name, large_files.TOOL.name})
    data["tools"][betterleaks.TOOL.name] = version
    data["leak_scan"] = {
        "commit": head(repo), "date": now(), "betterleaks": version, "internal": internal,
        "rules_sha256": hashlib.sha256(rules.encode()).hexdigest()[:12],
    }
    save_issues(d, data, repo)
    print_leaks(repo, data, found, version)
    untracked = git(repo, "ls-files", "--others", "--exclude-standard").split()
    if untracked:
        print(f"warning: {len(untracked)} untracked files were not scanned (e.g. {untracked[0]}); "
              "commit or ignore them and re-run")
