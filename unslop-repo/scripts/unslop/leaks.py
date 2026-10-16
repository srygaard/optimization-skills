"""The mandatory leak scan: Betterleaks over the full history, and large files in history."""

import hashlib
import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from .common import (
    BETTERLEAKS_SHA256, BETTERLEAKS_VERSION, LARGE_FILE_MB, RULES, SEVERITIES, die, git, head, now,
    repo_root, run,
)
from .state import issue, load_issues, merge, save_issues, state_dir


# --- Betterleaks -----------------------------------------------------------------------------------


def _platform() -> tuple:
    os_name = {"darwin": "darwin", "linux": "linux"}.get(sys.platform)
    machine = platform.machine().lower()
    arch = {"x86_64": "x64", "amd64": "x64", "arm64": "arm64", "aarch64": "arm64"}.get(machine)
    return os_name, arch


def tarball_ok(data: bytes, os_name: str, arch: str) -> bool:
    return hashlib.sha256(data).hexdigest() == BETTERLEAKS_SHA256.get((os_name, arch))


def _download(url: str) -> bytes:
    try:
        with urllib.request.urlopen(url, timeout=120) as r:
            return r.read()
    except Exception as e:  # a standalone Python may lack a CA bundle: fall back to curl
        if curl := shutil.which("curl"):
            p = subprocess.run([curl, "-fsSL", url], capture_output=True, timeout=600)
            if p.returncode == 0:
                return p.stdout
        die(f"could not download {url} ({e}). Offline? Fetch it elsewhere, check its SHA-256 "
            "and set BETTERLEAKS=/path/to/betterleaks")


def betterleaks() -> Path:
    if env := os.environ.get("BETTERLEAKS"):
        exe = Path(env).expanduser()
        if not (exe.is_file() and os.access(exe, os.X_OK)):
            die(f"BETTERLEAKS={env} is not an executable file")
        return exe
    os_name, arch = _platform()
    if (os_name, arch) not in BETTERLEAKS_SHA256:
        die(f"no pinned Betterleaks build for {sys.platform}/{platform.machine()}; "
            "set BETTERLEAKS=/path/to/betterleaks")
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    cache = cache / "unslop-repo" / f"betterleaks-{BETTERLEAKS_VERSION}"
    exe = cache / "betterleaks"
    if exe.is_file():
        return exe
    name = f"betterleaks_{BETTERLEAKS_VERSION}_{os_name}_{arch}.tar.gz"
    print(f"fetching {name} (once) ...", file=sys.stderr)
    data = _download(f"https://github.com/betterleaks/betterleaks/releases/download/"
                     f"v{BETTERLEAKS_VERSION}/{name}")
    if not tarball_ok(data, os_name, arch):
        die(f"SHA-256 mismatch for {name}: refusing to run it")
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        member = tar.getmember("betterleaks")
        if not member.isfile():
            die(f"{name} does not contain a betterleaks binary")
        blob = tar.extractfile(member).read()
    cache.mkdir(parents=True, exist_ok=True)
    part = cache / "betterleaks.part"
    part.write_bytes(blob)
    part.chmod(0o755)
    part.replace(exe)
    return exe


def _internal_rule(n: int, regex: str) -> str:
    if "'''" in regex:
        die("--internal patterns cannot contain '''")
    return (f'\n[[rules]]\nid = "internal-custom-{n}"\ndescription = "Internal reference (--internal)"\n'
            f"regex = '''({regex})'''\nconfidence = \"medium\"\ntags = [\"internal\"]\n")


# --- leaks -----------------------------------------------------------------------------------------

CONFIDENCE_RANK = {"high": 0, "medium": 1, "low": 2}
# public prefixes that name a token's kind; everything after them is secret
KNOWN_PREFIX = re.compile(r"^(?:gh[pousr]_|github_pat_|hf_|sk-(?:proj-|ant-)?|xox[abprs]-|glpat-|npm_|pypi-|AKIA|ASIA|AIza)")


def preview(secret: str) -> str:
    """How a secret is shown: its public prefix, if it has one, and its length. Nothing of the secret part."""
    prefix = KNOWN_PREFIX.match(secret)
    return f"{prefix[0] if prefix else ''}… ({len(secret)} chars)"


def group_findings(findings: list) -> dict:
    """{basis: group} of Betterleaks findings, grouped the way they get fixed: a credential wherever it
    appears (it is rotated once), an internal reference per rule and file. Groups hold the plaintext
    (`needles`, `matches`); issues get only `preview`."""
    groups = {}
    for f in findings:
        secret = f.get("Secret") or ""
        internal = "internal" in (f.get("Tags") or [])
        basis = f"internal|{f['RuleID']}|{f['File']}" if internal else secret or f"path:{f['File']}"
        first = (f.get("Date", ""), f["Commit"], f["File"], f["StartLine"])
        g = groups.setdefault(basis, {
            "basis": basis, "rules": {}, "internal": internal, "locs": [], "fingerprints": [], "needles": set(),
            "paths": set(), "first": first, "example": (f["File"], f["StartLine"]), "matches": set(),
            "preview": preview(secret) if secret else "(file name)",
        })
        g["rules"][f["RuleID"]] = (f.get("Attributes") or {}).get("confidence", "high")
        if internal:
            g["matches"].add(secret[:80])
            g["preview"] = min(g["matches"])
        g["locs"].append(f"{f['File']}:{f['StartLine']}@{f['Commit'][:8]}")
        g["fingerprints"].append(f["Fingerprint"])
        g["paths"].add(f["File"])
        if secret.strip():
            g["needles"].add(secret)
        g["first"] = min(g["first"], first)
    return groups


def find_at_head(repo: Path, groups: dict, tmp: Path):
    """Set each group's `head_files`: the files at HEAD that hold it. Betterleaks reports the commit and path
    that added a line, and a file may have been moved since, so HEAD's contents are searched; a finding on a
    file name is at HEAD while the path is. A needle reaches git through a file in the private temp dir,
    never a command line; a multi-line secret is searched by its longest line."""
    pattern = Path(tmp, "needle")
    hits = {}
    for needle in {n for g in groups.values() for n in g["needles"]}:
        pattern.write_text(max(needle.splitlines(), key=len) + "\n")
        p = run(["git", "-C", repo, "grep", "-l", "-z", "-F", "-f", pattern, "HEAD", "--"])
        if p.returncode not in (0, 1):  # 1: no match
            die(f"git grep failed: {p.stderr.strip()[-200:]}")
        hits[needle] = {x.split(":", 1)[1] for x in p.stdout.split("\0") if x}
    pattern.unlink(missing_ok=True)
    tree = set(git(repo, "ls-tree", "-r", "-z", "--name-only", "HEAD").split("\0"))
    for g in groups.values():
        g["head_files"] = sorted(set().union(*(hits[n] for n in g["needles"])) if g["needles"] else g["paths"] & tree)


def located(g: dict) -> tuple:
    """(rule, issue arguments) shared by both kinds of group: the most confident and specific rule, where the
    group is now, and the evidence."""
    rule = min(g["rules"], key=lambda r: (CONFIDENCE_RANK.get(g["rules"][r], 9), "generic" in r, r))
    now_in = g["head_files"]
    file, line = g["example"]
    if now_in and file not in now_in:
        file, line = now_in[0], None
    return rule, {"file": file, "line": line, "evidence": {
        "rules": sorted(g["rules"]), "preview": g["preview"], "at_head": bool(now_in), "head_files": now_in[:10],
        "first_commit": g["first"][1][:12], "first_date": g["first"][0],
        "occurrences": len(g["locs"]), "locations": g["locs"][:10], "fingerprints": g["fingerprints"][:10],
    }}


def internal_issue(g: dict) -> dict:
    rule, at = located(g)
    here = at["evidence"]["at_head"]
    more = f" and {len(g['matches']) - 1} more" if len(g["matches"]) > 1 else ""
    at["evidence"]["matches"] = sorted(g["matches"])[:20]
    return issue(
        "INT", "betterleaks", "internal-ref", rule, "internal-head" if here else "internal-history",
        f"Internal reference `{g['preview']}`{more} ({'at HEAD' if here else 'only in history'})",
        key=f"INT|{hashlib.sha256(g['basis'].encode()).hexdigest()}", **at,
        fix=("Replace it with a config value, CLI argument or environment variable "
             "(or a neutral placeholder in docs)." if here else
             "Only matters if the history is published; a fresh squashed history drops it."),
        done_when="`cleanup.py leaks` no longer reports it at HEAD.")


def credential_issue(g: dict) -> dict:
    rule, at = located(g)
    here = at["evidence"]["at_head"]
    # IDs come from facts about the finding, never from the secret: a hash of a short password next to its
    # prefix and length could be brute-forced offline
    key = "|".join(map(str, g["first"][1:])) if g["needles"] else hashlib.sha256(g["basis"].encode()).hexdigest()
    certain = g["rules"][rule] in ("high", "medium")
    if certain:
        fix = ("User: revoke/rotate this credential at its provider first. "
               + ("Then read it from an environment variable (add a `.env.example`, gitignore `.env`) "
                  "and keep it out of the published history: publish a fresh squashed history, or "
                  "rewrite history with the user's approval." if here else
                  "It is gone from HEAD but still in history: publish a fresh squashed history, or "
                  "rewrite history with the user's approval.")
               + " The scan keeps finding it in the local history, so then dismiss this issue with a note "
                 "saying it was rotated and how history is handled.")
    else:
        fix = ("Check whether this is a real credential by looking at the location; never print the "
               "value. If it is not, dismiss it with the reason. If it is, rotate it and remove it as for "
               "any secret.")
    return issue(
        "LEAK", "betterleaks", "leak", rule, "credential" if certain else "possible-credential",
        f"{'Secret' if certain else 'Possible secret'} `{g['preview']}` matched {rule} "
        f"({'at HEAD' if here else 'only in history'})",
        key=f"LEAK|{key}", **at, fix=fix,
        done_when="Rotated (user confirms), gone from HEAD, and dismissed with a note on how history is handled.")


def leak_issues(repo: Path, findings: list, tmp: Path) -> list:
    """One issue per group of Betterleaks findings. Plaintext stays in the groups."""
    groups = group_findings(findings)
    find_at_head(repo, groups, tmp)
    return [internal_issue(g) if g["internal"] else credential_issue(g) for g in groups.values()]


def large_file_issues(repo: Path) -> list:
    paths = {}
    for row in git(repo, "rev-list", "--objects", "--all").splitlines():
        sha, _, path = row.partition(" ")
        if path:
            paths.setdefault(sha, path)
    p = run(["git", "-C", repo, "cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)"],
            input="\n".join(paths) + "\n")
    at_head = {}  # blob -> its path at HEAD: compare blobs, not paths, so a moved file is still at HEAD
    for row in git(repo, "ls-tree", "-r", "-z", "HEAD").split("\0"):
        if row:
            meta, path = row.split("\t", 1)
            at_head[meta.split()[2]] = path
    sizes = {}  # path -> (largest size, whether a large blob of it is at HEAD)
    for row in p.stdout.splitlines():
        sha, kind, size = (row.split() + ["", ""])[:3]
        if kind == "blob" and int(size) > LARGE_FILE_MB * 1_000_000:
            path = at_head.get(sha) or paths[sha]
            old_size, old_here = sizes.get(path, (0, False))
            sizes[path] = (max(old_size, int(size)), old_here or sha in at_head)
    out = []
    for path, (size, here) in sorted(sizes.items(), key=lambda kv: -kv[1][0]):
        out.append(issue(
            "FILE", "git-history", "large-file", "large-file", "large-file-head" if here else "large-file-history",
            f"{size / 1e6:.1f} MB file `{path}` ({'at HEAD' if here else 'only in history'})",
            file=path, evidence={"bytes": size, "at_head": here}, effort="M",
            fix=("Take it out of the repo: host weights or data elsewhere (release asset, Hugging Face Hub, "
                 "Zenodo) and download on demand; check that you may redistribute it." if here else
                 "Only matters if the history is published; a fresh squashed history drops it."),
            done_when="Not in the tree, and not in any history that will be published."))
    return out


def scan_history(repo: Path, exe: Path, rules: str) -> list:
    """Betterleaks over every commit, as issues."""
    with tempfile.TemporaryDirectory(prefix="cleanup-leaks-") as tmp:
        config = Path(tmp, "rules.toml")
        report = Path(tmp, "report.json")
        mirror = Path(tmp, "scan.git")
        config.write_text(rules)
        # Trust neither allow comments nor the repo's ignore files: a key waved past a pre-commit hook is what
        # this scan is for, and dismissals live in issues.json. Betterleaks reads ignore files from the root
        # of the repo it scans, so it scans a bare mirror (--shared copies no objects); tmp has none either.
        git(repo, "clone", "--quiet", "--mirror", "--shared", str(repo), str(mirror))
        p = run([exe, "git", mirror, "--config", config, "--report-format", "json", "--report-path", report,
                 "--exit-code", "0", "--no-banner", "--no-color", "--log-level", "error",
                 "--ignore-gitleaks-allow", "--gitleaks-ignore-path", tmp])
        if p.returncode or not report.exists():
            die(f"betterleaks failed (exit {p.returncode}): {p.stderr.strip()[-400:]}")
        return leak_issues(repo, json.loads(report.read_text() or "[]"), Path(tmp))


def print_leaks(repo: Path, data: dict, found: list, version: str):
    """Counts by severity, then the first 15 issues as issues.json now has them."""
    counts = " · ".join(f"{n} {s}" for s in SEVERITIES if (n := sum(i["severity"] == s for i in found)))
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
    exe = betterleaks()
    version = run([exe, "version"]).stdout.strip() or "?"
    # --internal patterns are remembered, so a later scan without them doesn't mark their issues fixed
    internal = list(dict.fromkeys([*data.get("leak_scan", {}).get("internal", []), *a.internal]))
    rules = RULES.read_text() + "".join(_internal_rule(n, rx) for n, rx in enumerate(internal, 1))
    found = scan_history(repo, exe, rules) + large_file_issues(repo)
    merge(data, found, {"betterleaks", "git-history"})
    data["tools"]["betterleaks"] = version
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
