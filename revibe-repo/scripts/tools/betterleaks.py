"""Betterleaks over the full git history: credentials and internal references. It is not on PyPI, so a
pinned release binary is downloaded once, checked against its SHA-256 and cached."""

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

from utils.git import git
from utils.process import run
from tools.base import Tool

TOOL = Tool(
    name="betterleaks",
    policy={
        "credential": ("critical", "credential matched by a confident rule, at HEAD or only in history"),
        "possible-credential": ("high", "possible credential (low-confidence rule or sensitive file name)"),
        "internal-head": ("high", "internal path, host, email or account at HEAD"),
        "internal-history": ("low", "internal reference only in history: a fresh squashed history drops it"),
    },
    settings={
        "version": "1.9.0",  # not on PyPI: the release binary is pinned here, with its checksums
        "sha256": {  # checksums.txt of the v1.9.0 release, signature verified with sigstore
            ("darwin", "arm64"): "83dd7eaab13d44a1e8e347a17064cdef153a1478db473276d41b38d55613d3f2",
            ("darwin", "x64"): "68dfd83458d9e7f90663daa7a81c7f944083b170dae82e4f5f8d0581ed519ef1",
            ("linux", "arm64"): "1d39116e0a58dc94574715e2aa12a2dbd5062f193eee3fec011fef6ba06bd13b",
            ("linux", "x64"): "f8b185a39ffcece2a1ca82bf3a4e7435cd81963ffd16b7a9128daf75f35f6de7",
        },
    },
)

# --- the binary ------------------------------------------------------------------------------------


def _platform() -> tuple:
    os_name = {"darwin": "darwin", "linux": "linux"}.get(sys.platform)
    machine = platform.machine().lower()
    arch = {"x86_64": "x64", "amd64": "x64", "arm64": "arm64", "aarch64": "arm64"}.get(machine)
    return os_name, arch


def tarball_ok(data: bytes, os_name: str, arch: str) -> bool:
    return hashlib.sha256(data).hexdigest() == TOOL.settings["sha256"].get((os_name, arch))


def _download(url: str) -> bytes:
    try:
        with urllib.request.urlopen(url, timeout=120) as r:
            return r.read()
    except Exception as e:  # a standalone Python may lack a CA bundle: fall back to curl
        if curl := shutil.which("curl"):
            p = subprocess.run([curl, "-fsSL", url], capture_output=True, timeout=600)
            if p.returncode == 0:
                return p.stdout
        raise SystemExit(f"could not download {url} ({e}). Offline? Fetch it elsewhere, check its SHA-256 "
            "and set BETTERLEAKS=/path/to/betterleaks")


def executable() -> Path:
    if env := os.environ.get("BETTERLEAKS"):
        exe = Path(env).expanduser()
        if not (exe.is_file() and os.access(exe, os.X_OK)):
            raise SystemExit(f"BETTERLEAKS={env} is not an executable file")
        return exe
    os_name, arch = _platform()
    if (os_name, arch) not in TOOL.settings["sha256"]:
        raise SystemExit(f"no pinned Betterleaks build for {sys.platform}/{platform.machine()}; "
            "set BETTERLEAKS=/path/to/betterleaks")
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    version = TOOL.settings["version"]
    cache = cache / "revibe-repo" / f"betterleaks-{version}"
    exe = cache / "betterleaks"
    if exe.is_file():
        return exe
    name = f"betterleaks_{version}_{os_name}_{arch}.tar.gz"
    print(f"fetching {name} (once) ...", file=sys.stderr)
    data = _download(f"https://github.com/betterleaks/betterleaks/releases/download/v{version}/{name}")
    if not tarball_ok(data, os_name, arch):
        raise SystemExit(f"SHA-256 mismatch for {name}: refusing to run it")
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        member = tar.getmember("betterleaks")
        if not member.isfile():
            raise SystemExit(f"{name} does not contain a betterleaks binary")
        blob = tar.extractfile(member).read()
    cache.mkdir(parents=True, exist_ok=True)
    part = cache / "betterleaks.part"
    part.write_bytes(blob)
    part.chmod(0o755)
    part.replace(exe)
    return exe


def _internal_rule(n: int, regex: str) -> str:
    if "'''" in regex:
        raise SystemExit("--internal patterns cannot contain '''")
    return f"\n[[rules]]\nid = \"internal-custom-{n}\"\nregex = '''({regex})'''\n"


def rule_set(internal: list) -> str:
    """betterleaks.toml, next to this file, plus one rule per --internal pattern."""
    rules = Path(__file__).with_suffix(".toml").read_text()
    return rules + "".join(_internal_rule(n, rx) for n, rx in enumerate(internal, 1))


# --- findings to issues ----------------------------------------------------------------------------



def preview(secret: str) -> str:
    """How a secret is shown: its public prefix, if it has one, and its length. Nothing of the secret part."""
    # public prefixes that name a token's kind; everything after them is secret
    known = r"^(?:gh[pousr]_|github_pat_|hf_|sk-(?:proj-|ant-)?|xox[abprs]-|glpat-|npm_|pypi-|AKIA|ASIA|AIza)"
    prefix = re.match(known, secret)
    return f"{prefix[0] if prefix else ''}… ({len(secret)} chars)"


def group_findings(findings: list) -> dict:
    """{basis: group} of Betterleaks findings, grouped the way they get fixed: a credential wherever it
    appears (it is rotated once), an internal reference per rule and file. Groups hold the plaintext
    (`needles`, `matches`); issues get only `preview`."""
    groups = {}
    for f in findings:
        secret = f.get("Secret") or ""
        internal = f["RuleID"].startswith("internal-")
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
            raise SystemExit(f"git grep failed: {p.stderr.strip()[-200:]}")
        hits[needle] = {x.split(":", 1)[1] for x in p.stdout.split("\0") if x}
    pattern.unlink(missing_ok=True)
    tree = set(git(repo, "ls-tree", "-r", "-z", "--name-only", "HEAD").split("\0"))
    for g in groups.values():
        g["head_files"] = sorted(set().union(*(hits[n] for n in g["needles"])) if g["needles"] else g["paths"] & tree)


def located(g: dict) -> tuple:
    """(rule, issue arguments) shared by both kinds of group: the most confident and specific rule, where the
    group is now, and the evidence."""
    rank = {"high": 0, "medium": 1, "low": 2}
    rule = min(g["rules"], key=lambda r: (rank.get(g["rules"][r], 9), "generic" in r, r))
    now_in = g["head_files"]
    file, line = g["example"]
    if now_in and file not in now_in:
        file = now_in[0]
        line = None
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
    return TOOL.issue(
        "INT", "internal-ref", rule, "internal-head" if here else "internal-history",
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
    return TOOL.issue(
        "LEAK", "leak", rule, "credential" if certain else "possible-credential",
        f"{'Secret' if certain else 'Possible secret'} `{g['preview']}` matched {rule} "
        f"({'at HEAD' if here else 'only in history'})",
        key=f"LEAK|{key}", **at, fix=fix,
        done_when="Rotated (user confirms), gone from HEAD, and dismissed with a note on how history is handled.")


def leak_issues(repo: Path, findings: list, tmp: Path) -> list:
    """One issue per group of Betterleaks findings. Plaintext stays in the groups."""
    groups = group_findings(findings)
    find_at_head(repo, groups, tmp)
    return [internal_issue(g) if g["internal"] else credential_issue(g) for g in groups.values()]


def scan(repo: Path, exe: Path, rules: str) -> list:
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
            raise SystemExit(f"betterleaks failed (exit {p.returncode}): {p.stderr.strip()[-400:]}")
        return leak_issues(repo, json.loads(report.read_text() or "[]"), Path(tmp))
