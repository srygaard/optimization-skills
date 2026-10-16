#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Measure, track and report the cleanup of a Python ML repo before open-sourcing.

    cleanup.py leaks   REPO [--internal REGEX ...]    secrets, internal refs, large files (run first)
    cleanup.py measure REPO [--label L] [--exclude G]  metrics snapshot + mechanical issues
    cleanup.py compare REPO BASE [--to LABEL]         before/after table and regression warnings
    cleanup.py report  REPO                           render REPO/.cleanup/report.md

Everything is written to REPO/.cleanup/, which ignores itself in git. Secret values never reach
stdout or any file: only a known public prefix (`ghp_`, `hf_`) and the length. Tests: test_cleanup.py.

Stdlib only. Analyzers run through pinned `uvx`. Betterleaks is not on PyPI, so it is downloaded once
as a pinned release binary, checked against SHA-256 values from the release's sigstore-signed
checksums.txt, and cached. Set BETTERLEAKS=/path/to/betterleaks to skip the download.
"""

import argparse
import ast
import datetime as dt
import hashlib
import io
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import urllib.request
from fnmatch import fnmatch
from pathlib import Path
from typing import NamedTuple

# --- pinned tools ----------------------------------------------------------------------------------

BETTERLEAKS_VERSION = "1.9.0"
BETTERLEAKS_SHA256 = {  # checksums.txt of the v1.9.0 release, signature verified with sigstore
    ("darwin", "arm64"): "83dd7eaab13d44a1e8e347a17064cdef153a1478db473276d41b38d55613d3f2",
    ("darwin", "x64"): "68dfd83458d9e7f90663daa7a81c7f944083b170dae82e4f5f8d0581ed519ef1",
    ("linux", "arm64"): "1d39116e0a58dc94574715e2aa12a2dbd5062f193eee3fec011fef6ba06bd13b",
    ("linux", "x64"): "f8b185a39ffcece2a1ca82bf3a4e7435cd81963ffd16b7a9128daf75f35f6de7",
}
TOOLS = {"scb-check": "0.2.0", "vulture": "2.16", "deptry": "0.25.1", "pip-audit": "2.10.1"}
RETIRED_SOURCES = {"ruff": "scb-check reports complexity now"}  # their open issues get dismissed

# --- thresholds ------------------------------------------------------------------------------------

CC_THRESHOLD = 10  # scb-check 0.2.0's threshold: Erosion is the share of mass in functions above it
MEDIUM_MASS = 100  # complexity mass CC x sqrt(SLOC) from which a function is medium
MEDIUM_DUP_LINES = 20  # duplicate-group weight lines x (copies - 1) from which it is medium
MIN_CLONE_LINES = 10  # smaller duplicate blocks still count in Clone Ratio but get no issue
MIN_DEAD_LINES = 5  # smaller unused symbols still count in dead-code LOC but get no issue
DEAD_HIGH_LINES = 20  # unused symbols at least this long are high: remove them before release
LARGE_FILE_MB = 5
LONG_FILE_LINES, WARN_FILE_LINES = 1000, 500  # source files longer than this should be split / are counted
DEFAULT_EXCLUDES = (
    "tests/**", "test/**", "**/test_*.py", "**/*_test.py", "**/conftest.py", "docs/**", ".cleanup/**",
)
# SlopCodeBench (arXiv 2603.24755): maintained human Python repos vs agent-written code. Only these two
# are comparable with scb-check 0.2.0; its Violation % and Clone Ratio are measured differently.
REFERENCE = {"verbosity": (0.15, 0.33), "erosion": (0.31, 0.68)}

SEVERITIES = ("critical", "high", "medium", "low")
# The one place severities are set: every issue the tools find names a row, and `report` prints this
# table into report.md, so the docs point here instead of restating it. Agents set the severity of the
# red flags and bugs they add.
POLICY = {
    "credential": ("critical", "credential matched by a confident rule, at HEAD or only in history"),
    "possible-credential": ("high", "possible credential (low-confidence rule or sensitive file name)"),
    "internal-head": ("high", "internal path, host, email or account at HEAD"),
    "large-file-head": ("high", f"file over {LARGE_FILE_MB} MB at HEAD"),
    "dead-large": ("high", f"unused function or class of {DEAD_HIGH_LINES}+ lines"),
    "license-missing": ("high", "no LICENSE file: nobody may use the code"),
    "license-mismatch": ("high", "pyproject.toml's license differs from the LICENSE file"),
    "license-third-party": ("high", "files under another license: keep their notices, check compatibility"),
    "readme-missing": ("high", "no README"),
    "deps-none": ("high", "no dependency list at all: nobody can install it"),
    "deps-undeclared": ("high", "an import that isn't a declared dependency, or only a dev one"),
    "vuln-fixable": ("high", "dependency with a known vulnerability and a fixed release"),
    "large-file-history": ("medium", f"file over {LARGE_FILE_MB} MB only in history: every clone downloads it"),
    "complex": ("medium", f"function with CC > {CC_THRESHOLD} and CC x sqrt(SLOC) >= {MEDIUM_MASS} "
                          "(e.g. CC 15 over 45 lines)"),
    "repetition": ("medium", f"duplicate group with lines x (copies - 1) >= {MEDIUM_DUP_LINES}"),
    "wrapper": ("medium", "trivial wrapper or shallow function"),
    "long-file": ("medium", f"source file over {LONG_FILE_LINES} lines: split it into modules"),
    "internal-history": ("low", "internal reference only in history: a fresh squashed history drops it"),
    "dead-small": ("low", f"unused function or class of {MIN_DEAD_LINES}-{DEAD_HIGH_LINES - 1} lines"),
    "complex-small": ("low", f"function with CC > {CC_THRESHOLD} and a smaller mass"),
    "repetition-small": ("low", f"smaller duplicate group of {MIN_CLONE_LINES}+ lines"),
    "inside-dead-code": ("low", "complexity or a red flag inside a dead-code candidate: delete that instead"),
    "deps-unused": ("low", "declared dependency that is never imported"),
    "deps-unreadable": ("low", "dependencies listed only where deptry can't read them (environment.yml, "
                               "setup.py, PEP 723)"),
    "vuln-unfixed": ("low", "known vulnerability without a fixed release"),
    "pins-none": ("low", "no lockfile or pinned requirements: can't audit, installs aren't reproducible"),
    "license-unrecognized": ("low", "LICENSE text not recognized"),
}
STATUSES = ("open", "fixed", "dismissed")
CATEGORIES = ("leak", "internal-ref", "large-file", "license", "vulnerable-dependency", "readiness", "dead-code",
              "complexity", "red-flag", "file-length", "bug")
ISSUE_DEFAULTS = {"source": "agent", "rule": "", "file": "", "line": None, "end_line": None, "symbol": "",
                  "evidence": {}, "fix": "", "done_when": "", "effort": "S", "status": "open", "note": ""}
AGENT_FIELDS = ("status", "severity", "fix", "done_when", "note", "effort")
HERE = Path(__file__).resolve().parent
RULES = HERE / "leak_rules.toml"

# --- small helpers ---------------------------------------------------------------------------------


class ToolError(Exception):
    """A tool could not measure: it failed, or printed something this script doesn't recognize."""


class CleanupError(Exception):
    """Stops the command; main() prints the message."""


class Result(NamedTuple):
    """What a measure step found. A step with nothing to read sets `skipped`: it measured nothing, so its
    earlier issues stay open instead of being closed as fixed."""

    issues: list
    metrics: dict
    skipped: str = ""
    notes: dict | None = None  # metric -> why it is incomplete


def die(message: str):
    raise CleanupError(message)


def run(cmd, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True, **kwargs)


def git(repo: Path, *args: str) -> str:
    p = run(["git", "-C", repo, *args])
    if p.returncode:
        die(f"git {' '.join(args)} failed: {p.stderr.strip()}")
    return p.stdout


def repo_root(path: str) -> Path:
    p = run(["git", "-C", path, "rev-parse", "--show-toplevel"])
    if p.returncode:
        die(f"{path} is not inside a git repository")
    return Path(p.stdout.strip()).resolve()


def rel(base: Path, path) -> str:
    """`path` relative to `base`, as posix. A tracked symlink can point outside the repo, so the path as
    written is tried before the resolved one; a path outside `base` comes back unchanged."""
    p = Path(path)
    if not p.is_absolute():
        return p.as_posix()
    for candidate in (p, p.resolve()):
        try:
            return candidate.relative_to(base).as_posix()
        except ValueError:
            pass
    return p.as_posix()


def head(repo: Path) -> str:
    return git(repo, "rev-parse", "HEAD").strip()


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def uvx() -> str:
    exe = shutil.which("uvx")
    if not exe:
        die("uvx not found: install uv (https://docs.astral.sh/uv/)")
    return exe


def version_key(v: str) -> tuple:
    return tuple(int(n) for n in re.findall(r"\d+", v))


# --- issues.json -----------------------------------------------------------------------------------


def state_dir(repo: Path) -> Path:
    d = repo / ".cleanup"
    (d / "snapshots").mkdir(parents=True, exist_ok=True)
    if not (d / ".gitignore").exists():
        (d / ".gitignore").write_text("*\n")
    return d


def load_issues(d: Path) -> dict:
    p = d / "issues.json"
    if not p.exists():
        return {"schema": 1, "repo": d.parent.name, "tools": {}, "issues": []}
    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        die(f"{p} is not valid JSON (line {e.lineno}: {e.msg}). Fix it by hand; .cleanup/ is not in git, so "
            "there is no older copy")
    check_issues(data, p)
    return data


def check_issues(data, path: Path):
    """issues.json is edited by hand, so a typo must stop the command rather than drop the issue from the
    report and the release gate. Missing optional fields get their defaults."""
    if not isinstance(data, dict) or not isinstance(data.setdefault("issues", []), list):
        die(f"{path} must be an object with an \"issues\" list")
    problems, seen = [], set()
    for n, i in enumerate(data["issues"], 1):
        ident = i.get("id") or f"issue #{n}"
        if missing := [k for k in ("id", "category", "severity", "summary") if not i.get(k)]:
            problems.append(f"{ident}: missing {', '.join(missing)}")
            continue
        for k, v in ISSUE_DEFAULTS.items():
            i.setdefault(k, dict(v) if isinstance(v, dict) else v)
        for k, allowed in (("severity", SEVERITIES), ("status", STATUSES), ("category", CATEGORIES)):
            if i[k] not in allowed:
                problems.append(f'{ident}: {k} "{i[k]}" is not one of {", ".join(allowed)}')
        if i["status"] == "dismissed" and not i["note"]:
            problems.append(f"{ident}: dismissed without a note saying why")
        if ident in seen:
            problems.append(f"{ident}: duplicate id")
        seen.add(ident)
    if problems:
        die(f"{path} needs fixing by hand (`grep <ID>` finds each issue):\n  " + "\n  ".join(problems[:20]))


def open_issues(data: dict, *severities: str) -> list:
    return [
        i for i in data["issues"]
        if i["status"] == "open" and (not severities or i["severity"] in severities)
    ]


def leak_scan_current(data: dict, repo: Path) -> bool:
    return data.get("leak_scan", {}).get("commit") == head(repo)


def release_status(data: dict, repo: Path) -> str:
    """Blocked while any critical or high issue is open, or while the leak scan is older than HEAD."""
    blocked = not leak_scan_current(data, repo) or open_issues(data, "critical", "high")
    return "blocked" if blocked else "clear"


def require_leak_scan(data: dict, repo: Path):
    if not leak_scan_current(data, repo):
        die("the leak scan is mandatory: run `cleanup.py leaks <repo>` on the current commit first")


def save_issues(d: Path, data: dict, repo: Path):
    """Valid JSON with one issue per line, so `grep <ID> issues.json` shows a whole issue."""
    rank = {s: n for n, s in enumerate(SEVERITIES)}
    data["issues"].sort(key=lambda i: (
        i["status"] != "open", rank.get(i["severity"], 9), i["category"],
        -(i.get("evidence") or {}).get("weight", (i.get("evidence") or {}).get("lines", 0)), i["id"]))
    data["commit"], data["generated"] = head(repo), now()
    data["release_status"] = release_status(data, repo)
    meta = [f"  {json.dumps(k)}: {json.dumps(v, ensure_ascii=False)}," for k, v in data.items()
            if k not in ("issues", "defaults")]
    issues = ",\n".join("    " + json.dumps(i, ensure_ascii=False) for i in data["issues"])
    defaults = json.dumps(data.get("defaults", {}), separators=(",", ":"))
    (d / "issues.json").write_text("{\n" + "\n".join(meta) + f'\n  "issues": [\n{issues}\n  ],\n'
                                   f'  "defaults": {defaults}\n}}\n')


def issue(prefix, source, category, rule, policy, summary, *, file="", symbol="", key="", line=None,
          end_line=None, evidence=None, fix="", done_when="", effort="S", status="open", note="") -> dict:
    """One finding, with the severity of its POLICY row. IDs hash stable facts (never line numbers), so
    re-runs keep them."""
    basis = key or "|".join((category, rule, file, symbol))
    return {
        "id": f"{prefix}-{hashlib.sha1(basis.encode()).hexdigest()[:8]}",
        "source": source, "category": category, "rule": rule, "severity": POLICY[policy][0],
        "file": file, "line": line, "end_line": end_line, "symbol": symbol, "summary": summary,
        "evidence": evidence or {}, "fix": fix, "done_when": done_when, "effort": effort,
        "status": status, "note": note,
    }


def _digest(value) -> str:
    return hashlib.sha1(json.dumps(value).encode()).hexdigest()[:10]


def merge(data: dict, found: list, sources: set):
    """Fold fresh findings into issues.json.

    A field an agent edited (status, severity, fix, ...) survives; one still holding the value the tool
    last wrote takes the new value. Open issues from `sources` that are no longer detected become fixed.
    A fixed issue that is detected again reopens, whoever marked it fixed: a fix the tool can still see
    is not one.
    """
    defaults = data.setdefault("defaults", {})  # per issue: hashes of the values the tool last wrote
    by_id = {i["id"]: i for i in data["issues"]}
    seen = set()
    for new in found:
        ident = new["id"]
        if ident in seen:
            continue
        seen.add(ident)
        old, prev = by_id.get(ident), defaults.get(ident)
        defaults[ident] = {k: _digest(new[k]) for k in AGENT_FIELDS}
        if old is None:
            data["issues"].append(new)
            by_id[ident] = new
            continue
        edited = {k: old[k] for k in AGENT_FIELDS if k in old and (prev is None or _digest(old[k]) != prev.get(k))}
        was_fixed = old["status"] == "fixed"
        old.update(new)
        old.update(edited)
        if was_fixed:
            earlier = edited.get("note") or ""
            old["status"] = new["status"]
            old["note"] = "detected again after being marked fixed" + (f" (earlier note: {earlier})" if earlier else "")
            defaults[ident].update(status=_digest(old["status"]), note=_digest(old["note"]))
    for i in data["issues"]:
        if i.get("source") in sources and i["id"] not in seen and i["status"] == "open":
            i["status"], i["note"] = "fixed", "no longer detected"
            defaults.setdefault(i["id"], {}).update(status=_digest("fixed"), note=_digest("no longer detected"))


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


def head_paths(repo: Path, needles: set, tmp: Path) -> dict:
    """{needle: paths of the files at HEAD that contain it}. A needle reaches git through a file in the
    private temp dir, never a command line; a multi-line secret is searched by its longest line."""
    pattern, out = Path(tmp, "needle"), {}
    for needle in needles:
        pattern.write_text(max(needle.splitlines(), key=len) + "\n")
        p = run(["git", "-C", repo, "grep", "-l", "-z", "-F", "-f", pattern, "HEAD", "--"])
        if p.returncode not in (0, 1):  # 1: no match
            die(f"git grep failed: {p.stderr.strip()[-200:]}")
        out[needle] = sorted(x.split(":", 1)[1] for x in p.stdout.split("\0") if x)
    pattern.unlink(missing_ok=True)
    return out


def leak_issues(repo: Path, findings: list, tmp: Path) -> list:
    """Group Betterleaks findings by distinct secret. Plaintext never leaves this function."""
    groups = {}
    for f in findings:
        secret = f.get("Secret") or ""
        internal = "internal" in (f.get("Tags") or [])
        # a credential is one issue wherever it appears (it is rotated once); internal references are
        # grouped per file, which is how they get fixed
        basis = f"internal|{f['RuleID']}|{f['File']}" if internal else secret or f"path:{f['File']}"
        g = groups.setdefault(basis, {
            "rules": {}, "internal": internal, "locs": [], "fingerprints": [], "needles": set(), "paths": set(),
            "first": None, "example": (f["File"], f["StartLine"]), "matches": set(),
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
        g["first"] = min(g["first"] or (f.get("Date", ""), f["Commit"], f["File"], f["StartLine"]),
                         (f.get("Date", ""), f["Commit"], f["File"], f["StartLine"]))

    # Where is it now? Betterleaks reports the commit and path that added a line, and a file may have
    # been moved since, so search HEAD's contents. A finding on a file name is at HEAD while the path is.
    hits = head_paths(repo, {n for g in groups.values() for n in g["needles"]}, tmp)
    tree = set(git(repo, "ls-tree", "-r", "-z", "--name-only", "HEAD").split("\0"))
    out = []
    for basis, g in groups.items():
        # IDs come from facts about the finding, never from the secret: a hash of a short password
        # next to its prefix and length could be brute-forced offline
        key = (hashlib.sha256(basis.encode()).hexdigest() if g["internal"] or not g["needles"] else
               "|".join(map(str, g["first"][1:])))
        rule = min(g["rules"], key=lambda r: (CONFIDENCE_RANK.get(g["rules"][r], 9), "generic" in r, r))
        now_in = sorted({p for n in g["needles"] for p in hits[n]} if g["needles"] else g["paths"] & tree)
        g["at_head"] = bool(now_in)
        file, line = g["example"]
        if now_in and file not in now_in:
            file, line = now_in[0], None
        where = "at HEAD" if g["at_head"] else "only in history"
        evidence = {
            "rules": sorted(g["rules"]), "preview": g["preview"], "at_head": g["at_head"], "head_files": now_in[:10],
            "first_commit": g["first"][1][:12], "first_date": g["first"][0],
            "occurrences": len(g["locs"]), "locations": g["locs"][:10], "fingerprints": g["fingerprints"][:10],
        }
        if g["internal"]:
            more = f" and {len(g['matches']) - 1} more" if len(g["matches"]) > 1 else ""
            evidence["matches"] = sorted(g["matches"])[:20]
            out.append(issue(
                "INT", "betterleaks", "internal-ref", rule, "internal-head" if g["at_head"] else "internal-history",
                f"Internal reference `{g['preview']}`{more} ({where})",
                key=f"INT|{key}", file=file, line=line, evidence=evidence,
                fix=("Replace it with a config value, CLI argument or environment variable "
                     "(or a neutral placeholder in docs)." if g["at_head"] else
                     "Only matters if the history is published; a fresh squashed history drops it."),
                done_when="`cleanup.py leaks` no longer reports it at HEAD."))
            continue
        certain = g["rules"][rule] in ("high", "medium")
        if certain:
            fix = ("User: revoke/rotate this credential at its provider first. "
                   + ("Then read it from an environment variable (add a `.env.example`, gitignore `.env`) "
                      "and keep it out of the published history: publish a fresh squashed history, or "
                      "rewrite history with the user's approval." if g["at_head"] else
                      "It is gone from HEAD but still in history: publish a fresh squashed history, or "
                      "rewrite history with the user's approval.")
                   + " The scan keeps finding it in the local history, so then dismiss this issue with a note "
                     "saying it was rotated and how history is handled.")
        else:
            fix = ("Check whether this is a real credential by looking at the location; never print the "
                   "value. If it is not, dismiss it with the reason. If it is, rotate it and remove it as for "
                   "any secret.")
        out.append(issue(
            "LEAK", "betterleaks", "leak", rule, "credential" if certain else "possible-credential",
            f"{'Secret' if certain else 'Possible secret'} `{g['preview']}` matched {rule} ({where})",
            key=f"LEAK|{key}", file=file, line=line, evidence=evidence, fix=fix,
            done_when="Rotated (user confirms), gone from HEAD, and dismissed with a note on how history is handled."))
    return out


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


def cmd_leaks(a):
    repo = repo_root(a.repo)
    d = state_dir(repo)
    data = load_issues(d)
    exe = betterleaks()
    version = run([exe, "version"]).stdout.strip() or "?"
    # --internal patterns are remembered, so a later scan without them doesn't mark their issues fixed
    internal = list(dict.fromkeys([*data.get("leak_scan", {}).get("internal", []), *a.internal]))
    rules = RULES.read_text() + "".join(_internal_rule(n, rx) for n, rx in enumerate(internal, 1))
    with tempfile.TemporaryDirectory(prefix="cleanup-leaks-") as tmp:
        config, report, mirror = Path(tmp, "rules.toml"), Path(tmp, "report.json"), Path(tmp, "scan.git")
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
        found = leak_issues(repo, json.loads(report.read_text() or "[]"), Path(tmp))
    found += large_file_issues(repo)
    merge(data, found, {"betterleaks", "git-history"})
    data["tools"]["betterleaks"] = version
    data["leak_scan"] = {
        "commit": head(repo), "date": now(), "betterleaks": version, "internal": internal,
        "rules_sha256": hashlib.sha256(rules.encode()).hexdigest()[:12],
    }
    save_issues(d, data, repo)

    cur = [i for i in found if i["category"] in ("leak", "internal-ref", "large-file")]
    counts = {s: sum(i["severity"] == s for i in cur) for s in SEVERITIES}
    print(f"leaks @ {head(repo)[:8]}  betterleaks {version}  "
          + " · ".join(f"{n} {s}" for s, n in counts.items() if n) if cur else
          f"leaks @ {head(repo)[:8]}  betterleaks {version}  nothing found")
    for i in [i for i in data["issues"] if i["id"] in {f["id"] for f in cur}][:15]:
        print(f"  {i['id']}  {i['severity']:<8} {i['status']:<9} {i['summary'][:90]}")
    if len(cur) > 15:
        print(f"  ... {len(cur) - 15} more in .cleanup/issues.json")
    untracked = git(repo, "ls-files", "--others", "--exclude-standard").split()
    if untracked:
        print(f"warning: {len(untracked)} untracked files were not scanned (e.g. {untracked[0]}); "
              "commit or ignore them and re-run")


# --- code index ------------------------------------------------------------------------------------


def matches(rel: str, pattern: str) -> bool:
    """scb-check's exclude semantics: '/'-separated fnmatch segments, '**' spans directories."""
    def seg(pp, pt):
        if not pt:
            return not pp
        if pt[0] == "**":
            return any(seg(pp[i:], pt[1:]) for i in range(len(pp) + 1))
        return bool(pp) and fnmatch(pp[0], pt[0]) and seg(pp[1:], pt[1:])
    return seg(tuple(p for p in rel.split("/") if p), tuple(p for p in pattern.split("/") if p))


def shipped_files(repo: Path, excludes: list) -> list:
    rels = git(repo, "ls-files", "-z", "--", "*.py").split("\0")
    return [r for r in rels if r and (repo / r).is_file() and not any(matches(r, g) for g in excludes)]


class CodeIndex:
    """Functions and classes per file (qualname, span), the lines that hold code, and each file's length."""

    def __init__(self, repo: Path, files: list):
        self.defs, self.code, self.lines, self.syntax_errors = {}, {}, {}, []
        for rel in files:
            text = (repo / rel).read_text(errors="replace")
            self.lines[rel] = len(text.splitlines())
            self.code[rel] = {n for n, s in enumerate(text.splitlines(), 1)
                              if s.strip() and not s.strip().startswith("#")}
            try:
                tree = ast.parse(text)
            except SyntaxError:
                self.syntax_errors.append(rel)
                continue
            defs = self.defs[rel] = []

            def walk(node, prefix):
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        start = min([child.lineno] + [x.lineno for x in child.decorator_list])
                        defs.append((start, child.end_lineno, prefix + child.name, child.lineno,
                                     "class" if isinstance(child, ast.ClassDef) else "function"))
                        walk(child, prefix + child.name + ".")
                    else:
                        walk(child, prefix)

            walk(tree, "")

    def enclosing(self, rel: str, line: int):
        """Innermost def/class containing `line`: (start, end, qualname, def_line, kind) or None."""
        best = None
        for d in self.defs.get(rel, ()):
            if d[0] <= line <= d[1] and (best is None or d[0] >= best[0]):
                best = d
        return best

    def at_def_line(self, rel: str, line: int):
        return next((d for d in self.defs.get(rel, ()) if d[3] == line), None) or self.enclosing(rel, line)

    def sloc(self, rel: str, start: int, end: int) -> int:
        return sum(start <= n <= end for n in self.code.get(rel, ()))

    def qualname(self, rel: str, line: int) -> str:
        d = self.enclosing(rel, line)
        return d[2] if d else "<module>"


# --- measure: scb-check ----------------------------------------------------------------------------

SCB_HEADER = re.compile(r"^(?:warning\[(?P<a>[\w-]+)\]|(?P<b>[\w-]+)\[warning\]|(?P<c>[\w-]+)):\s+(?P<msg>.+)$")
SCB_LOCATION = re.compile(r"┌─\s+(?P<path>.+?):(?P<line>\d+)(?::\d+)?\s*$")
SCB_CLONE = re.compile(r"duplicated block \((\d+) lines?, (\d+) instances?\)")
SCB_NOTE = re.compile(r"^\s*= (?P<note>.+)$")
SCB_CC = re.compile(r"complexity: (\d+), sloc: (\d+)")
SCB_FUNCTION = re.compile(r"function `([^`]+)`")


def parse_scb(text: str) -> list:
    findings = []
    for row in text.splitlines():
        if m := SCB_HEADER.match(row):
            findings.append({"rule": m["a"] or m["b"] or m["c"], "msg": m["msg"], "locs": [], "notes": []})
        elif (m := SCB_LOCATION.search(row)) and findings:
            findings[-1]["locs"].append((m["path"], int(m["line"])))
        elif (m := SCB_NOTE.match(row)) and findings:
            findings[-1]["notes"].append(m["note"])
    return findings


def scb_check(repo, files, index) -> Result:
    with tempfile.TemporaryDirectory(prefix="cleanup-scb-") as tmp:
        # scb-check walks a directory, so it would also read untracked files and other languages: give it
        # a copy of exactly the files in scope
        src = Path(tmp, "src").resolve()
        src.mkdir()
        for f in files:
            (src / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(repo / f, src / f)
        config = Path(tmp, "scb-check.toml")
        config.write_text("")
        shim = Path(tmp, "bin", "sg")  # a failing `sg` first on PATH makes scb-check use its bundled ast-grep
        shim.parent.mkdir()
        shim.write_text("#!/bin/sh\nexit 1\n")
        shim.chmod(0o755)
        env = {**os.environ, "PATH": f"{shim.parent}{os.pathsep}{os.environ.get('PATH', '')}"}
        base = [uvx(), f"scb-check@{TOOLS['scb-check']}", "check", src, "--config", config]
        p = run([*base, "--output-format", "json"], env=env)
        h = run([*base, "--min-duplicate-lines", MIN_CLONE_LINES], env=env)
        for q in (p, h):
            if q.returncode not in (0, 1):  # 1 only means findings
                raise ToolError(q.stderr.strip()[-300:])
        m = json.loads(p.stdout)
        out = scb_issues(h.stdout, src, index, m["high_cc_functions"])
    loc = m["total_loc"] or 1
    metrics = dict(
        loc=m["total_loc"], cc_over_10=m["high_cc_functions"], verbosity=round(m["verbosity"], 4),
        erosion=round(m["erosion"], 4), cog_erosion=round(m["cog_erosion"], 4),
        violation_pct=round(m["ast_grep_flagged_loc"] / loc, 4), clone_ratio=round(m["clone_loc"] / loc, 4),
        # absolute amounts behind the ratios: deleting clean code raises a ratio without adding slop
        verbosity_flagged_loc=m["verbosity_flagged_loc"], high_cc_mass=round(m["high_cc_mass"], 1))
    # no ast-grep binary worked: the slop rules silently counted nothing
    return Result(out, metrics, notes={"violation_pct": "ast-grep did not run"} if "ast-grep" in p.stderr else None)


def scb_issues(text: str, src: Path, index, high_cc: int) -> list:
    """Duplicate blocks, trivial wrappers and complex functions from scb-check's human output. Raises
    ToolError when it doesn't parse as expected, so a changed format can't read as "all fixed"."""
    clones, complex_fns, cognitive, out = [], [], {}, []
    for f in parse_scb(text):
        locs = [(rel(src, path), line) for path, line in f["locs"]]
        if not locs and f["rule"] in ("erosion", "duplicate-structure", "trivial-wrapper", "low-use-short-function"):
            raise ToolError(f"no location parsed for {f['rule']}: {f['msg'][:80]}")
        size = next((m for n in f["notes"] if (m := SCB_CC.search(n))), None)
        if f["rule"] in ("erosion", "cog_erosion") and size:
            if f["rule"] == "erosion":
                name = SCB_FUNCTION.search(f["msg"])
                complex_fns.append((*locs[0], int(size[1]), int(size[2]), name[1] if name else "<function>"))
            else:
                cognitive[locs[0]] = int(size[1])
        elif f["rule"] == "duplicate-structure" and (c := SCB_CLONE.search(f["msg"])):
            clones.append((int(c[1]) * (int(c[2]) - 1), int(c[1]), int(c[2]), locs))
        elif f["rule"] in ("trivial-wrapper", "low-use-short-function"):
            file, line = locs[0]
            name = (index.at_def_line(file, line) or (0, 0, "<module>"))[2]
            flag ="pass-through-method" if f["rule"] == "trivial-wrapper" else "shallow-module"
            out.append(issue(
                "RF", "scb-check", "red-flag", flag, "wrapper", f"{f['rule']}: {f['msg']}",
                file=file, symbol=name, line=line, evidence={"scb_rule": f["rule"]}, effort="S",
                fix=("Inline it into its callers or remove it. If it is deliberate (a public alias, or an "
                     "override such as `model.train = disabled_train`), dismiss the issue with that reason."),
                done_when="scb-check no longer reports it (or it is dismissed), and the safety net passes."))
    if len(complex_fns) != high_cc:
        raise ToolError(f"output not recognized: its JSON counts {high_cc} functions with CC > {CC_THRESHOLD}, "
                        f"{len(complex_fns)} were parsed")
    for weight, lines, count, locs in clones:
        members = sorted({(file, index.qualname(file, line)) for file, line in locs})
        where = ", ".join(f"{file}:{line}" for file, line in locs[:4])
        out.append(issue(
            "RF", "scb-check", "red-flag", "repetition",
            "repetition" if weight >= MEDIUM_DUP_LINES else "repetition-small",
            f"{lines}-line block duplicated {count}x: {where}",
            key="RF|repetition|" + "|".join(f"{f}:{q}" for f, q in members),
            file=locs[0][0], line=locs[0][1], symbol=", ".join(q for _, q in members),
            evidence={"lines": lines, "instances": count, "weight": weight,
                      "locations": [f"{f}:{n}" for f, n in locs]},
            effort="M" if lines >= 30 else "S",
            fix=("Extract the shared block into one function, or delete the copy that is no longer used. "
                 "Pass the real differences as a few parameters."),
            done_when="scb-check no longer reports this duplicate group and the safety net passes."))
    for file, line, cc, sloc, name in complex_fns:
        d = index.at_def_line(file, line)
        qual, start, end = (d[2], d[0], d[1]) if d else (name, line, line)
        mass = cc * math.sqrt(sloc)
        evidence = {"cc": cc, "sloc": sloc, "mass": round(mass, 1), "weight": round(mass, 1)}
        if (file, line) in cognitive:
            evidence["cognitive"] = cognitive[file, line]
        out.append(issue(
            "CX", "scb-check", "complexity", "cc", "complex" if mass >= MEDIUM_MASS else "complex-small",
            f"`{qual}` has cyclomatic complexity {cc} ({sloc} SLOC)",
            file=file, symbol=qual, line=start, end_line=end, evidence=evidence,
            effort="L" if sloc > 150 else "M",
            fix=("Simplify the control flow: drop dead branches and flags, return early, move special cases "
                 "out of the general path. Extracting a step with its own clear purpose is fine; moving "
                 "branches into a helper that only this function calls, with most of its arguments, is not. "
                 "If every branch is needed and it reads clearly, dismiss it with that reason. Add a "
                 "characterization test first."),
            done_when=(f"scb-check no longer lists it (CC <= {CC_THRESHOLD}), or it is gone or dismissed with "
                       "a reason, and the safety net passes.")))
    return out


# --- measure: vulture ----------------------------------------------------------------------------

VULTURE = re.compile(
    r"^(?P<path>.+?):(?P<line>\d+): (?P<msg>unused (?P<kind>function|class|method|property|variable|"
    r"attribute|import) '(?P<name>[^']+)'|unreachable code after '[^']+'|.+?) "
    r"\((?P<conf>\d+)% confidence(?:, (?P<size>\d+) lines?)?\)$")
TEXT_SUFFIXES = {".yaml", ".yml", ".json", ".toml", ".cfg", ".ini", ".sh", ".bash", ".slurm", ".sbatch",
                 ".ipynb", ".jinja", ".j2"}


def _references(repo: Path, shipped: list) -> list:
    """(path, text, strings_only) of tracked files that can name code without importing it.

    Configs, scripts and notebooks count wholesale, as do Python files outside the scope (tests). In the
    shipped Python files only string literals count, e.g. {"target": "main.SetupCallback"}.
    """
    out, shipped = [], set(shipped)
    for rel in git(repo, "ls-files", "-z").split("\0"):
        path, suffix = repo / rel, Path(rel).suffix
        if not rel or not path.is_file() or path.stat().st_size > 5_000_000:
            continue
        if suffix in TEXT_SUFFIXES or (suffix == ".py" and rel not in shipped):
            out.append((rel, path.read_text(errors="replace"), False))
        elif rel in shipped:
            out.append((rel, path.read_text(errors="replace"), True))
    return out


def _named_in(refs: list, name: str, module: str) -> tuple:
    """(files that name it, files that only mention it). Naming takes its module: `module.name` or
    `module:name` (a yaml `target:`, an entry point, a "main.SetupCallback" string in the code) or
    `from ...module import name`. A bare `.name` or "name" (`model.train()`, `split="train"`) is only a
    mention, left for review. Dunders are never references."""
    if name.startswith("__") and name.endswith("__"):
        return [], []
    n, m = re.escape(name), re.escape(module)
    qualified = re.compile(rf"\b{m}[.:]{n}\b|\bfrom\s+[\w.]*\b{m}\s+import\s*(?:\([^)]*|[^\n]*)\b{n}\b")
    in_string = re.compile(rf"[\"'][^\"'\n]*\b{m}[.:]{n}[\"']")
    bare, bare_string = re.compile(rf"\b{n}\b"), re.compile(rf"[\"']{n}[\"']")
    named, mentioned = [], []
    for rel, text, strings_only in refs:
        if (in_string if strings_only else qualified).search(text):
            named.append(rel)
        elif (bare_string if strings_only else bare).search(text):
            mentioned.append(rel)
    return named, mentioned


def runtime_lines(index, file: str, line: int) -> set:
    """Lines that only run when the code is used: a function's body, or the bodies of a class's methods.
    Def, decorator and class-body lines run on import, so coverage of them says nothing."""
    d = index.at_def_line(file, line)
    if not d:
        return set()
    start, end, _, _, kind = d
    bodies = [d] if kind == "function" else [
        x for x in index.defs.get(file, ()) if x[4] == "function" and start <= x[0] and x[1] <= end]
    return {n for _, e, _, def_line, _ in bodies for n in range(def_line + 1, e + 1)}


def vulture_rows(text: str) -> list:
    """vulture's report lines as regex matches. An unrecognized line raises ToolError instead of being
    skipped, so a changed format can't read as "no dead code"."""
    rows = []
    for row in text.splitlines():
        if row.strip():
            if not (m := VULTURE.match(row)):
                raise ToolError(f"output not recognized: {row[:120]}")
            rows.append(m)
    return rows


def vulture_dead(repo, files, index) -> Result:
    if not files:
        return Result([], {"dead_code_loc": 0})
    p = run([uvx(), f"vulture@{TOOLS['vulture']}", *files, "--min-confidence", "60", "--sort-by-size"], cwd=repo)
    if p.returncode not in (0, 3):
        raise ToolError((p.stderr or p.stdout).strip()[-300:])
    refs = _references(repo, files)
    coverage = {}
    if (cov := repo / "coverage.json").exists():  # optional: a coverage.py JSON report from any run
        for name, f in json.loads(cov.read_text()).get("files", {}).items():
            coverage[Path(name).as_posix()] = set(f.get("executed_lines", []))
    total, out = 0, []
    for m in vulture_rows(p.stdout):
        # methods and properties are skipped: in ML code they are mostly framework hooks (Lightning's
        # training_step, nn.Module overrides) that vulture can't see being called
        if m["kind"] in ("method", "property", "variable", "attribute", "import"):
            continue
        file, line, size = Path(m["path"]).as_posix(), int(m["line"]), int(m["size"] or 1)
        d = index.at_def_line(file, line)
        qual = d[2] if d and m["name"] and d[2].endswith(m["name"]) else (m["name"] or index.qualname(file, line))
        name = qual.rsplit(".", 1)[-1]
        module = Path(file).parent.name if Path(file).stem == "__init__" else Path(file).stem
        # unreachable code can't be referenced
        hits, mentions = _named_in(refs, name, module) if m["kind"] else ([], [])
        ran = bool(coverage.get(file, set()) & runtime_lines(index, file, line))
        if not hits and not ran:
            total += size
        if size < MIN_DEAD_LINES:
            continue
        note = (f"referenced by name in {', '.join(hits[:3])}" if hits else
                "executed in the coverage run" if ran else "")
        out.append(issue(
            "DEAD", "vulture", "dead-code", m["kind"] or "unreachable",
            "dead-large" if size >= DEAD_HIGH_LINES else "dead-small",
            f"{m['msg']} ({size} lines, vulture {m['conf']}% confidence)",
            file=file, symbol=qual, line=line, end_line=line + size - 1,
            evidence={"lines": size, "confidence": int(m["conf"]), "referenced_in": hits[:5],
                      "mentioned_in": mentions[:5], "executed": ran if coverage else None},
            status="dismissed" if note else "open", note=note, effort="S",
            fix=("Confirm nothing references it (yaml `_target_`/`target:` strings, registries, getattr, "
                 "notebooks, shell scripts; files that mention its bare name are in `evidence.mentioned_in`), "
                 "then delete it."),
            done_when="Deleted, and the import test and safety net pass."))
    return Result(out, {"dead_code_loc": total})


# --- measure: dependencies -------------------------------------------------------------------------

DEPTRY_MAP = ("scikit-learn=sklearn,pillow=PIL,opencv-python=cv2,opencv-python-headless=cv2,pyyaml=yaml,"
              "hydra-core=hydra,pytorch-lightning=pytorch_lightning,scikit-image=skimage,beautifulsoup4=bs4,"
              "protobuf=google.protobuf,huggingface-hub=huggingface_hub,faiss-cpu=faiss,faiss-gpu=faiss,"
              "python-dateutil=dateutil,attrs=attr,tensorboardx=tensorboardX,ftfy=ftfy")


def _pyproject(repo: Path) -> dict:
    try:
        return tomllib.loads((repo / "pyproject.toml").read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def deptry_deps(repo, files, index) -> Result:
    pyproject = _pyproject(repo)
    if pyproject.get("project", {}).get("dependencies") is not None or \
            pyproject.get("tool", {}).get("poetry", {}).get("dependencies"):
        source = ["--config", repo / "pyproject.toml"]
    elif (repo / "requirements.txt").exists():
        source = ["--requirements-files", repo / "requirements.txt"]
    else:
        elsewhere = [n for n in ("environment.yml", "environment.yaml", "setup.py", "setup.cfg", "Pipfile")
                     if (repo / n).exists()] + sorted(p.name for p in repo.glob("requirements?*.txt"))
        if any("# /// script" in (repo / f).read_text(errors="replace") for f in files):
            elsewhere.append("PEP 723 script blocks")
        return Result([issue(
            "DEPS", "deptry", "readiness", "no-dependency-list", "deps-unreadable" if elsewhere else "deps-none",
            f"Dependencies are declared only in {', '.join(elsewhere)}, which deptry can't check" if elsewhere
            else "No dependency list (pyproject.toml, requirements.txt, environment.yml): nobody can install it",
            key="DEPS|declare", effort="S",
            fix=("Declare the runtime dependencies in pyproject.toml ([project].dependencies)."
                 + (" Self-contained PEP 723 scripts can dismiss this." if elsewhere else "")),
            done_when="`cleanup.py measure` runs deptry without 'not measured'.")], {},
            skipped="no pyproject.toml dependencies or requirements.txt")
    with tempfile.TemporaryDirectory(prefix="cleanup-deptry-") as tmp:
        report = Path(tmp) / "deptry.json"
        p = run([uvx(), f"deptry@{TOOLS['deptry']}", repo, *source, "--package-module-name-map", DEPTRY_MAP,
                 "--json-output", report])
        if not report.exists():
            raise ToolError((p.stderr or p.stdout).strip()[-300:])
        violations = json.loads(report.read_text())
    out = []
    for v in violations:
        code, module = v["error"]["code"], v["module"]
        file = rel(repo, v["location"]["file"])
        fixes = {"DEP001": f"Declare the package that provides `{module}` as a dependency.",
                 "DEP002": f"Remove `{module}` from the dependencies if it is really unused.",
                 "DEP003": f"`{module}` is only installed transitively: declare it directly.",
                 "DEP004": f"`{module}` is a dev dependency but the code imports it: move it to the main dependencies."}
        out.append(issue(
            "DEP", "deptry", "readiness", code, "deps-unused" if code == "DEP002" else "deps-undeclared",
            v["error"]["message"], file=file, symbol=module, line=v["location"]["line"],
            fix=fixes.get(code, v["error"]["message"]), effort="S",
            done_when="deptry no longer reports it."))
    return Result(out, {"deps_missing": sum(v["error"]["code"] != "DEP002" for v in violations)})


PINNED = re.compile(r"^[A-Za-z0-9][\w.\-\[\], ]*==\s*[\w.+!-]+\s*(?:;.*)?(?:\\)?$")


def _fully_pinned(path: Path) -> bool:
    lines = [ln.split(" #")[0].strip() for ln in path.read_text().splitlines()]
    reqs = [ln for ln in lines if ln and not ln.startswith(("#", "--hash"))]
    return bool(reqs) and all(PINNED.match(ln) for ln in reqs)


def pip_audit_vulns(repo, files, index) -> Result:
    with tempfile.TemporaryDirectory(prefix="cleanup-audit-") as tmp:
        source, label = None, ""
        if (repo / "uv.lock").exists() and (uv := shutil.which("uv")):
            req = Path(tmp) / "requirements.txt"
            p = run([uv, "export", "--project", repo, "--frozen", "--format", "requirements.txt",
                     "--no-emit-project", "--quiet", "-o", req])
            if p.returncode == 0:
                source, label = ["-r", req, "--require-hashes", "--disable-pip"], "uv.lock"
        if source is None and list(repo.glob("pylock*.toml")):
            source, label = ["--locked", repo], "pylock.toml"
        if source is None:
            for req in [repo / "requirements.txt", *sorted(repo.glob("requirements*.txt"))]:
                if req.exists() and _fully_pinned(req):
                    source, label = ["-r", req, "--no-deps", "--disable-pip"], req.name
                    break
        if source is None and (sites := sorted(repo.glob(".venv/lib/python*/site-packages"))):
            source, label = ["--path", sites[-1]], ".venv"
        if source is None:
            return Result([issue(
                "DEPS", "pip-audit", "readiness", "no-pinned-dependencies", "pins-none",
                "No lockfile or pinned requirements: dependencies can't be audited and installs aren't "
                "reproducible", key="DEPS|pin", effort="S",
                fix="Commit a lockfile (uv.lock) or a fully pinned requirements.txt.",
                done_when="`cleanup.py measure` runs pip-audit without 'not measured'.")], {},
                skipped="no pinned source (uv.lock, pylock, pinned requirements, .venv)")
        report = Path(tmp) / "audit.json"
        p = run([uvx(), f"pip-audit@{TOOLS['pip-audit']}", *source, "--format", "json", "-o", report,
                 "--progress-spinner", "off"])
        if p.returncode not in (0, 1) or not report.exists():
            raise ToolError(p.stderr.strip()[-300:])
        deps = json.loads(report.read_text()).get("dependencies", [])
    out = []
    for dep in deps:
        vulns = {v["id"]: v for v in dep.get("vulns", [])}
        if not vulns:
            continue
        fixed = [min(v["fix_versions"], key=version_key) for v in vulns.values() if v["fix_versions"]]
        target = max(fixed, key=version_key) if fixed else None
        ids = sorted(vulns)
        out.append(issue(
            "VULN", "pip-audit", "vulnerable-dependency", "pip-audit", "vuln-fixable" if target else "vuln-unfixed",
            f"{dep['name']} {dep['version']}: {len(ids)} known vulnerabilities ({', '.join(ids[:3])}"
            f"{', ...' if len(ids) > 3 else ''})" + (f"; fixed in {target}" if target else "; no fix yet"),
            key=f"VULN|{dep['name'].lower()}", file=label, symbol=dep["name"],
            evidence={"version": dep["version"], "vulns": ids, "fix_to": target,
                      "aliases": sorted({a for v in vulns.values() for a in v.get("aliases", [])})[:10]},
            fix=(f"Bump {dep['name']} to >= {target} in the dependency spec, then run the safety net."
                 if target else "No fixed release yet: note it, or drop the dependency if it is easy."),
            done_when=f"pip-audit reports no vulnerabilities for {dep['name']}."))
    return Result(out, {"pip_audit_source": label, "vulnerable_deps": len(out)})


# --- measure: license and readme -------------------------------------------------------------------

HEADER_LICENSES = (
    ("Apache-2.0", r"apache license|apache-2\.0"), ("AGPL", r"affero general public|agpl"),
    ("LGPL", r"lesser general public|lgpl"), ("GPL", r"gnu general public|\bgpl"),
    ("MPL-2.0", r"mozilla public|mpl-2\.0"), ("CC-BY-NC", r"noncommercial|cc-by-nc"),
    ("BSD", r"redistribution and use in source and binary|\bbsd"), ("MIT", r"permission is hereby granted|\bmit\b"),
)


def detect_license(text: str) -> str | None:
    t = " ".join(text.split()).lower()
    if "apache license" in t and "version 2.0" in t:
        return "Apache-2.0"
    if "mozilla public license" in t:
        return "MPL-2.0"
    for family, words in (("AGPL", "gnu affero general public license"), ("LGPL", "gnu lesser general public"),
                          ("GPL", "gnu general public license")):
        if words in t:
            return family + ("-3.0" if "version 3" in t else "-2.0" if "version 2" in t else "")
    if "permission is hereby granted, free of charge" in t:
        return "MIT"
    if "redistribution and use in source and binary forms" in t:
        return "BSD-3-Clause" if "neither the name" in t else "BSD-2-Clause"
    if "permission to use, copy, modify, and/or distribute this software" in t:
        return "ISC"
    if "free and unencumbered software released into the public domain" in t:
        return "Unlicense"
    if "attribution-noncommercial" in t:
        return "CC-BY-NC"
    return None


def family(spdx: str | None) -> str:
    """Compare licenses by family: GPL-3.0-only and GPL-3.0 match, MIT and Apache-2.0 don't."""
    if not spdx:
        return ""
    s = spdx.lower().replace("-only", "").replace("-or-later", "").replace("+", "")
    for fam, pattern in HEADER_LICENSES:
        if re.search(pattern, s):
            return fam
    return s


def license_readme(repo, files, index) -> Result:
    out = []
    names = git(repo, "ls-tree", "--name-only", "HEAD").splitlines()
    lic_files = [n for n in names if re.fullmatch(r"(?i)(?:licen[cs]e|copying)(?:\.(?:md|txt|rst))?", n)]
    repo_license = None
    if not lic_files:
        out.append(issue(
            "LIC", "license", "license", "missing-license", "license-missing",
            "No LICENSE file: without one, nobody may legally use, copy or modify the code",
            key="LIC|missing", effort="S",
            fix=("Ask the user which license to use (MIT and Apache-2.0 are common for research code; "
                 "Apache-2.0 adds a patent grant). Add LICENSE at the repo root and set `license` in "
                 "pyproject.toml."),
            done_when="A LICENSE file is at the root and `cleanup.py measure` recognizes it."))
    else:
        repo_license = detect_license((repo / lic_files[0]).read_text(errors="replace"))
        if repo_license is None:
            out.append(issue(
                "LIC", "license", "license", "unrecognized-license", "license-unrecognized",
                f"{lic_files[0]} doesn't match a common license (MIT, Apache-2.0, BSD, GPL, MPL)",
                key="LIC|unrecognized", file=lic_files[0], effort="S",
                fix="Check the text; a standard license text lets people and tools recognize it.",
                done_when="`cleanup.py measure` recognizes the license."))
        declared = _pyproject(repo).get("project", {}).get("license")
        declared = declared.get("text") if isinstance(declared, dict) else declared
        if repo_license and isinstance(declared, str) and family(declared) != family(repo_license):
            out.append(issue(
                "LIC", "license", "license", "license-mismatch", "license-mismatch",
                f"pyproject.toml declares `{declared}` but {lic_files[0]} is {repo_license}",
                key="LIC|mismatch", file="pyproject.toml", effort="S",
                fix="Make pyproject's `license` match the LICENSE file.",
                done_when="Both name the same license."))
    metrics = {"license": repo_license or ("missing" if not lic_files else "unrecognized")}

    vendored = {}
    sources = [r for r in git(repo, "ls-files", "-z").split("\0")
               if Path(r).suffix in {".py", ".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".sh", ".js", ".ts"}]
    for rel in sources:
        try:
            top = "\n".join((repo / rel).read_text(errors="replace").splitlines()[:40]).lower()
        except OSError:
            continue
        spdx = re.search(r"spdx-license-identifier:\s*([\w.+-]+)", top)
        found = family(spdx[1]) if spdx else next(
            (fam for fam, pattern in HEADER_LICENSES if fam != "MIT" and re.search(pattern, top)
             and ("license" in top or "licence" in top)), None)
        if found and found != family(repo_license):
            vendored.setdefault(found, []).append(rel)
    for fam, paths in sorted(vendored.items()):
        out.append(issue(
            "LIC", "license", "license", "third-party-license", "license-third-party",
            f"{len(paths)} file(s) carry a {fam} license header but the repo is "
            f"{repo_license or 'unlicensed'}: {', '.join(paths[:4])}{', ...' if len(paths) > 4 else ''}",
            key=f"LIC|vendored|{fam}", file=paths[0], evidence={"files": paths[:50]}, effort="S",
            fix=(f"Keep these notices. Check that {fam} code may ship under the repo license (GPL/AGPL code "
                 "can't be relicensed as MIT or Apache-2.0; non-commercial code isn't open source) and "
                 "credit the original project in the README or a NOTICE file."),
            done_when="The user has confirmed the licenses are compatible (or the files are replaced)."))

    if not any(re.fullmatch(r"(?i)readme(?:\.\w+)?", n) for n in names):
        out.append(issue(
            "DOC", "license", "readiness", "missing-readme", "readme-missing", "No README at the repo root",
            key="DOC|readme", effort="M",
            fix="Add a README: what the code does, how to install it, a minimal quickstart, how to cite it.",
            done_when="README exists and its quickstart runs."))
    return Result(out, metrics)


def count_suppressions(repo: Path, files: list) -> int:
    pattern = re.compile(r"#\s*(?:noqa|scbc\s+ignore)", re.I)
    return sum(len(pattern.findall((repo / f).read_text(errors="replace"))) for f in files)


def file_lengths(repo, files, index) -> Result:
    """Average length of the source files, how many pass 500 and 1000 lines, and an issue for the latter."""
    lines = index.lines
    out = [issue(
        "SIZE", "file-length", "file-length", "long-file", "long-file", f"`{f}` has {n:,} lines",
        file=f, evidence={"lines": n, "weight": n}, effort="M",
        fix=("Split it into modules of a few hundred lines, each owning one concept (what it hides, not when it "
             "runs). Move the code unchanged in its own commit, then refactor; if other code imports from this "
             "module, keep that import path working."),
        done_when=f"It is under {LONG_FILE_LINES} lines, and the import test and safety net pass.")
        for f, n in sorted(lines.items()) if n > LONG_FILE_LINES]
    return Result(out, {"avg_file_lines": round(sum(lines.values()) / len(lines)) if lines else 0,
                        "files_over_500": sum(n > WARN_FILE_LINES for n in lines.values()),
                        "files_over_1000": len(out)})


# --- measure ---------------------------------------------------------------------------------------

STEPS = (("scb-check", scb_check), ("vulture", vulture_dead),
         ("deptry", deptry_deps), ("pip-audit", pip_audit_vulns), ("license", license_readme),
         ("file-length", file_lengths))


def run_steps(repo, files, index, steps=None) -> tuple:
    """Run the measure steps: (issues, metrics, not_measured, measured). A step that fails or has nothing
    to read is not measured, and one broken step doesn't lose the others' results."""
    found, metrics, not_measured, measured = [], {}, {}, set()
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


def load_snapshot(d: Path, label: str) -> dict:
    p = d / "snapshots" / f"{label}.json"
    if not p.exists():
        have = ", ".join(sorted(x.stem for x in (d / "snapshots").glob("*.json"))) or "none"
        die(f"no snapshot '{label}' (have: {have})")
    return json.loads(p.read_text())


def write_snapshot(d: Path, snap: dict) -> Path:
    path = d / "snapshots" / f"{snap['label']}.json"
    path.write_text(json.dumps(snap, indent=2) + "\n")
    return path


def count_issues(snap: dict, data: dict, repo: Path):
    """The snapshot fields that come from issues.json rather than from the tools."""
    still_open = open_issues(data)
    snap["metrics"]["red_flags_open"] = sum(i["category"] == "red-flag" for i in still_open)
    snap["metrics"]["leaks_open"] = sum(i["category"] == "leak" for i in still_open)
    snap["open_issues"] = {s: sum(i["severity"] == s for i in still_open) for s in SEVERITIES}
    snap["release_status"] = release_status(data, repo)


def recount_head_snapshot(d: Path, data: dict, repo: Path):
    """Review adds and closes issues after `measure`, so recount the snapshot taken at HEAD: the baseline
    then includes the red flags found in review. Snapshots of older commits keep their counts."""
    snap = latest_snapshot(d)
    if snap and snap["commit"] == head(repo):
        count_issues(snap, data, repo)
        write_snapshot(d, snap)


def latest_snapshot(d: Path, exclude: str = "") -> dict | None:
    snaps = [json.loads(p.read_text()) for p in (d / "snapshots").glob("*.json")]
    snaps = [s for s in snaps if s["label"] != exclude]
    return max(snaps, key=lambda s: s["date"]) if snaps else None


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
              "Set by `POLICY` in cleanup.py; red flags and bugs you add keep the severity you give them.", ""]
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
    for src in (Path(__file__).resolve(), RULES):  # lets another agent re-measure without the skill
        if src.parent != d.resolve():
            shutil.copy2(src, d / src.name)
    o = {s: len(open_issues(data, s)) for s in SEVERITIES}
    print(f"release: {data['release_status'].upper()}  ·  open " + " · ".join(f"{o[s]} {s}" for s in SEVERITIES))
    print(f"-> {d / 'report.md'}")


# --- cli -------------------------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("leaks", help="scan the full git history for secrets, internal refs and large files")
    p.add_argument("repo")
    p.add_argument("--internal", action="append", default=[], metavar="REGEX",
                   help="extra internal-reference pattern (RE2); repeatable")
    p = sub.add_parser("measure", help="metrics snapshot and mechanical issues")
    p.add_argument("repo")
    p.add_argument("--label", default="baseline")
    p.add_argument("--exclude", action="append", default=[], metavar="GLOB",
                   help="extra path glob to leave out of the metrics; repeatable")
    p = sub.add_parser("compare", help="before/after table")
    p.add_argument("repo")
    p.add_argument("base")
    p.add_argument("--to", help="snapshot to compare with (default: the latest)")
    p = sub.add_parser("report", help="render .cleanup/report.md")
    p.add_argument("repo")
    a = ap.parse_args()
    try:
        {"leaks": cmd_leaks, "measure": cmd_measure, "compare": cmd_compare, "report": cmd_report}[a.cmd](a)
    except CleanupError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
