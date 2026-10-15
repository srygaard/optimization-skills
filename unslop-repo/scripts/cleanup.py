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
stdout or any file: only a 4-character preview, the length and a hash. Tests: test_cleanup.py.

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

# --- pinned tools ----------------------------------------------------------------------------------

BETTERLEAKS_VERSION = "1.9.0"
BETTERLEAKS_SHA256 = {  # checksums.txt of the v1.9.0 release, signature verified with sigstore
    ("darwin", "arm64"): "83dd7eaab13d44a1e8e347a17064cdef153a1478db473276d41b38d55613d3f2",
    ("darwin", "x64"): "68dfd83458d9e7f90663daa7a81c7f944083b170dae82e4f5f8d0581ed519ef1",
    ("linux", "arm64"): "1d39116e0a58dc94574715e2aa12a2dbd5062f193eee3fec011fef6ba06bd13b",
    ("linux", "x64"): "f8b185a39ffcece2a1ca82bf3a4e7435cd81963ffd16b7a9128daf75f35f6de7",
}
TOOLS = {"scb-check": "0.2.0", "ruff": "0.15.7", "vulture": "2.16", "deptry": "0.25.1", "pip-audit": "2.10.1"}

# --- thresholds ------------------------------------------------------------------------------------

CC_THRESHOLD = 10  # functions above this cyclomatic complexity get an issue
MEDIUM_MASS = 100  # ...medium when CC x sqrt(SLOC) reaches this (e.g. CC 15 over 45 lines), else low
MEDIUM_DUP_LINES = 20  # duplicate groups are medium when lines x (copies - 1) reaches this, else low
MIN_CLONE_LINES = 10  # smaller duplicate blocks still count in Clone Ratio but get no issue
MIN_DEAD_LINES = 5  # smaller unused symbols still count in dead-code LOC but get no issue
LARGE_FILE_MB = 5
DEFAULT_EXCLUDES = (
    "tests/**", "test/**", "**/test_*.py", "**/*_test.py", "**/conftest.py", "docs/**", ".cleanup/**",
)
# SlopCodeBench (arXiv 2603.24755): maintained human Python repos vs agent-written code. Only these two
# are comparable with scb-check 0.2.0; its Violation % and Clone Ratio are measured differently.
REFERENCE = {"verbosity": (0.15, 0.33), "erosion": (0.31, 0.68)}

SEVERITIES = ("critical", "high", "medium", "low")
AGENT_FIELDS = ("status", "severity", "fix", "done_when", "note", "effort")
MEASURE_SOURCES = {"scb-check", "ruff", "vulture", "deptry", "pip-audit", "license"}
HERE = Path(__file__).resolve().parent
RULES = HERE / "leak_rules.toml"

# --- small helpers ---------------------------------------------------------------------------------


class ToolError(Exception):
    """A tool could not run; its metrics are recorded as not measured."""

    def __init__(self, tool: str, message: str):
        super().__init__(message)
        self.tool = tool


def die(message: str, code: int = 2):
    print(f"error: {message}", file=sys.stderr)
    sys.exit(code)


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
    if p.exists():
        return json.loads(p.read_text())
    return {"schema": 1, "repo": d.parent.name, "tools": {}, "issues": []}


def open_issues(data: dict, *severities: str) -> list:
    return [
        i for i in data["issues"]
        if i["status"] == "open" and (not severities or i["severity"] in severities)
    ]


def leak_scan_current(data: dict, repo: Path) -> bool:
    return data.get("leak_scan", {}).get("commit") == head(repo)


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
    blocked = not leak_scan_current(data, repo) or open_issues(data, "critical", "high")
    data["release_status"] = "blocked" if blocked else "clear"
    meta = [f"  {json.dumps(k)}: {json.dumps(v, ensure_ascii=False)}," for k, v in data.items()
            if k not in ("issues", "defaults")]
    issues = ",\n".join("    " + json.dumps(i, ensure_ascii=False) for i in data["issues"])
    defaults = json.dumps(data.get("defaults", {}), separators=(",", ":"))
    (d / "issues.json").write_text("{\n" + "\n".join(meta) + f'\n  "issues": [\n{issues}\n  ],\n'
                                   f'  "defaults": {defaults}\n}}\n')


def issue(prefix, source, category, rule, severity, summary, *, file="", symbol="", key="", line=None,
          end_line=None, evidence=None, fix="", done_when="", effort="S", status="open", note="") -> dict:
    """One finding. IDs hash stable facts (never line numbers), so re-runs keep them."""
    basis = key or "|".join((category, rule, file, symbol))
    return {
        "id": f"{prefix}-{hashlib.sha1(basis.encode()).hexdigest()[:8]}",
        "source": source, "category": category, "rule": rule, "severity": severity,
        "file": file, "line": line, "end_line": end_line, "symbol": symbol, "summary": summary,
        "evidence": evidence or {}, "fix": fix, "done_when": done_when, "effort": effort,
        "status": status, "note": note,
    }


def _digest(value) -> str:
    return hashlib.sha1(json.dumps(value).encode()).hexdigest()[:10]


def merge(data: dict, found: list, sources: set):
    """Fold fresh findings into issues.json.

    A field an agent edited (status, severity, fix, ...) survives; one still holding the value the tool
    last wrote takes the new value. Open issues from `sources` that are no longer detected become fixed;
    fixed issues that are detected again reopen.
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
        reopened = old["status"] == "fixed" and "status" not in edited
        old.update(new)
        old.update(edited)
        if reopened:
            old["note"] = "detected again after being marked fixed"
            defaults[ident]["note"] = _digest(old["note"])
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


def leak_issues(repo: Path, findings: list) -> list:
    """Group Betterleaks findings by distinct secret. Plaintext never leaves this function."""
    head_blobs = {}

    def in_head(path: str, needle: str) -> bool:
        if path not in head_blobs:
            p = subprocess.run(["git", "-C", str(repo), "show", f"HEAD:{path}"], capture_output=True)
            head_blobs[path] = p.stdout if p.returncode == 0 else None
        blob = head_blobs[path]
        return blob is not None and needle.encode() in blob

    groups = {}
    for f in findings:
        secret = f.get("Secret") or ""
        internal = "internal" in (f.get("Tags") or [])
        # a credential is one issue wherever it appears (it is rotated once); internal references are
        # grouped per file, which is how they get fixed
        basis = f"internal|{f['RuleID']}|{f['File']}" if internal else secret or f"path:{f['File']}"
        key = hashlib.sha256(basis.encode()).hexdigest()
        g = groups.setdefault(key, {
            "rules": {}, "internal": internal, "locs": [], "fingerprints": [], "at_head": False,
            "first": None, "example": (f["File"], f["StartLine"]), "matches": set(),
            "preview": f"{secret[:4]}… ({len(secret)} chars)" if secret else "(file name)",
        })
        g["rules"][f["RuleID"]] = (f.get("Attributes") or {}).get("confidence", "high")
        if internal:
            g["matches"].add(secret[:80])
            g["preview"] = min(g["matches"])
        g["locs"].append(f"{f['File']}:{f['StartLine']}@{f['Commit'][:8]}")
        g["fingerprints"].append(f["Fingerprint"])
        g["at_head"] = g["at_head"] or in_head(f["File"], secret)
        if g["first"] is None or f.get("Date", "") < g["first"][0]:
            g["first"] = (f.get("Date", ""), f["Commit"][:12])

    out = []
    for key, g in groups.items():
        rule = min(g["rules"], key=lambda r: (CONFIDENCE_RANK.get(g["rules"][r], 9), "generic" in r, r))
        file, line = g["example"]
        where = "at HEAD" if g["at_head"] else "only in history"
        evidence = {
            "rules": sorted(g["rules"]), "preview": g["preview"], "at_head": g["at_head"],
            "first_commit": g["first"][1], "first_date": g["first"][0],
            "occurrences": len(g["locs"]), "locations": g["locs"][:10], "fingerprints": g["fingerprints"][:10],
        }
        if g["internal"]:
            more = f" and {len(g['matches']) - 1} more" if len(g["matches"]) > 1 else ""
            evidence["matches"] = sorted(g["matches"])[:20]
            out.append(issue(
                "INT", "betterleaks", "internal-ref", rule, "high" if g["at_head"] else "low",
                f"Internal reference `{g['preview']}`{more} ({where})",
                key=f"INT|{key}", file=file, line=line, evidence=evidence,
                fix=("Replace it with a config value, CLI argument or environment variable "
                     "(or a neutral placeholder in docs)." if g["at_head"] else
                     "Only matters if the history is published; a fresh squashed history drops it."),
                done_when="`cleanup.py leaks` no longer reports it at HEAD."))
            continue
        certain = g["rules"][rule] in ("high", "medium")
        severity = "critical" if certain else "high"
        if certain:
            fix = ("User: revoke/rotate this credential at its provider first. "
                   + ("Then read it from an environment variable (add a `.env.example`, gitignore `.env`) "
                      "and keep it out of the published history: publish a fresh squashed history, or "
                      "rewrite history with the user's approval." if g["at_head"] else
                      "It is gone from HEAD but still in history: publish a fresh squashed history, or "
                      "rewrite history with the user's approval."))
        else:
            fix = ("Check whether this is a real credential by looking at the location; never print the "
                   "value. If it is not, dismiss it with the reason and add its fingerprint to "
                   "`.betterleaksignore`. If it is, rotate it and remove it as for any secret.")
        out.append(issue(
            "LEAK", "betterleaks", "leak", rule, severity,
            f"{'Secret' if certain else 'Possible secret'} `{g['preview']}` matched {rule} ({where})",
            key=f"LEAK|{key}", file=file, line=line, evidence=evidence, fix=fix,
            done_when="Rotated (user confirms) and absent from HEAD and from any history that will be published."))
    return out


def large_file_issues(repo: Path) -> list:
    paths = {}
    for row in git(repo, "rev-list", "--objects", "--all").splitlines():
        sha, _, path = row.partition(" ")
        if path:
            paths.setdefault(sha, path)
    p = run(["git", "-C", repo, "cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)"],
            input="\n".join(paths) + "\n")
    sizes = {}
    for row in p.stdout.splitlines():
        sha, kind, size = row.split()
        if kind == "blob" and int(size) > LARGE_FILE_MB * 1_000_000:
            sizes[paths[sha]] = max(sizes.get(paths[sha], 0), int(size))
    at_head = set(git(repo, "ls-tree", "-r", "--name-only", "HEAD").splitlines())
    out = []
    for path, size in sorted(sizes.items(), key=lambda kv: -kv[1]):
        here = path in at_head
        out.append(issue(
            "FILE", "git-history", "large-file", "large-file", "high" if here else "medium",
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
        config, report = Path(tmp) / "rules.toml", Path(tmp) / "report.json"
        config.write_text(rules)
        p = run([exe, "git", repo, "--config", config, "--report-format", "json", "--report-path", report,
                 "--exit-code", "0", "--no-banner", "--no-color", "--log-level", "error",
                 "--gitleaks-ignore-path", repo])
        if p.returncode or not report.exists():
            die(f"betterleaks failed (exit {p.returncode}): {p.stderr.strip()[-400:]}")
        found = leak_issues(repo, json.loads(report.read_text() or "[]"))
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
    """Functions and classes per file (qualname, span) and the lines that hold code."""

    def __init__(self, repo: Path, files: list):
        self.defs, self.code, self.syntax_errors = {}, {}, []
        for rel in files:
            text = (repo / rel).read_text(errors="replace")
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


def parse_scb(text: str) -> list:
    findings = []
    for row in text.splitlines():
        if m := SCB_HEADER.match(row):
            findings.append({"rule": m["a"] or m["b"] or m["c"], "msg": m["msg"], "locs": []})
        elif (m := SCB_LOCATION.search(row)) and findings:
            findings[-1]["locs"].append((m["path"], int(m["line"])))
    return findings


def scb_check(repo, files, index, snap) -> list:
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
        if p.returncode not in (0, 1):
            raise ToolError("scb-check", p.stderr.strip()[-300:])
        m = json.loads(p.stdout)
        human = run([*base, "--min-duplicate-lines", MIN_CLONE_LINES], env=env).stdout
    loc = m["total_loc"] or 1
    snap["metrics"].update(
        loc=m["total_loc"], verbosity=round(m["verbosity"], 4), erosion=round(m["erosion"], 4),
        cog_erosion=round(m["cog_erosion"], 4), violation_pct=round(m["ast_grep_flagged_loc"] / loc, 4),
        clone_ratio=round(m["clone_loc"] / loc, 4),
        # absolute amounts behind the ratios: deleting clean code raises a ratio without adding slop
        verbosity_flagged_loc=m["verbosity_flagged_loc"], high_cc_mass=round(m["high_cc_mass"], 1))
    if "ast-grep" in p.stderr:  # no ast-grep binary worked: the slop rules silently counted nothing
        snap["not_measured"]["violation_pct"] = "ast-grep did not run"

    def rel(path):
        try:
            return Path(path).resolve().relative_to(src).as_posix()
        except ValueError:
            return path

    clones, out = [], []
    for f in parse_scb(human):
        locs = [(rel(path), line) for path, line in f["locs"]]
        if f["rule"] == "duplicate-structure" and (c := SCB_CLONE.search(f["msg"])):
            clones.append((int(c[1]) * (int(c[2]) - 1), int(c[1]), int(c[2]), locs))
        elif f["rule"] in ("trivial-wrapper", "low-use-short-function") and locs:
            file, line = locs[0]
            name = (index.at_def_line(file, line) or (0, 0, "<module>"))[2]
            flag ="pass-through-method" if f["rule"] == "trivial-wrapper" else "shallow-module"
            out.append(issue(
                "RF", "scb-check", "red-flag", flag, "medium", f"{f['rule']}: {f['msg']}",
                file=file, symbol=name, line=line, evidence={"scb_rule": f["rule"]}, effort="S",
                fix=("Inline it into its callers or remove it. If it is deliberate (a public alias, or an "
                     "override such as `model.train = disabled_train`), dismiss the issue with that reason."),
                done_when="scb-check no longer reports it (or it is dismissed), and the safety net passes."))
    for weight, lines, count, locs in clones:
        members = sorted({(file, index.qualname(file, line)) for file, line in locs})
        where = ", ".join(f"{file}:{line}" for file, line in locs[:4])
        out.append(issue(
            "RF", "scb-check", "red-flag", "repetition", "medium" if weight >= MEDIUM_DUP_LINES else "low",
            f"{lines}-line block duplicated {count}x: {where}",
            key="RF|repetition|" + "|".join(f"{f}:{q}" for f, q in members),
            file=locs[0][0], line=locs[0][1], symbol=", ".join(q for _, q in members),
            evidence={"lines": lines, "instances": count, "weight": weight,
                      "locations": [f"{f}:{n}" for f, n in locs]},
            effort="M" if lines >= 30 else "S",
            fix=("Extract the shared block into one function, or delete the copy that is no longer used. "
                 "Pass the real differences as a few parameters."),
            done_when="scb-check no longer reports this duplicate group and the safety net passes."))
    return out


# --- measure: ruff, vulture ------------------------------------------------------------------------

RUFF_C901 = re.compile(r"^`(?P<name>[^`]+)` is too complex \((?P<cc>\d+) > \d+\)$")
VULTURE = re.compile(
    r"^(?P<path>.+?):(?P<line>\d+): (?P<msg>unused (?P<kind>function|class|method|property|variable|"
    r"attribute|import) '(?P<name>[^']+)'|unreachable code after '[^']+'|.+?) "
    r"\((?P<conf>\d+)% confidence(?:, (?P<size>\d+) lines?)?\)$")
TEXT_SUFFIXES = {".yaml", ".yml", ".json", ".toml", ".cfg", ".ini", ".sh", ".bash", ".slurm", ".sbatch",
                 ".ipynb", ".jinja", ".j2"}
DEAD_HIGH_LINES = 20  # unused symbols at least this long are high (remove before release), smaller ones low


def ruff_cc(repo, files, index, snap) -> list:
    if not files:
        snap["metrics"]["cc_over_10"] = 0
        return []
    p = run([uvx(), f"ruff@{TOOLS['ruff']}", "check", "--isolated", "--no-cache", "--exit-zero",
             "--ignore-noqa", "--select", "C901", "--config", f"lint.mccabe.max-complexity = {CC_THRESHOLD}",
             "--output-format", "json", *files], cwd=repo)
    if p.returncode:
        raise ToolError("ruff", p.stderr.strip()[-300:])
    rows = []
    for r in json.loads(p.stdout):
        m = RUFF_C901.match(r["message"]) if r["code"] == "C901" else None
        if not m:
            continue
        file = Path(r["filename"]).resolve().relative_to(repo).as_posix()
        d = index.at_def_line(file, r["location"]["row"])
        start, end, qual = (d[0], d[1], d[2]) if d else (r["location"]["row"],) * 2 + (m["name"],)
        sloc = max(index.sloc(file, start, end), 1)
        cc = int(m["cc"])
        rows.append((cc * math.sqrt(sloc), cc, sloc, file, qual, start, end))
    snap["metrics"]["cc_over_10"] = len(rows)
    if index.syntax_errors:
        snap["not_measured"]["cc_partial"] = f"syntax errors in {len(index.syntax_errors)} files"
    out = []
    for mass, cc, sloc, file, qual, start, end in rows:
        out.append(issue(
            "CX", "ruff", "complexity", "C901", "medium" if mass >= MEDIUM_MASS else "low",
            f"`{qual}` has cyclomatic complexity {cc} ({sloc} SLOC)",
            file=file, symbol=qual, line=start, end_line=end,
            evidence={"cc": cc, "sloc": sloc, "mass": round(mass, 1), "weight": round(mass, 1)},
            effort="L" if sloc > 150 else "M",
            fix=("Simplify the control flow: drop dead branches and flags, return early, move special cases "
                 "out of the general path. Extracting a step with its own clear purpose is fine; moving "
                 "branches into a helper that only this function calls, with most of its arguments, is not. "
                 "If every branch is needed and it reads clearly, dismiss it with that reason. Add a "
                 "characterization test first."),
            done_when=(f"ruff reports CC <= {CC_THRESHOLD} for it (or it is gone, or dismissed with a reason), "
                       "and the safety net passes.")))
    return out


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


def _named_in(refs: list, name: str, module: str) -> list:
    """Files that name `name` as `x.name`, or as `name` alongside the module's name. A bare common
    word (`Model`, `run`) elsewhere is not a reference, and neither is a dunder."""
    if name.startswith("__") and name.endswith("__"):
        return []
    dotted = re.compile(rf"\.{re.escape(name)}\b")
    word, mod = re.compile(rf"\b{re.escape(name)}\b"), re.compile(rf"\b{re.escape(module)}\b")
    literal = re.compile(rf"[\"'](?:[\w.:/-]*\.)?{re.escape(name)}[\"']")  # "SetupCallback", "main.SetupCallback"
    hits = []
    for rel, text, strings_only in refs:
        if literal.search(text) if strings_only else dotted.search(text) or (word.search(text) and mod.search(text)):
            hits.append(rel)
    return hits


def vulture_dead(repo, files, index, snap) -> list:
    if not files:
        snap["metrics"]["dead_code_loc"] = 0
        return []
    p = run([uvx(), f"vulture@{TOOLS['vulture']}", *files, "--min-confidence", "60", "--sort-by-size"], cwd=repo)
    if p.returncode not in (0, 3):
        raise ToolError("vulture", (p.stderr or p.stdout).strip()[-300:])
    refs = _references(repo, files)
    coverage = {}
    if (cov := repo / "coverage.json").exists():  # optional: a coverage.py JSON report from any run
        for name, f in json.loads(cov.read_text()).get("files", {}).items():
            coverage[Path(name).as_posix()] = set(f.get("executed_lines", []))
    total, out = 0, []
    for row in p.stdout.splitlines():
        m = VULTURE.match(row)
        # methods and properties are skipped: in ML code they are mostly framework hooks (Lightning's
        # training_step, nn.Module overrides) that vulture can't see being called
        if not m or m["kind"] in ("method", "property", "variable", "attribute", "import"):
            continue
        file, line, size = Path(m["path"]).as_posix(), int(m["line"]), int(m["size"] or 1)
        d = index.at_def_line(file, line)
        qual = d[2] if d and m["name"] and d[2].endswith(m["name"]) else (m["name"] or index.qualname(file, line))
        name = qual.rsplit(".", 1)[-1]
        module = Path(file).parent.name if Path(file).stem == "__init__" else Path(file).stem
        hits = _named_in(refs, name, module) if m["kind"] else []  # unreachable code can't be referenced
        ran = bool(coverage.get(file, set()) & set(range(line, line + size)))
        if not hits and not ran:
            total += size
        if size < MIN_DEAD_LINES:
            continue
        note = (f"referenced by name in {', '.join(hits[:3])}" if hits else
                "executed in the coverage run" if ran else "")
        out.append(issue(
            "DEAD", "vulture", "dead-code", m["kind"] or "unreachable",
            "high" if size >= DEAD_HIGH_LINES else "low",
            f"{m['msg']} ({size} lines, vulture {m['conf']}% confidence)",
            file=file, symbol=qual, line=line, end_line=line + size - 1,
            evidence={"lines": size, "confidence": int(m["conf"]), "referenced_in": hits[:5],
                      "executed": ran if coverage else None},
            status="dismissed" if note else "open", note=note, effort="S",
            fix=("Confirm nothing references it (yaml `_target_`/`target:` strings, registries, getattr, "
                 "notebooks, shell scripts), then delete it."),
            done_when="Deleted, and the import test and safety net pass."))
    snap["metrics"]["dead_code_loc"] = total
    return out


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


def deptry_deps(repo, files, index, snap) -> list:
    pyproject = _pyproject(repo)
    if pyproject.get("project", {}).get("dependencies") is not None or \
            pyproject.get("tool", {}).get("poetry", {}).get("dependencies"):
        source = ["--config", repo / "pyproject.toml"]
    elif (repo / "requirements.txt").exists():
        source = ["--requirements-files", repo / "requirements.txt"]
    else:
        snap["not_measured"]["deptry"] = "no pyproject.toml dependencies or requirements.txt"
        return [issue("DEPS", "deptry", "readiness", "no-dependency-list", "low",
                      "Dependencies aren't declared in pyproject.toml or requirements.txt, so they can't be checked",
                      key="DEPS|declare", effort="S",
                      fix="Declare the runtime dependencies in pyproject.toml ([project].dependencies).",
                      done_when="`cleanup.py measure` runs deptry without 'not measured'.")]
    with tempfile.TemporaryDirectory(prefix="cleanup-deptry-") as tmp:
        report = Path(tmp) / "deptry.json"
        p = run([uvx(), f"deptry@{TOOLS['deptry']}", repo, *source, "--package-module-name-map", DEPTRY_MAP,
                 "--json-output", report])
        if not report.exists():
            raise ToolError("deptry", (p.stderr or p.stdout).strip()[-300:])
        violations = json.loads(report.read_text())
    out = []
    for v in violations:
        code, module = v["error"]["code"], v["module"]
        file = v["location"]["file"]
        try:
            file = Path(file).resolve().relative_to(repo).as_posix()
        except ValueError:
            pass
        fixes = {"DEP001": f"Declare the package that provides `{module}` as a dependency.",
                 "DEP002": f"Remove `{module}` from the dependencies if it is really unused.",
                 "DEP003": f"`{module}` is only installed transitively: declare it directly.",
                 "DEP004": f"`{module}` is a dev dependency but the code imports it: move it to the main dependencies."}
        out.append(issue(
            "DEP", "deptry", "readiness", code, "low" if code == "DEP002" else "high",
            v["error"]["message"], file=file, symbol=module, line=v["location"]["line"],
            fix=fixes.get(code, v["error"]["message"]), effort="S",
            done_when="deptry no longer reports it."))
    snap["metrics"]["deps_missing"] = sum(v["error"]["code"] != "DEP002" for v in violations)
    return out


PINNED = re.compile(r"^[A-Za-z0-9][\w.\-\[\], ]*==\s*[\w.+!-]+\s*(?:;.*)?(?:\\)?$")


def _fully_pinned(path: Path) -> bool:
    lines = [ln.split(" #")[0].strip() for ln in path.read_text().splitlines()]
    reqs = [ln for ln in lines if ln and not ln.startswith(("#", "--hash"))]
    return bool(reqs) and all(PINNED.match(ln) for ln in reqs)


def pip_audit_vulns(repo, files, index, snap) -> list:
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
            snap["not_measured"]["pip-audit"] = "no pinned source (uv.lock, pylock, pinned requirements, .venv)"
            return [issue("DEPS", "pip-audit", "readiness", "no-pinned-dependencies", "low",
                          "No lockfile or pinned requirements: dependencies can't be audited and installs "
                          "aren't reproducible", key="DEPS|pin", effort="S",
                          fix="Commit a lockfile (uv.lock) or a fully pinned requirements.txt.",
                          done_when="`cleanup.py measure` runs pip-audit without 'not measured'.")]
        report = Path(tmp) / "audit.json"
        p = run([uvx(), f"pip-audit@{TOOLS['pip-audit']}", *source, "--format", "json", "-o", report,
                 "--progress-spinner", "off"])
        if p.returncode not in (0, 1) or not report.exists():
            raise ToolError("pip-audit", p.stderr.strip()[-300:])
        deps = json.loads(report.read_text()).get("dependencies", [])
    snap["metrics"]["pip_audit_source"] = label
    out = []
    for dep in deps:
        vulns = {v["id"]: v for v in dep.get("vulns", [])}
        if not vulns:
            continue
        fixed = [min(v["fix_versions"], key=version_key) for v in vulns.values() if v["fix_versions"]]
        target = max(fixed, key=version_key) if fixed else None
        ids = sorted(vulns)
        out.append(issue(
            "VULN", "pip-audit", "vulnerable-dependency", "pip-audit", "high" if target else "low",
            f"{dep['name']} {dep['version']}: {len(ids)} known vulnerabilities ({', '.join(ids[:3])}"
            f"{', ...' if len(ids) > 3 else ''})" + (f"; fixed in {target}" if target else "; no fix yet"),
            key=f"VULN|{dep['name'].lower()}", file=label, symbol=dep["name"],
            evidence={"version": dep["version"], "vulns": ids, "fix_to": target,
                      "aliases": sorted({a for v in vulns.values() for a in v.get("aliases", [])})[:10]},
            fix=(f"Bump {dep['name']} to >= {target} in the dependency spec, then run the safety net."
                 if target else "No fixed release yet: note it, or drop the dependency if it is easy."),
            done_when=f"pip-audit reports no vulnerabilities for {dep['name']}."))
    snap["metrics"]["vulnerable_deps"] = len(out)
    return out


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


def license_readme(repo, files, index, snap) -> list:
    out = []
    names = git(repo, "ls-tree", "--name-only", "HEAD").splitlines()
    lic_files = [n for n in names if re.fullmatch(r"(?i)(?:licen[cs]e|copying)(?:\.(?:md|txt|rst))?", n)]
    repo_license = None
    if not lic_files:
        out.append(issue(
            "LIC", "license", "license", "missing-license", "high",
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
                "LIC", "license", "license", "unrecognized-license", "low",
                f"{lic_files[0]} doesn't match a common license (MIT, Apache-2.0, BSD, GPL, MPL)",
                key="LIC|unrecognized", file=lic_files[0], effort="S",
                fix="Check the text; a standard license text lets people and tools recognize it.",
                done_when="`cleanup.py measure` recognizes the license."))
        declared = _pyproject(repo).get("project", {}).get("license")
        declared = declared.get("text") if isinstance(declared, dict) else declared
        if repo_license and isinstance(declared, str) and family(declared) != family(repo_license):
            out.append(issue(
                "LIC", "license", "license", "license-mismatch", "high",
                f"pyproject.toml declares `{declared}` but {lic_files[0]} is {repo_license}",
                key="LIC|mismatch", file="pyproject.toml", effort="S",
                fix="Make pyproject's `license` match the LICENSE file.",
                done_when="Both name the same license."))
    snap["metrics"]["license"] = repo_license or ("missing" if not lic_files else "unrecognized")

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
            "LIC", "license", "license", "third-party-license", "high",
            f"{len(paths)} file(s) carry a {fam} license header but the repo is "
            f"{repo_license or 'unlicensed'}: {', '.join(paths[:4])}{', ...' if len(paths) > 4 else ''}",
            key=f"LIC|vendored|{fam}", file=paths[0], evidence={"files": paths[:50]}, effort="S",
            fix=(f"Keep these notices. Check that {fam} code may ship under the repo license (GPL/AGPL code "
                 "can't be relicensed as MIT or Apache-2.0; non-commercial code isn't open source) and "
                 "credit the original project in the README or a NOTICE file."),
            done_when="The user has confirmed the licenses are compatible (or the files are replaced)."))

    if not any(re.fullmatch(r"(?i)readme(?:\.\w+)?", n) for n in names):
        out.append(issue(
            "DOC", "license", "readiness", "missing-readme", "high", "No README at the repo root",
            key="DOC|readme", effort="M",
            fix="Add a README: what the code does, how to install it, a minimal quickstart, how to cite it.",
            done_when="README exists and its quickstart runs."))
    return out


def count_suppressions(repo: Path, files: list) -> int:
    pattern = re.compile(r"#\s*(?:noqa|scbc\s+ignore)", re.I)
    return sum(len(pattern.findall((repo / f).read_text(errors="replace"))) for f in files)


# --- measure ---------------------------------------------------------------------------------------

STEPS = (("scb-check", scb_check), ("ruff", ruff_cc), ("vulture", vulture_dead),
         ("deptry", deptry_deps), ("pip-audit", pip_audit_vulns), ("license", license_readme))


def mark_inside_dead_code(found: list):
    """Complexity and red-flag issues inside a dead-code candidate drop to low: delete before refactoring."""
    dead = [i for i in found if i["category"] == "dead-code" and i["status"] == "open"]
    for i in found:
        if i["category"] not in ("complexity", "red-flag") or not i.get("line"):
            continue
        host = next((d for d in dead if d["file"] == i["file"] and d["line"] <= i["line"] <= d["end_line"]), None)
        if host:
            i["severity"] = "low"
            i["evidence"]["inside_dead_code"] = host["id"]
            i["fix"] = f"Inside dead-code candidate {host['id']}: delete that instead, unless it turns out to be used. " + i["fix"]


def cmd_measure(a):
    repo = repo_root(a.repo)
    d = state_dir(repo)
    data = load_issues(d)
    require_leak_scan(data, repo)
    excludes = list(dict.fromkeys([*DEFAULT_EXCLUDES, *a.exclude]))
    files = shipped_files(repo, excludes)
    index = CodeIndex(repo, files)
    dirty = bool(git(repo, "status", "--porcelain", "--untracked-files=no").strip())
    snap = {"label": a.label, "date": now(), "commit": head(repo), "dirty": dirty, "scope": excludes,
            "tools": dict(TOOLS), "files": len(files), "metrics": {}, "not_measured": {}}
    found, failed = [], set()
    for name, step in STEPS:
        try:
            found += step(repo, files, index, snap)
        except ToolError as e:
            snap["not_measured"][e.tool] = str(e) or "failed"
            failed.add(name)
    snap["metrics"]["suppressions"] = count_suppressions(repo, files)
    mark_inside_dead_code(found)
    merge(data, found, MEASURE_SOURCES - failed)
    data["tools"].update(TOOLS)
    save_issues(d, data, repo)
    snap["metrics"]["red_flags_open"] = len([i for i in open_issues(data) if i["category"] == "red-flag"])
    snap["metrics"]["leaks_open"] = len([i for i in open_issues(data) if i["category"] == "leak"])
    snap["open_issues"] = {s: len(open_issues(data, s)) for s in SEVERITIES}
    snap["release_status"] = data["release_status"]
    path = d / "snapshots" / f"{a.label}.json"
    if path.exists():
        print(f"note: overwriting snapshot {a.label}", file=sys.stderr)
    path.write_text(json.dumps(snap, indent=2) + "\n")
    print_scorecard(snap)
    print(f"-> {d / 'issues.json'}  ·  {path}")


ROWS = (  # key, label, format
    ("loc", "LOC", "{:,}"), ("verbosity", "Verbosity", "{:.3f}"), ("erosion", "Erosion", "{:.3f}"),
    ("cog_erosion", "Cognitive erosion", "{:.3f}"), ("violation_pct", "Violation %", "{:.3f}"),
    ("clone_ratio", "Clone Ratio", "{:.3f}"), ("cc_over_10", f"Functions CC > {CC_THRESHOLD}", "{:,}"),
    ("dead_code_loc", "Dead code LOC", "{:,}"), ("red_flags_open", "Red flags open", "{:,}"),
    ("leaks_open", "Leaks open", "{:,}"), ("vulnerable_deps", "Vulnerable deps", "{:,}"),
    ("deps_missing", "Dependency problems", "{:,}"), ("suppressions", "Suppressions", "{:,}"),
)


def fmt(key: str, value) -> str:
    fmt_ = next((f for k, _, f in ROWS if k == key), "{}")
    return "-" if value is None else fmt_.format(value)


def print_scorecard(snap: dict):
    m = snap["metrics"]
    print(f"scorecard  {snap['label']}  @ {snap['commit'][:8]}{' (dirty)' if snap['dirty'] else ''}  "
          f"release: {snap['release_status'].upper()}")
    for key, label, _ in ROWS:
        if key not in m:
            continue
        ref = REFERENCE.get(key)
        hint = f"   (paper: human {ref[0]}, agent code {ref[1]})" if ref else ""
        print(f"  {label:<22}{fmt(key, m[key]):>10}{hint}")
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


def latest_snapshot(d: Path, exclude: str = "") -> dict | None:
    snaps = [json.loads(p.read_text()) for p in (d / "snapshots").glob("*.json")]
    snaps = [s for s in snaps if s["label"] != exclude]
    return max(snaps, key=lambda s: s["date"]) if snaps else None


def cmd_compare(a):
    repo = repo_root(a.repo)
    d = state_dir(repo)
    base = load_snapshot(d, a.base)
    after = load_snapshot(d, a.to) if a.to else latest_snapshot(d, exclude=a.base)
    if after is None:
        die("nothing to compare: run `cleanup.py measure --label <after>` first")
    print(f"{'metric':<22}{base['label']:>14}{after['label']:>14}{'delta':>10}")
    for key, label, _ in ROWS:
        b, n = base["metrics"].get(key), after["metrics"].get(key)
        if b is None and n is None:
            continue
        delta = "" if b is None or n is None else (f"{n - b:+,}" if isinstance(b, int) else f"{n - b:+.3f}")
        print(f"{label:<22}{fmt(key, b):>14}{fmt(key, n):>14}{delta:>10}")
    for s in SEVERITIES:
        print(f"{'open ' + s:<22}{base['open_issues'][s]:>14}{after['open_issues'][s]:>14}"
              f"{after['open_issues'][s] - base['open_issues'][s]:>+10}")
    for w in comparison_warnings(base, after):
        print(f"warning: {w}")


def comparison_warnings(base: dict, after: dict) -> list:
    """Regressions in absolute terms. Ratios are not used: deleting clean code raises them on its own."""
    out = []
    for key, label in (("verbosity_flagged_loc", "verbosity-flagged LOC"),
                       ("high_cc_mass", "mass in functions with CC > 10"), ("suppressions", "suppressions")):
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
5. Record each outcome in `.cleanup/issues.json` (one issue per line; `grep <ID>` finds it): `status` `fixed`, or `dismissed` with a `note`. A false positive, a deliberate design, or not worth the change are all valid reasons. Bugs you find get their own issue (`"category": "bug"`, `"source": "agent"`, ID `BUG-A1`, ...) and their own commit, never inside a refactor. Don't edit this report; regenerate it.
6. No nitpicks: only change what makes the code meaningfully easier to understand, use or change. A refactor that adds lines needs a reason.
7. Re-measure when you stop (the label is any name): `.cleanup/cleanup.py leaks .`, `.cleanup/cleanup.py measure . --label after-<what>`, `.cleanup/cleanup.py compare . baseline`, `.cleanup/cleanup.py report .` (needs uv)."""

PHASES = (("High", ("critical", "high")), ("Medium", ("medium",)), ("Low", ("low",)))


def cell(text) -> str:
    return str(text if text is not None else "").replace("|", "\\|").replace("\n", " ")


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
        lines.append("| metric | " + " | ".join(s["label"] for s in cols)
                     + (" | Δ" if len(cols) == 2 else "") + " | reference (paper: human / agent) |")
        lines.append("|---" * (len(cols) + (3 if len(cols) == 2 else 2)) + "|")
        for key, label, _ in ROWS:
            vals = [s["metrics"].get(key) for s in cols]
            if all(v is None for v in vals):
                continue
            row = [label, *[fmt(key, v) for v in vals]]
            if len(cols) == 2:
                b, n = vals
                row.append("" if b is None or n is None else
                           f"{n - b:+,}" if isinstance(b, int) else f"{n - b:+.3f}")
            ref = REFERENCE.get(key)
            row.append(f"{ref[0]} / {ref[1]}" if ref else "")
            lines.append("| " + " | ".join(row) + " |")
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
        done = phase != "Low"
        lines.append("| ID | severity | where | problem | fix |" + (" done when |" if done else "") + " effort |")
        lines.append("|---|---|---|---|---|" + ("---|" if done else "") + "---|")
        prev = {}
        for i in items:
            same = {k: "same as above" if i[k] and i[k] == prev.get(k) else i[k] for k in ("fix", "done_when")}
            row = [i["id"], i["severity"], where(i), i["summary"], same["fix"]]
            row += [same["done_when"]] if done else []
            lines.append("| " + " | ".join(cell(x) for x in [*row, i["effort"]]) + " |")
            prev = i
    dismissed = [i for i in data["issues"] if i["status"] == "dismissed"]
    fixed = [i for i in data["issues"] if i["status"] == "fixed"]
    lines += ["", f"## Dismissed ({len(dismissed)})", ""]
    lines += (["| ID | where | problem | reason |", "|---|---|---|---|"]
              + [f"| {i['id']} | {cell(where(i))} | {cell(i['summary'])} | {cell(i['note'])} |" for i in dismissed]
              if dismissed else ["None."])
    lines += ["", f"Fixed so far: {len(fixed)} issue(s)."]
    return "\n".join(lines) + "\n"


def cmd_report(a):
    repo = repo_root(a.repo)
    d = state_dir(repo)
    data = load_issues(d)
    require_leak_scan(data, repo)
    save_issues(d, data, repo)  # refresh release status
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
    {"leaks": cmd_leaks, "measure": cmd_measure, "compare": cmd_compare, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
