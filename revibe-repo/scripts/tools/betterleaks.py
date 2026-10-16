"""Betterleaks over the git history, once, then over the files at HEAD: credentials and internal references.
It is not on PyPI, so a pinned release binary is downloaded once, checked against its SHA-256 and cached."""

import datetime as dt
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
from tools.base import Result, Tool


class Betterleaks(Tool):
    """Credentials and internal references anywhere in the git history."""

    name = "betterleaks"
    version = "1.9.0"  # not on PyPI: the release binary is pinned here, with its checksums
    sha256 = {  # checksums.txt of the v1.9.0 release, signature verified with sigstore
        ("darwin", "arm64"): "83dd7eaab13d44a1e8e347a17064cdef153a1478db473276d41b38d55613d3f2",
        ("darwin", "x64"): "68dfd83458d9e7f90663daa7a81c7f944083b170dae82e4f5f8d0581ed519ef1",
        ("linux", "arm64"): "1d39116e0a58dc94574715e2aa12a2dbd5062f193eee3fec011fef6ba06bd13b",
        ("linux", "x64"): "f8b185a39ffcece2a1ca82bf3a4e7435cd81963ffd16b7a9128daf75f35f6de7",
    }
    policy = {
        "credential": "credential matched by a confident rule, at HEAD or only in history",
        "possible-credential": "possible credential (low-confidence rule or sensitive file name)",
        "internal-head": "internal path, host, email or account at HEAD",
        "internal-history": "internal reference only in history: a fresh squashed history drops it",
    }

    def tarball_ok(self, data: bytes, os_name: str, arch: str) -> bool:
        return hashlib.sha256(data).hexdigest() == self.sha256.get((os_name, arch))

    def executable(self) -> Path:
        if env := os.environ.get("BETTERLEAKS"):
            exe = Path(env).expanduser()
            if not (exe.is_file() and os.access(exe, os.X_OK)):
                raise SystemExit(f"BETTERLEAKS={env} is not an executable file")
            return exe
        os_name, arch = _platform()
        if (os_name, arch) not in self.sha256:
            raise SystemExit(f"no pinned Betterleaks build for {sys.platform}/{platform.machine()}; "
                "set BETTERLEAKS=/path/to/betterleaks")
        cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
        version = self.version
        cache = cache / "revibe-repo" / f"betterleaks-{version}"
        exe = cache / "betterleaks"
        if exe.is_file():
            return exe
        name = f"betterleaks_{version}_{os_name}_{arch}.tar.gz"
        print(f"fetching {name} (once) ...", file=sys.stderr)
        data = _download(f"https://github.com/betterleaks/betterleaks/releases/download/v{version}/{name}")
        if not self.tarball_ok(data, os_name, arch):
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

    def group_issue(self, g: dict) -> dict:
        """The issue for one group of findings."""
        rule, at = located(g)
        ev = at["evidence"]
        if g["internal"]:
            matches = sorted(g["head_matches"] or g["matches"])  # those at HEAD, while it is there
            ev["preview"] = matches[0]
            ev["matches"] = matches[:20]
            ev["match_count"] = len(matches)
            return self.internal_issue(rule, f"INT|{hashlib.sha256(g['basis'].encode()).hexdigest()}", at)
        ev["certain"] = g["rules"][rule] in ("high", "medium")
        # IDs come from facts about the finding, never from the secret: a hash of a short password next to its
        # prefix and length could be brute-forced offline
        key = "|".join(map(str, g["first"][1:])) if g["named"] else hashlib.sha256(g["basis"].encode()).hexdigest()
        return self.credential_issue(rule, f"LEAK|{key}", at)

    def internal_issue(self, rule: str, key: str, at: dict) -> dict:
        """An internal reference, at HEAD or only in history; `at` is where it is (file, line, evidence)."""
        ev = at["evidence"]
        here = ev["at_head"]
        more = f" and {ev['match_count'] - 1} more" if ev["match_count"] > 1 else ""
        return self.issue(
            "INT", "internal-ref", rule, "internal-head" if here else "internal-history",
            f"Internal reference `{ev['preview']}`{more} ({'at HEAD' if here else 'only in history'})",
            key=key, **at,
            fix=("Replace it with a config value, CLI argument or environment variable "
                 "(or a neutral placeholder in docs)." if here else
                 "Only matters if the history is published; a fresh squashed history drops it."),
            done_when="`cleanup.py leaks` no longer reports it at HEAD.")

    def credential_issue(self, rule: str, key: str, at: dict) -> dict:
        """A certain or possible credential, at HEAD or only in history; `at` is where it is."""
        ev = at["evidence"]
        here = ev["at_head"]
        if ev["certain"]:
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
        return self.issue(
            "LEAK", "leak", rule, "credential" if ev["certain"] else "possible-credential",
            f"{'Secret' if ev['certain'] else 'Possible secret'} `{ev['preview']}` matched {rule} "
            f"({'at HEAD' if here else 'only in history'})",
            key=key, **at, fix=fix,
            done_when="Rotated (user confirms), gone from HEAD, and dismissed with a note on how history is handled.")

    def carry_over(self, found: list, known: list, history: bool) -> list:
        """`found` with the IDs of the issues `known` before: an internal reference matches by ID (its rule and
        file), a credential by a fingerprint in common (a commit that added it). After a scan of HEAD only, each
        also keeps what the history scan saw, and a known issue not found is still in the history: it comes back
        as history-only rather than fixed."""
        left = {k["id"]: k for k in known if k["status"] != "fixed"}
        out = []
        for i in found:
            k = take(left, i)
            if k and history:
                i = {**i, "id": k["id"]}
            elif k:
                i = self.seen_at_head(k, i)
            out.append(i)
        if not history:
            out += [self.seen_at_head(k, None) for k in left.values()]
        return out

    def seen_at_head(self, known: dict, now: dict | None) -> dict:
        """`known` after a scan of HEAD only: `now` is what it found of it, or None once it has left HEAD (it is
        still in the history). It keeps its ID and what the history scan saw; it becomes certain if HEAD is."""
        ev = {**known["evidence"], "at_head": False, "head_files": []}
        at = {"file": known["file"], "line": known["line"], "evidence": ev}
        rule = known["rule"]
        if now:
            seen = now["evidence"]
            ev.update({k: seen[k] for k in ("at_head", "head_files", "preview", "matches", "match_count") if k in seen})
            ev["rules"] = sorted({*ev["rules"], *seen["rules"]})
            at.update(file=now["file"], line=now["line"])
            if seen.get("certain") and not ev.get("certain"):
                ev["certain"] = True
                rule = now["rule"]
        build = self.internal_issue if known["category"] == "internal-ref" else self.credential_issue
        issue = build(rule, "", at)
        issue["id"] = known["id"]
        return issue

    def scan(self, repo: Path, internal: list, last: dict, known: list) -> Result:
        """Betterleaks over the files at HEAD, and over the full history the first time, or when the rules or the
        binary changed since `last` (the previous scan's record; {} to force it, after rewriting the history).
        Otherwise `known`, this tool's issues, says what is in the history. Its metrics record the version and the
        rules it ran, and whether it read the history."""
        exe = self.executable()
        version = run([exe, "version"]).stdout.strip() or "?"
        rules = rule_set(internal)
        rules_sha256 = hashlib.sha256(rules.encode()).hexdigest()[:12]
        history = not last.get("history") or (last.get("rules_sha256"), last.get(self.name)) != (rules_sha256, version)
        with tempfile.TemporaryDirectory(prefix="cleanup-leaks-") as tmp:
            config = Path(tmp, "rules.toml")
            config.write_text(rules)
            # Betterleaks reads ignore files from the root of the repo it scans, so it scans a bare mirror
            # (--shared copies no objects)
            mirror = Path(tmp, "scan.git")
            git(repo, "clone", "--quiet", "--mirror", "--shared", str(repo), str(mirror))
            at_head = betterleaks(exe, mirror, config, ["--log-opts", head_as_commit(repo, mirror)])
            committed = betterleaks(exe, mirror, config, []) if history else []
        for f in at_head:  # found in the stand-in commit: say HEAD
            f["head"] = (f["File"], f["StartLine"])
            f.update(Commit="HEAD", Date="", Fingerprint=f"HEAD:{f['File']}:{f['RuleID']}:{f['StartLine']}")
        if not history:
            trace(repo, at_head)
        groups = group_findings(committed, at_head)
        issues = self.carry_over([self.group_issue(g) for g in groups.values()], known, history)
        return Result(issues, {self.name: version, "rules_sha256": rules_sha256, "history": history})

# --- the binary ------------------------------------------------------------------------------------


def _platform() -> tuple:
    os_name = {"darwin": "darwin", "linux": "linux"}.get(sys.platform)
    machine = platform.machine().lower()
    arch = {"x86_64": "x64", "amd64": "x64", "arm64": "arm64", "aarch64": "arm64"}.get(machine)
    return os_name, arch


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


def _internal_rule(n: int, regex: str) -> str:
    if "'''" in regex:
        raise SystemExit("--internal patterns cannot contain '''")
    return f"\n[[rules]]\nid = \"internal-custom-{n}\"\nregex = '''({regex})'''\n"


def rule_set(internal: list) -> str:
    """betterleaks.toml, next to this file, plus one rule per --internal pattern."""
    rules = Path(__file__).with_suffix(".toml").read_text()
    return rules + "".join(_internal_rule(n, rx) for n, rx in enumerate(internal, 1))


# --- the scans -------------------------------------------------------------------------------------


def betterleaks(exe: Path, mirror: Path, config: Path, log_opts: list) -> list:
    """The findings of one run over `mirror`. It trusts neither allow comments nor ignore files: a key waved past a
    pre-commit hook is what this scan is for, and dismissals live in issues.json. Its ignore file is looked for in
    the private temp dir, which has none."""
    report = config.with_name("report.json")
    report.unlink(missing_ok=True)
    p = run([exe, "git", mirror, *log_opts, "--config", config, "--report-format", "json", "--report-path", report,
             "--exit-code", "0", "--no-banner", "--no-color", "--log-level", "error", "--ignore-gitleaks-allow",
             "--gitleaks-ignore-path", config.parent])
    if p.returncode or not report.exists():
        raise SystemExit(f"betterleaks failed (exit {p.returncode}): {p.stderr.strip()[-400:]}")
    return json.loads(report.read_text() or "[]")


def head_as_commit(repo: Path, mirror: Path) -> str:
    """A root commit in the scan's mirror that holds HEAD's tree, so a scan of it reads every tracked file at HEAD
    as added, once. No ref names it, so a scan of the history never sees it."""
    tree = git(repo, "rev-parse", "HEAD^{tree}").strip()
    who = {"GIT_AUTHOR_NAME": "revibe", "GIT_AUTHOR_EMAIL": "revibe@localhost", "GIT_COMMITTER_NAME": "revibe",
           "GIT_COMMITTER_EMAIL": "revibe@localhost"}
    p = run(["git", "-C", mirror, "commit-tree", tree, "-m", "HEAD"], env={**os.environ, **who})
    if p.returncode:
        raise SystemExit(f"git commit-tree failed: {p.stderr.strip()[-200:]}")
    return p.stdout.strip()


def trace(repo: Path, at_head: list):
    """Point each credential found at HEAD at the commit that added its line, as a scan of the history reports it,
    so the fingerprints of the two scans match. A file name, a file in an archive, or a line blame can't trace
    stays at HEAD."""
    for f in at_head:
        if f["RuleID"].startswith("internal-") or not f["StartLine"] or "!" in f["File"]:
            continue
        p = run(["git", "-C", repo, "blame", "--porcelain", "-L", f"{f['StartLine']},+1", "HEAD", "--", f["File"]])
        if p.returncode:
            continue
        header, *rest = p.stdout.splitlines()  # the last line is the line itself: never shown
        commit, line = header.split()[:2]
        meta = dict(r.split(" ", 1) for r in rest if " " in r and not r.startswith("\t"))
        when = dt.datetime.fromtimestamp(int(meta["author-time"]), dt.timezone.utc)
        f.update(Commit=commit, Date=when.strftime("%Y-%m-%dT%H:%M:%SZ"), File=meta["filename"], StartLine=int(line),
                 Fingerprint=f"{commit}:{meta['filename']}:{f['RuleID']}:{line}")


# --- findings to issues ----------------------------------------------------------------------------


def preview(secret: str) -> str:
    """How a secret is shown: its public prefix, if it has one, and its length. Nothing of the secret part."""
    # public prefixes that name a token's kind; everything after them is secret
    known = r"^(?:gh[pousr]_|github_pat_|hf_|sk-(?:proj-|ant-)?|xox[abprs]-|glpat-|npm_|pypi-|AKIA|ASIA|AIza)"
    prefix = re.match(known, secret)
    return f"{prefix[0] if prefix else ''}… ({len(secret)} chars)"


def group_findings(committed: list, at_head: list) -> dict:
    """{basis: group} of Betterleaks findings, grouped the way they get fixed: a credential wherever it appears
    (it is rotated once), an internal reference per rule and file. `committed` were found in the history, each
    in the commit that added it; `at_head` in HEAD's files, each with `head`, where it is there. With no history
    read, HEAD's findings say where they were committed. Groups hold the plaintext (`matches`); issues get only
    `preview`."""
    groups = {}
    for f in committed:
        place(group(groups, f), f)
    for f in at_head:
        g = group(groups, f)
        g["head_files"].add(f["head"][0])
        g["at_head"] = min(g["at_head"] or f["head"], f["head"])
        if g["internal"]:
            g["head_matches"].add(f["Secret"][:80])
        if not committed or not g["locs"]:  # the history scan doesn't say where it was committed
            place(g, f)
    return groups


def group(groups: dict, f: dict) -> dict:
    """The group finding `f` belongs to, created if it is the first, with the rule that found it added."""
    secret = f.get("Secret") or ""
    internal = f["RuleID"].startswith("internal-")
    basis = f"internal|{f['RuleID']}|{f['File']}" if internal else secret or f"path:{f['File']}"
    g = groups.setdefault(basis, {
        "basis": basis, "internal": internal, "named": bool(secret.strip()), "rules": {}, "first": None, "locs": [],
        "fingerprints": [], "matches": set(), "head_files": set(), "at_head": None, "head_matches": set(),
        "preview": preview(secret) if secret else "(file name)",
    })
    seen = (f.get("Attributes") or {}).get("confidence", "high")
    g["rules"][f["RuleID"]] = min(seen, g["rules"].get(f["RuleID"], seen), key=rank)  # its most confident finding
    return g


def place(g: dict, f: dict):
    """Add the commit where finding `f` was added to its group."""
    at = (f["Date"], f["Commit"], f["File"], f["StartLine"])
    g["first"] = min(g["first"] or at, at)
    g["locs"].append(f"{f['File']}:{f['StartLine']}@{f['Commit'][:8]}")
    g["fingerprints"].append(f["Fingerprint"])
    if g["internal"]:
        g["matches"].add(f["Secret"][:80])


def take(left: dict, found: dict) -> dict | None:
    """The known issue that `found` is, taken out of `left` (the known issues by ID): the one with its ID, else
    the first with a fingerprint in common."""
    prints = set(found["evidence"]["fingerprints"])
    k = left.get(found["id"]) or next(
        (k for k in left.values() if prints.intersection(k["evidence"].get("fingerprints", []))), None)
    if k:
        del left[k["id"]]
    return k


def rank(confidence: str) -> int:
    """How a rule's confidence sorts, most confident first."""
    return {"high": 0, "medium": 1, "low": 2}.get(confidence, 9)


def located(g: dict) -> tuple:
    """(rule, issue arguments) shared by both kinds of group: the most confident and specific rule, where the
    group is now (at HEAD, or where it was first committed), and the evidence. A credential keeps all its
    fingerprints, which a later scan of HEAD matches it by."""
    rule = min(g["rules"], key=lambda r: (rank(g["rules"][r]), "generic" in r, r))
    date, commit, file, line = g["first"]
    file, line = g["at_head"] or (file, line)
    return rule, {"file": file, "line": line or None, "evidence": {
        "rules": sorted(g["rules"]), "preview": g["preview"], "at_head": bool(g["head_files"]),
        "head_files": sorted(g["head_files"])[:10], "first_commit": commit[:12], "first_date": date,
        "occurrences": len(g["locs"]), "locations": g["locs"][:10],
        "fingerprints": g["fingerprints"][:10] if g["internal"] else g["fingerprints"],
    }}
