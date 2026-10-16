"""REPO/.cleanup/issues.json, the single source of truth: the issues, the snapshots and the leak scan. Every
command opens it as a `Cleanup`; report.md is generated from it."""

import datetime as dt
import hashlib
import json
from enum import IntEnum
from pathlib import Path

from tools.base import Metric
from utils.git import head, repo_root


class Severity(IntEnum):
    """How bad an issue is: a higher value is more severe. Members are named as issues.json spells them, most
    severe first. Each tool's `TOOL` policy sets the severity of its issues."""

    critical = 4
    high = 3
    medium = 2
    low = 1


def now() -> str:
    """The current time as issues.json records it: UTC, to the second."""
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


# --- issues.json -----------------------------------------------------------------------------------


class Cleanup:
    """A repo's .cleanup/ folder and its issues.json, loaded and checked."""

    # the scorecard rows count_issues sets, after the measure steps' own
    metrics = (Metric("red_flags_open", "Red flags open"), Metric("leaks_open", "Leaks open"))

    def __init__(self, path: str):
        self.repo = repo_root(path)
        self.dir = self.repo / ".cleanup"
        self.dir.mkdir(exist_ok=True)
        if not (self.dir / ".gitignore").exists():
            (self.dir / ".gitignore").write_text("*\n")
        self.path = self.dir / "issues.json"
        self.data = load_issues(self.path)

    def open_issues(self, *severities: Severity) -> list:
        """The open issues, of the given severities if any are given."""
        return [
            i for i in self.data["issues"]
            if i["status"] == "open" and (not severities or Severity[i["severity"]] in severities)
        ]

    def leak_scan_current(self) -> bool:
        """Whether the leak scan ran on HEAD."""
        return self.data.get("leak_scan", {}).get("commit") == head(self.repo)

    def require_leak_scan(self):
        """Stop the command unless the leak scan ran on HEAD."""
        if not self.leak_scan_current():
            raise SystemExit("the leak scan is mandatory: run `cleanup.py leaks <repo>` on the current commit first")

    def release_status(self) -> str:
        """Blocked while any critical or high issue is open, or while the leak scan is older than HEAD."""
        blocked = not self.leak_scan_current() or self.open_issues(Severity.critical, Severity.high)
        return "blocked" if blocked else "clear"

    def snapshot(self, label: str) -> dict:
        """The snapshot `measure --label` recorded; stops the command if there is none."""
        snaps = self.data["snapshots"]
        if label not in snaps:
            raise SystemExit(f"no snapshot '{label}' (have: {', '.join(sorted(snaps)) or 'none'})")
        return snaps[label]

    def latest_snapshot(self, exclude: str = "") -> dict | None:
        """The most recent snapshot, other than `exclude`."""
        snaps = [s for label, s in self.data["snapshots"].items() if label != exclude]
        return max(snaps, key=lambda s: s["date"]) if snaps else None

    def count_issues(self, snap: dict):
        """Set the snapshot fields that come from the issues rather than from the tools."""
        still_open = self.open_issues()
        snap["metrics"]["red_flags_open"] = sum(i["category"] == "red-flag" for i in still_open)
        snap["metrics"]["leaks_open"] = sum(i["category"] == "leak" for i in still_open)
        snap["open_issues"] = {s.name: sum(i["severity"] == s.name for i in still_open) for s in Severity}
        snap["release_status"] = self.release_status()

    def save(self):
        """Write issues.json as valid JSON with one issue and one snapshot per line, so `grep <ID>` shows a whole
        issue. Review adds and closes issues after `measure`, so the snapshot taken at HEAD is recounted first:
        the baseline then includes the red flags found in review. Snapshots of older commits keep their counts."""
        data = self.data
        data["issues"].sort(key=lambda i: (
            i["status"] != "open", -Severity[i["severity"]], i["category"],
            -(i.get("evidence") or {}).get("weight", (i.get("evidence") or {}).get("lines", 0)), i["id"]))
        data["commit"] = head(self.repo)
        data["generated"] = now()
        data["release_status"] = self.release_status()
        latest = self.latest_snapshot()
        if latest and latest["commit"] == data["commit"]:
            self.count_issues(latest)
        meta = [f"  {json.dumps(k)}: {json.dumps(v, ensure_ascii=False)}," for k, v in data.items()
                if k not in ("issues", "snapshots", "defaults")]
        issues = ",\n".join("    " + json.dumps(i, ensure_ascii=False) for i in data["issues"])
        snaps = ",\n".join(f"    {json.dumps(label)}: {json.dumps(s, ensure_ascii=False)}"
                           for label, s in data["snapshots"].items())
        defaults = json.dumps(data.get("defaults", {}), separators=(",", ":"))
        self.path.write_text("{\n" + "\n".join(meta) + f'\n  "issues": [\n{issues}\n  ],\n'
                             f'  "snapshots": {{\n{snaps}\n  }},\n  "defaults": {defaults}\n}}\n')


def load_issues(path: Path) -> dict:
    """issues.json, checked, or an empty one if there is none yet."""
    if not path.exists():
        return {"schema": 1, "repo": path.parent.parent.name, "tools": {}, "issues": [], "snapshots": {}}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise SystemExit(f"{path} is not valid JSON (line {e.lineno}: {e.msg}). Fix it by hand; .cleanup/ is not in "
            "git, so there is no older copy")
    if not isinstance(data, dict) or not isinstance(data.setdefault("issues", []), list) \
            or not isinstance(data.setdefault("snapshots", {}), dict):
        raise SystemExit(f"{path} must be an object with an \"issues\" list and a \"snapshots\" object")
    data.setdefault("tools", {})
    check_issues(data["issues"], path)
    return data


def check_issues(issues: list, path: Path):
    """issues.json is edited by hand, so a typo must stop the command rather than drop the issue from the
    report and the release gate. Missing optional fields get their defaults."""
    defaults = {"source": "agent", "rule": "", "file": "", "line": None, "end_line": None, "symbol": "",
                "evidence": {}, "fix": "", "done_when": "", "effort": "S", "status": "open", "note": ""}
    statuses = ("open", "fixed", "dismissed")
    categories = ("leak", "internal-ref", "large-file", "license", "vulnerable-dependency", "readiness",
                  "dead-code", "complexity", "red-flag", "file-length", "globals", "bug")
    problems = []
    seen = set()
    for n, i in enumerate(issues, 1):
        ident = i.get("id") or f"issue #{n}"
        if missing := [k for k in ("id", "category", "severity", "summary") if not i.get(k)]:
            problems.append(f"{ident}: missing {', '.join(missing)}")
            continue
        for k, v in defaults.items():
            i.setdefault(k, dict(v) if isinstance(v, dict) else v)
        for k, allowed in (("severity", [s.name for s in Severity]), ("status", statuses), ("category", categories)):
            if i[k] not in allowed:
                problems.append(f'{ident}: {k} "{i[k]}" is not one of {", ".join(allowed)}')
        if i["status"] == "dismissed" and not i["note"]:
            problems.append(f"{ident}: dismissed without a note saying why")
        if ident in seen:
            problems.append(f"{ident}: duplicate id")
        seen.add(ident)
    if problems:
        raise SystemExit(f"{path} needs fixing by hand (`grep <ID>` finds each issue):\n  " + "\n  ".join(problems[:20]))


def _digest(value) -> str:
    return hashlib.sha1(json.dumps(value).encode()).hexdigest()[:10]


def merge(data: dict, found: list, sources: set):
    """Fold fresh findings into issues.json.

    A field an agent edited (status, severity, fix, ...) survives; one still holding the value the tool
    last wrote takes the new value. Open issues from `sources` that are no longer detected become fixed.
    A fixed issue that is detected again reopens, whoever marked it fixed: a fix the tool can still see
    is not one.
    """
    agent_fields = ("status", "severity", "fix", "done_when", "note", "effort")
    defaults = data.setdefault("defaults", {})  # per issue: hashes of the values the tool last wrote
    by_id = {i["id"]: i for i in data["issues"]}
    seen = set()
    for new in found:
        ident = new["id"]
        if ident in seen:
            continue
        seen.add(ident)
        old = by_id.get(ident)
        prev = defaults.get(ident)
        defaults[ident] = {k: _digest(new[k]) for k in agent_fields}
        if old is None:
            data["issues"].append(new)
            by_id[ident] = new
            continue
        edited = {k: old[k] for k in agent_fields if k in old and (prev is None or _digest(old[k]) != prev.get(k))}
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
            i["status"] = "fixed"
            i["note"] = "no longer detected"
            defaults.setdefault(i["id"], {}).update(status=_digest("fixed"), note=_digest("no longer detected"))
