"""Everything under REPO/.cleanup/: issues.json (load, check, merge, save) and the snapshots."""

import hashlib
import json
from pathlib import Path

from .common import AGENT_FIELDS, CATEGORIES, ISSUE_DEFAULTS, POLICY, SEVERITIES, STATUSES, die, head, now


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
    problems = []
    seen = set()
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
        old = by_id.get(ident)
        prev = defaults.get(ident)
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


# --- snapshots ---------------------------------------------------------------------------------


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
