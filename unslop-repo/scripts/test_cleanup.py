#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Tests for cleanup.py: offline checks of its parsing and bookkeeping, then an end-to-end run.

    test_cleanup.py             everything; the end-to-end run needs uv and the network (a few minutes)
    test_cleanup.py --offline   only the offline checks (about a second)

The end-to-end run builds a throwaway repo with one known instance of each finding. Its fake secrets
and internal references are assembled at runtime, so the leak scan doesn't flag this file.
"""

import json
import random
import string
import subprocess
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True  # keep __pycache__ out of the skill folder
import cleanup as c

CLEANUP = Path(c.__file__).resolve()
failures = []


def check(cond, what):
    print(f"  {'ok  ' if cond else 'FAIL'} {what}")
    if not cond:
        failures.append(what)


# --- offline ---------------------------------------------------------------------------------------

# Recorded scb-check 0.2.0 output (paths shortened): the parser's contract with the pinned version.
SCB_SAMPLE = """\
trivial-wrapper[warning]: `disabled_train` adds no behavior
   ┌─ /r/ldm/models/diffusion/classifier.py:22:1
   │
21 │
22 │ def disabled_train(self, mode=True):
   │

duplicate-structure: duplicated block (17 lines, 2 instances)
    ┌─ /r/ldm/models/diffusion/classifier.py:70
    │
 70 │     def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
    │
    ┆
    ┌─ /r/ldm/models/diffusion/ddpm.py:186
    │
186 │     def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
    │

erosion: function `apply_model` exceeds complexity threshold
    ┌─ /r/ldm/models/diffusion/ddpm.py:891
    │
890 │
891 │     def apply_model(self, x_noisy, t, cond, return_ids=False):
892 │
    │
    = complexity: 25, sloc: 68 (threshold: complexity > 10)

warning[except-pass-silence]: except clause with bare pass silently swallows errors
    ┌─ /r/ldm/util.py:769:9
    │
"""


def offline():
    print("offline")
    found = c.parse_scb(SCB_SAMPLE)
    check([f["rule"] for f in found] == ["trivial-wrapper", "duplicate-structure", "erosion", "except-pass-silence"],
          "scb-check findings are parsed")
    check(found[1]["locs"] == [("/r/ldm/models/diffusion/classifier.py", 70), ("/r/ldm/models/diffusion/ddpm.py", 186)],
          "every instance of a duplicate block keeps its location")
    check(found[2]["notes"] == ["complexity: 25, sloc: 68 (threshold: complexity > 10)"],
          "a function's complexity and SLOC are parsed from its note line")

    check(c.matches("tests/test_a.py", "tests/**") and not c.matches("pkg/tests/a.py", "tests/**"),
          "'tests/**' matches only the top-level tests/")
    check(c.matches("a/b/test_x.py", "**/test_*.py") and c.matches("test_x.py", "**/test_*.py"),
          "'**/' matches any depth, including none")
    check(not c.tarball_ok(b"tampered", "darwin", "arm64"), "a tampered Betterleaks download is refused")

    def cx(symbol, severity="medium", **kw):
        return c.issue("CX", "scb-check", "complexity", "cc", severity, f"`{symbol}` is complex", file="a.py",
                       symbol=symbol, **kw)

    data = {"issues": []}
    c.merge(data, [cx("f"), cx("g")], {"scb-check"})
    get = {i["symbol"]: i for i in data["issues"]}
    check(len(data["issues"]) == 2 and cx("f", line=3)["id"] == get["f"]["id"], "IDs don't depend on line numbers")
    get["f"].update(status="dismissed", note="deliberate")
    c.merge(data, [cx("f"), cx("g", severity="low")], {"scb-check"})
    check(get["f"]["status"] == "dismissed", "an agent's dismissal survives a re-run")
    check(get["g"]["severity"] == "low", "a field the agent didn't edit takes the tool's new value")
    c.merge(data, [cx("f")], {"scb-check"})
    check(get["g"]["status"] == "fixed" and get["f"]["status"] == "dismissed",
          "an open issue that is no longer detected becomes fixed; a dismissed one stays dismissed")
    c.merge(data, [cx("f"), cx("g")], {"scb-check"})
    check(get["g"]["status"] == "open", "a fixed issue that is detected again reopens")
    c.merge(data, [], set())
    check(get["g"]["status"] == "open", "a tool that didn't run closes nothing")

    empty = c.CodeIndex(Path("/r"), [])
    issues = c.scb_issues(SCB_SAMPLE, Path("/r"), empty, high_cc=1)
    check(sorted(i["category"] for i in issues) == ["complexity", "red-flag", "red-flag"],
          "scb-check output becomes a complexity issue, a wrapper and a duplicate group")
    raised = []
    for args in ((SCB_SAMPLE, Path("/r"), empty, 2), (SCB_SAMPLE.replace("┌─", "--"), Path("/r"), empty, 1)):
        try:
            c.scb_issues(*args)
        except c.ToolError:
            raised.append(True)
    check(len(raised) == 2, "scb-check output that doesn't match its JSON totals or format is an error, not 'clean'")
    row = "pkg/a.py:5: unused function 'helper' (60% confidence, 5 lines)"
    check(c.vulture_rows(row + "\n")[0]["name"] == "helper", "vulture lines are parsed")
    try:
        c.vulture_rows(row + "\nsomething vulture never printed before\n")
        check(False, "an unrecognized vulture line is an error")
    except c.ToolError:
        check(True, "an unrecognized vulture line is an error")

    def ok(repo, files, index):
        return c.Result([cx("h")], {"m": 1})

    def nothing_to_read(repo, files, index):
        return c.Result([], {}, skipped="no lockfile")

    def broken(repo, files, index):
        raise KeyError("total_loc")

    found, metrics, not_measured, measured = c.run_steps(None, [], None, steps=(
        ("a", ok), ("b", nothing_to_read), ("c", broken)))
    check(measured == {"a"} and set(not_measured) == {"b", "c"} and metrics == {"m": 1},
          "a skipped or broken step is not measured, and the other steps still count")
    check(found[0]["source"] == "a", "issues carry the name of the step that found them")
    data = {"issues": []}
    c.merge(data, [dict(cx("v"), source="b")], {"b"})
    c.merge(data, found, measured)
    check(data["issues"][0]["status"] == "open", "a step that measured nothing doesn't close its issues")


# --- end to end ------------------------------------------------------------------------------------

def build_repo(root: Path) -> dict:
    """One known instance of each finding, in root/repo. Returns the fake secrets."""
    rnd = random.Random(1234)
    letters, alnum = string.ascii_letters, string.ascii_letters + string.digits
    secrets = {"github": "ghp" + "_" + "".join(rnd.choice(alnum) for _ in range(36)),
               "hf": "hf" + "_" + "".join(rnd.choice(letters) for _ in range(34))}
    sbatch, mila = "#SB" + "ATCH", "/net" + "work/scratch/s/student/data"  # split so this file scans clean

    repo = root / "repo"
    repo.mkdir()

    def write(rel, text):
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text)

    def sh(*cmd):
        subprocess.run(["git", "-C", str(repo), *cmd], check=True, capture_output=True)

    sh("init", "-q")
    sh("config", "user.email", "selftest@example.com")
    sh("config", "user.name", "selftest")
    write("old_key.py", f'HF_TOKEN = "{secrets["hf"]}"\n')
    sh("add", "-A")
    sh("commit", "-qm", "add a key that will only live in history")
    sh("rm", "-q", "old_key.py")
    branches = "\n".join(f"    {'if' if n == 0 else 'elif'} kind == 'k{n}':\n        return x + {n}" for n in range(12))
    legacy = "\n".join(f"    {'if' if n == 0 else 'elif'} mode == 'm{n}':\n        y = [v * {n} for v in y]" for n in range(12))
    body = "\n".join(f"        row = row * {n + 2} + {n}" for n in range(10))
    write("README.md", "# demo\n")
    write("requirements.txt", "requests==2.19.1\n")
    write("pkg/__init__.py", "")
    write("pkg/model.py", f"""import numpy as np


class ConfigOnlyModel:
    def __init__(self, width):
        self.width = width

    def forward(self, x):
        return x * self.width


def unused_helper(values):
    total = 0
    for v in values:
        total += v
    return total / max(len(values), 1)


def route(kind, x):
{branches}
    return np.zeros(1)


def legacy_route(mode, y):
{legacy}
    return y


def normalize_a(rows):
    out = []
    for row in rows:
{body}
        out.append(row)
    return out


def normalize_b(rows):
    out = []
    for row in rows:
{body}
        out.append(row)
    return out
""")
    write("pkg/vendored.py", "# Copyright 2020 Other Org\n# Licensed under the Apache License, Version 2.0\n\n"
          "def borrowed(x):\n    return x\n")
    write("pkg/main.py", "from pkg.model import route, normalize_a, normalize_b\nfrom pkg.shared import dispatch\n"
          "from pkg.vendored import borrowed\n\n\n"
          "def main():\n    print(route('k1', 1), normalize_a([1]), normalize_b([2]), borrowed(3), dispatch('k2', 2))\n\n\n"
          "if __name__ == '__main__':\n    main()\n")
    (root / "shared").mkdir()  # a tracked symlink to a module outside the repo
    (root / "shared/dispatch.py").write_text(f"def dispatch(kind, x):\n{branches}\n    return x\n")
    (repo / "pkg/shared.py").symlink_to("../../shared/dispatch.py")
    write("configs/model.yaml", "model:\n  target: pkg.model.ConfigOnlyModel\n  params:\n    width: 4\n")
    write("config.py", f'GITHUB_TOKEN = "{secrets["github"]}"\n')
    write("train.sh", f"#!/bin/bash\n{sbatch} --account=rrg-someprof\npython -m pkg.main --data {mila}\n")
    (repo / "data").mkdir()
    (repo / "data/blob.bin").write_bytes(rnd.randbytes(6_000_000))
    sh("add", "-A")
    sh("commit", "-qm", "demo repo")
    return secrets


def end_to_end():
    print("end to end")
    transcript = []

    def cli(*args):
        p = subprocess.run([sys.executable, str(CLEANUP), *args], capture_output=True, text=True)
        transcript.append(p.stdout + p.stderr)
        return p

    with tempfile.TemporaryDirectory(prefix="cleanup-test-") as tmp:
        secrets = build_repo(Path(tmp))
        repo = Path(tmp, "repo")
        r = str(repo)
        check(cli("measure", r).returncode != 0, "measure refuses to run before the leak scan")
        p = cli("leaks", r)
        check(p.returncode == 0, f"leaks runs ({p.stderr.strip()[-200:]})")
        p = cli("measure", r, "--label", "baseline")
        check(p.returncode == 0, f"measure runs ({p.stderr.strip()[-200:]})")
        data = json.loads((repo / ".cleanup/issues.json").read_text())
        by = {}
        for i in data["issues"]:
            by.setdefault((i["category"], i["rule"]), []).append(i)

        def one(category, rule, **want):
            return any(all(i.get(k) == v or i.get("evidence", {}).get(k) == v for k, v in want.items())
                       for i in by.get((category, rule), []))

        check(one("leak", "github-pat", severity="critical", at_head=True), "secret at HEAD -> critical")
        check(one("leak", "huggingface-access-token", severity="critical", at_head=False),
              "secret only in history -> critical")
        check(one("internal-ref", "internal-slurm-account", severity="high"), "SLURM account -> high")
        check(one("internal-ref", "internal-mila-path", severity="high"), "Mila path -> high")
        check(one("large-file", "large-file", severity="high", file="data/blob.bin"), "6 MB file -> high")
        check(one("license", "missing-license", severity="high"), "missing LICENSE -> high")
        check(one("license", "third-party-license", severity="high"), "vendored Apache header -> high")
        check(one("dead-code", "function", symbol="unused_helper", status="open"), "dead function -> open")
        check(one("dead-code", "class", symbol="ConfigOnlyModel", status="dismissed"),
              "class named only in yaml `target:` -> dismissed")
        check(one("complexity", "cc", symbol="route", source="scb-check"), "CC > 10 -> complexity issue")
        check(one("complexity", "cc", symbol="dispatch", file="pkg/shared.py"),
              "a tracked symlink to a module outside the repo is measured")
        check(one("complexity", "cc", symbol="legacy_route", severity="low")
              and one("dead-code", "function", symbol="legacy_route", severity="high"),
              "complex dead function -> delete it (high), don't refactor it (low)")
        check(one("red-flag", "repetition"), "duplicated block -> repetition issue")
        check(one("readiness", "DEP001", symbol="numpy", severity="high"), "undeclared numpy -> high")
        if "pip-audit" in json.loads((repo / ".cleanup/snapshots/baseline.json").read_text())["not_measured"]:
            print("  skip pip-audit check (offline?)")
        else:
            check(one("vulnerable-dependency", "pip-audit", symbol="requests", severity="high"),
                  "vulnerable pin -> high")

        route = next(i for i in data["issues"] if i["symbol"] == "route")
        route.update(status="dismissed", note="test edit")
        data["issues"].append({**route, "id": "CX-0123abcd", "source": "ruff", "status": "open", "note": ""})
        data["issues"].append({**route, "id": "RF-A1", "source": "agent", "category": "red-flag",
                               "rule": "shallow-module", "status": "open", "note": ""})
        ids = {i["id"] for i in data["issues"]}
        (repo / ".cleanup/issues.json").write_text(json.dumps(data))
        flags = json.loads((repo / ".cleanup/snapshots/baseline.json").read_text())["metrics"]["red_flags_open"]
        cli("report", r)
        check(json.loads((repo / ".cleanup/snapshots/baseline.json").read_text())["metrics"]["red_flags_open"]
              == flags + 1, "report counts red flags added in review into the snapshot taken at HEAD")
        loc = json.loads((repo / ".cleanup/snapshots/baseline.json").read_text())["metrics"]["loc"]
        (repo / "scratch.py").write_text("def scratch(x):\n" + "    x += 1\n" * 40 + "    return x\n")  # untracked
        cli("measure", r, "--label", "baseline")
        (repo / "scratch.py").unlink()
        check(json.loads((repo / ".cleanup/snapshots/baseline.json").read_text())["metrics"]["loc"] == loc,
              "untracked files are not measured")
        data = json.loads((repo / ".cleanup/issues.json").read_text())
        check({i["id"] for i in data["issues"]} == ids, "IDs are stable across runs")
        check(next(i for i in data["issues"] if i["id"] == route["id"])["status"] == "dismissed",
              "agent edits survive a re-run")
        check(next(i for i in data["issues"] if i["id"] == "CX-0123abcd")["status"] == "dismissed",
              "open issues from a retired tool are dismissed")

        model = (repo / "pkg/model.py").read_text()
        start, end = model.index("def unused_helper"), model.index("def route")
        (repo / "pkg/model.py").write_text(model[:start] + model[end:])
        subprocess.run(["git", "-C", r, "commit", "-qam", "delete dead code"], check=True)
        check(cli("measure", r, "--label", "after").returncode != 0, "a new commit needs a new leak scan")
        cli("leaks", r)
        cli("measure", r, "--label", "after")
        data = json.loads((repo / ".cleanup/issues.json").read_text())
        check(any(i["symbol"] == "unused_helper" and i["status"] == "fixed" for i in data["issues"]),
              "deleted dead code -> fixed")
        p = cli("compare", r, "baseline")
        check(p.returncode == 0 and "Dead code LOC" in p.stdout, "compare prints before/after")
        p = cli("report", r)
        report = (repo / ".cleanup/report.md").read_text()
        check(p.returncode == 0 and "**BLOCKED**" in report, "report renders and says BLOCKED")
        check((repo / ".cleanup/cleanup.py").exists(), "report copies cleanup.py for handoff")
        files = [f.read_text(errors="replace") for f in (repo / ".cleanup").rglob("*") if f.is_file()]
        leaked = [k for k, v in secrets.items() if any(v in t for t in files + transcript)]
        check(not leaked, f"no plaintext secret in output or .cleanup/ {leaked or ''}")


def main():
    offline()
    if "--offline" not in sys.argv[1:]:
        end_to_end()
    print("tests:", f"FAILED {len(failures)}" if failures else "passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
