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

import ast
import hashlib
import json
import random
import re
import string
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True  # keep __pycache__ out of the skill folder
import measure
import scope
import state
import tools
from tools import betterleaks, deptry, scb_check, vulture
from tools.base import Result, Step, Tool

CLEANUP = Path(__file__).resolve().parent / "cleanup.py"
failures = []


def check(cond, what):
    print(f"  {'ok  ' if cond else 'FAIL'} {what}")
    if not cond:
        failures.append(what)


# --- offline ---------------------------------------------------------------------------------------

def offline():
    print("offline")
    check(scope.matches("tests/test_a.py", "tests/**") and not scope.matches("pkg/tests/a.py", "tests/**"),
          "'tests/**' matches only the top-level tests/")
    check(scope.matches("a/b/test_x.py", "**/test_*.py") and scope.matches("test_x.py", "**/test_*.py"),
          "'**/' matches any depth, including none")
    check(not betterleaks.tarball_ok(b"tampered", "darwin", "arm64"), "a tampered Betterleaks download is refused")

    rows = tools.policies()
    check(all(sev in state.Severity.__members__ and when for sev, when in rows.values()), "every policy row has a severity")
    check(len(rows) == sum(len(m.TOOL.policy) for m in tools.ALL), "no policy key is set by two tools")
    check(all(isinstance(m, Step) for m in tools.MEASURE), "every measure step has a TOOL and a measure function")
    pinned = set(re.findall(r'"([\w.-]+==[\w.]+)"', CLEANUP.read_text().split("# ///")[1]))
    check(pinned == {f"{m.TOOL.package}=={m.TOOL.version}" for m in tools.ANALYZERS},
          "cleanup.py's header pins exactly the analyzers, at the versions their TOOL names")

    def cx(symbol, policy="complex", **kw):
        return scb_check.TOOL.issue("CX", "complexity", "cc", policy, f"`{symbol}` is complex", file="a.py",
                                    symbol=symbol, **kw)

    data = {"issues": []}
    state.merge(data, [cx("f"), cx("g")], {"scb-check"})
    get = {i["symbol"]: i for i in data["issues"]}
    check(len(data["issues"]) == 2 and cx("f", line=3)["id"] == get["f"]["id"], "IDs don't depend on line numbers")
    get["f"].update(status="dismissed", note="deliberate")
    state.merge(data, [cx("f"), cx("g", policy="complex-small")], {"scb-check"})
    check(get["f"]["status"] == "dismissed", "an agent's dismissal survives a re-run")
    check(get["g"]["severity"] == "low", "a field the agent didn't edit takes the tool's new value")
    state.merge(data, [cx("f")], {"scb-check"})
    check(get["g"]["status"] == "fixed" and get["f"]["status"] == "dismissed",
          "an open issue that is no longer detected becomes fixed; a dismissed one stays dismissed")
    state.merge(data, [cx("f"), cx("g")], {"scb-check"})
    check(get["g"]["status"] == "open", "a fixed issue that is detected again reopens")
    state.merge(data, [], set())
    check(get["g"]["status"] == "open", "a tool that didn't run closes nothing")
    get["g"].update(status="fixed", note="refactored")
    state.merge(data, [cx("f"), cx("g")], {"scb-check"})
    check(get["g"]["status"] == "open" and "refactored" in get["g"]["note"],
          "an issue an agent marked fixed reopens while the tool still detects it, keeping the agent's note")

    step_a = Tool("a", {"x": ("low", "a test row")})

    def ok(scope):
        return Result([step_a.issue("T", "bug", "x", "x", "found by a")], {"m": 1})

    def nothing_to_read(scope):
        return Result([], {}, skipped="no lockfile")

    def broken(scope):
        raise KeyError("total_loc")

    trainer = ["trainer", "pkg"]  # pkg/trainer.py: the module, then its packages
    cases = [  # (file, text, strings_only), name, modules, named?, mentioned?
        (("tests/test_a.py", "net.train()\n", False), "train", trainer, False, True),
        (("data.py", 'load(split="train")\n', True), "train", trainer, False, True),
        (("configs/a.yaml", "target: pkg.trainer.train\n", False), "train", trainer, True, False),
        (("tests/test_b.py", "from pkg.trainer import (\n    Trainer,\n    train,\n)\n", False), "train", trainer,
         True, False),
        (("pyproject.toml", 'fit = "pkg.trainer:train"\n', False), "train", trainer, True, False),
        (("main.py", 'cfg = {"target": "main.SetupCallback"}\n', True), "SetupCallback", ["main"], True, False),
        (("configs/kl.yaml", "target: ldm.modules.losses.LPIPS\n", False), "LPIPS",
         ["contperceptual", "losses", "modules", "ldm"], True, False),  # re-exported by losses/__init__.py
        (("utils.py", "# matlab 'imresize' function\n", True), "imresize", ["utils"], False, True),
    ]
    results = [vulture._named_in([ref], name, module) for ref, name, module, _, _ in cases]
    check(all((bool(named), bool(mentioned)) == (want_named, want_mention)
              for (named, mentioned), (_, _, _, want_named, want_mention) in zip(results, cases)),
          "only module-qualified references dismiss dead code; bare `.train` or \"train\" is a mention")

    with tempfile.TemporaryDirectory() as tmp:
        none = deptry.measure(scope.Scope(Path(tmp), []))
        Path(tmp, "tool.py").write_text("# /// script\n# dependencies = []\n# ///\n")
        inline = deptry.measure(scope.Scope(Path(tmp), ["tool.py"]))
        check(none.issues[0]["severity"] == "high" and inline.issues[0]["severity"] == "low" and none.skipped,
              "no dependency list at all is high; PEP 723 blocks deptry can't read are low")

    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "mod.py").write_text("import x\n\n\ndef f(a):\n    return a\n\n\nclass C:\n    X = 1\n\n"
                                       "    def m(self):\n        return 2\n")
        index = scope.Scope(Path(tmp), ["mod.py"])
        check(vulture.runtime_lines(index, "mod.py", 4) == {5} and vulture.runtime_lines(index, "mod.py", 8) == {12},
              "coverage counts function and method bodies, not def or class-body lines that run on import")

    module = ("A = 1\nx: int = 2\n__all__ = []\nif True:\n    B = 3\nfor i in range(3):\n    pass\n"
              "[y for y in range(2)]\ndef f():\n    global G\n    G = 1\n    local = 2\nclass C:\n    attr = 1\n")
    check(scope.module_globals(ast.parse(module)) == ["A", "B", "G", "i", "x"],
          "globals are the names bound at module level or declared `global`, not locals, class attributes or dunders")

    with tempfile.TemporaryDirectory() as tmp:
        def load(text):
            (Path(tmp) / "issues.json").write_text(text)
            try:
                return state.load_issues(Path(tmp) / "issues.json")
            except SystemExit as e:
                return str(e)

        bug = {"id": "BUG-A1", "category": "bug", "severity": "high", "summary": "found while reading"}
        data = load(json.dumps({"issues": [bug]}))
        check(isinstance(data, dict) and data["issues"][0]["status"] == "open" and data["issues"][0]["evidence"] == {},
              "an agent issue with only id, category, severity and summary gets the other fields")
        error = load(json.dumps({"issues": [{**bug, "severity": "Critical"}, {**bug, "id": "BUG-A2", "status": "done"},
                                            {**bug, "id": "BUG-A3", "status": "dismissed"}]}))
        check(isinstance(error, str) and all(f"BUG-A{n}" in error for n in (1, 2, 3)),
              "a misspelled severity or status, or a dismissal without a note, stops the command and names the issue")
        error = load('{"issues": [}')
        check(isinstance(error, str) and "not valid JSON (line 1" in error, "broken JSON stops the command with its line")

    found, metrics, not_measured, measured = measure.run_steps(None, steps=(
        SimpleNamespace(TOOL=step_a, measure=ok),
        SimpleNamespace(TOOL=Tool("b", {}), measure=nothing_to_read),
        SimpleNamespace(TOOL=Tool("c", {}), measure=broken)))
    check(measured == {"a"} and set(not_measured) == {"b", "c"} and metrics == {"m": 1},
          "a skipped or broken step is not measured, and the other steps still count")
    check(found[0]["source"] == "a", "issues carry the name of the step that found them")
    data = {"issues": []}
    state.merge(data, [dict(cx("v"), source="b")], {"b"})
    state.merge(data, found, measured)
    check(data["issues"][0]["status"] == "open", "a step that measured nothing doesn't close its issues")


# --- end to end ------------------------------------------------------------------------------------

def build_repo(root: Path) -> dict:
    """One known instance of each finding, in root/repo. Returns the fake secrets."""
    rnd = random.Random(1234)
    letters = string.ascii_letters
    alnum = string.ascii_letters + string.digits
    secrets = {"github": "ghp" + "_" + "".join(rnd.choice(alnum) for _ in range(36)),
               "hf": "hf" + "_" + "".join(rnd.choice(letters) for _ in range(34)),
               "allowed": "ghp" + "_" + "".join(rnd.choice(alnum) for _ in range(36))}
    # split so this file scans clean
    sbatch = "#SB" + "ATCH"
    account = "rrg" + "-someprof"
    mila = "/net" + "work/scratch/s/student/data"

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
    write("pkg/vendored.py", "def borrowed(x):\n    return x\n")
    write("pkg/main.py", "from pkg.model import route, normalize_a, normalize_b\nfrom pkg.shared import dispatch\n"
          "from pkg.vendored import borrowed\n\n\n"
          "def main():\n    print(route('k1', 1), normalize_a([1]), normalize_b([2]), borrowed(3), dispatch('k2', 2))\n\n\n"
          "if __name__ == '__main__':\n    main()\n")
    (root / "shared").mkdir()  # a tracked symlink to a module outside the repo
    (root / "shared/dispatch.py").write_text(f"def dispatch(kind, x):\n{branches}\n    return x\n")
    (repo / "pkg/shared.py").symlink_to("../../shared/dispatch.py")
    write("configs/model.yaml", "model:\n  target: pkg.model.ConfigOnlyModel\n  params:\n    width: 4\n")
    write("pkg/constants.py", "".join(f"C{n} = {n}\n" for n in range(1001)))
    write("pkg/trainer.py", "def train(model, steps):\n    for _ in range(steps):\n        model.step()\n"
          "    model.save()\n    return model\n")
    write("tests/test_model.py", "def test_mode(model):\n    model.train()\n")
    write("config.py", f'GITHUB_TOKEN = "{secrets["github"]}"\n')
    write("settings.py", f'TOKEN = "{secrets["allowed"]}"  # gitleaks' + ':allow\n')
    write("train.sh", f"#!/bin/bash\n{sbatch} --account={account}\npython -m pkg.main --data {mila}\n")
    (repo / "data").mkdir()
    (repo / "data/blob.bin").write_bytes(rnd.randbytes(6_000_000))
    sh("add", "-A")
    sh("commit", "-qm", "demo repo")
    (repo / "scripts").mkdir()
    sh("mv", "train.sh", "scripts/train.sh")
    sh("mv", "data/blob.bin", "data/weights.bin")
    sh("commit", "-qm", "move files")
    return secrets


def end_to_end():
    print("end to end")
    transcript = []

    def cli(*args):
        p = subprocess.run(["uv", "run", "--quiet", "--script", str(CLEANUP), *args], capture_output=True, text=True)
        transcript.append(p.stdout + p.stderr)
        return p

    with tempfile.TemporaryDirectory(prefix="cleanup-test-") as tmp:
        secrets = build_repo(Path(tmp))
        repo = Path(tmp, "repo")
        r = str(repo)

        def baseline() -> dict:
            return json.loads((repo / ".cleanup/issues.json").read_text())["snapshots"]["baseline"]

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
        check(one("leak", "github-pat", file="settings.py"), "a `gitleaks:allow` comment doesn't hide a secret")
        previews = {i["evidence"]["preview"] for i in data["issues"] if i["category"] == "leak"}
        check(previews == {"ghp_… (40 chars)", "hf_… (37 chars)"}, f"previews show only the public prefix {previews}")
        hashed = {"LEAK-" + hashlib.sha1(f"LEAK|{hashlib.sha256(v.encode()).hexdigest()}".encode()).hexdigest()[:8]
                  for v in secrets.values()}
        check(not hashed & {i["id"] for i in data["issues"]}, "leak IDs don't derive from the secret")
        check(one("internal-ref", "internal-slurm-account", severity="high", file="scripts/train.sh"),
              "SLURM account, in a file moved since -> high, at its path at HEAD")
        check(one("internal-ref", "internal-mila-path", severity="high"), "Mila path -> high")
        check(one("large-file", "large-file", severity="high", file="data/weights.bin"),
              "6 MB file at HEAD -> high, at its path there")
        check(one("license", "missing-license", severity="high"), "missing LICENSE -> high")
        check(one("dead-code", "function", symbol="unused_helper", status="open"), "dead function -> open")
        check(one("dead-code", "class", symbol="ConfigOnlyModel", status="dismissed"),
              "class named only in yaml `target:` -> dismissed")
        check(one("dead-code", "function", symbol="train", status="open"),
              "dead `train()` stays open although a test calls `model.train()`")
        check(one("complexity", "cc", symbol="route", source="scb-check"), "CC > 10 -> complexity issue")
        check(one("complexity", "cc", symbol="dispatch", file="pkg/shared.py"),
              "a tracked symlink to a module outside the repo is measured")
        check(one("complexity", "cc", symbol="legacy_route", severity="low")
              and one("dead-code", "function", symbol="legacy_route", severity="high"),
              "complex dead function -> delete it (high), don't refactor it (low)")
        check(one("red-flag", "repetition"), "duplicated block -> repetition issue")
        check(one("readiness", "DEP001", symbol="numpy", severity="high"), "undeclared numpy -> high")
        metrics = baseline()["metrics"]
        check(one("file-length", "long-file", file="pkg/constants.py", severity="medium")
              and metrics["files_over_1000"] == 1 and metrics["files_over_500"] == 1 and metrics["avg_file_lines"] > 0,
              "a 1001-line module -> medium 'split it' issue, counted with the average file length")
        check(one("globals", "module-globals", severity="low") and metrics["globals"] >= 1001,
              "module-level globals -> one low issue, and a scorecard count")
        if "pip-audit" in baseline()["not_measured"]:
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
        flags = baseline()["metrics"]["red_flags_open"]
        cli("report", r)
        check(baseline()["metrics"]["red_flags_open"]
              == flags + 1, "report counts red flags added in review into the snapshot taken at HEAD")
        loc = baseline()["metrics"]["loc"]
        (repo / "scratch.py").write_text("def scratch(x):\n" + "    x += 1\n" * 40 + "    return x\n")  # untracked
        cli("measure", r, "--label", "baseline")
        (repo / "scratch.py").unlink()
        check(baseline()["metrics"]["loc"] == loc,
              "untracked files are not measured")
        data = json.loads((repo / ".cleanup/issues.json").read_text())
        check({i["id"] for i in data["issues"]} == ids, "IDs are stable across runs")
        check(next(i for i in data["issues"] if i["id"] == route["id"])["status"] == "dismissed",
              "agent edits survive a re-run")
        check(next(i for i in data["issues"] if i["id"] == "CX-0123abcd")["status"] == "dismissed",
              "open issues from a retired tool are dismissed")

        model = (repo / "pkg/model.py").read_text()
        start = model.index("def unused_helper")
        end = model.index("def route")
        (repo / "pkg/model.py").write_text(model[:start] + model[end:])
        leak = next(i for i in data["issues"] if i["category"] == "leak" and i["file"] == "config.py")
        (repo / ".betterleaksignore").write_text("\n".join(leak["evidence"]["fingerprints"]) + "\n")
        subprocess.run(["git", "-C", r, "add", ".betterleaksignore"], check=True)
        subprocess.run(["git", "-C", r, "commit", "-qam", "delete dead code"], check=True)
        check(cli("measure", r, "--label", "after").returncode != 0, "a new commit needs a new leak scan")
        cli("leaks", r)
        cli("measure", r, "--label", "after")
        data = json.loads((repo / ".cleanup/issues.json").read_text())
        check(any(i["symbol"] == "unused_helper" and i["status"] == "fixed" for i in data["issues"]),
              "deleted dead code -> fixed")
        check(next(i for i in data["issues"] if i["id"] == leak["id"])["status"] == "open",
              "the repo's own .betterleaksignore doesn't hide a secret from the audit")
        p = cli("compare", r, "baseline")
        check(p.returncode == 0 and "Dead code LOC" in p.stdout, "compare prints before/after")
        p = cli("report", r)
        report = (repo / ".cleanup/report.md").read_text()
        check(p.returncode == 0 and "**BLOCKED**" in report, "report renders and says BLOCKED")
        check("## Severity rules" in report and "- **critical:** credential" in report,
              "report ends with the severity rules")
        p = subprocess.run([sys.executable, str(repo / ".cleanup/cleanup.py"), "compare", r, "baseline"],
                           capture_output=True, text=True)
        check(p.returncode == 0 and "Dead code LOC" in p.stdout, "the copy report leaves in .cleanup/ runs on its own")
        check(not any(f.name == "__pycache__" for f in (repo / ".cleanup").rglob("*")) and
              not any(CLEANUP.parent.rglob("__pycache__")), "no __pycache__ in .cleanup/ or the skill folder")
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
