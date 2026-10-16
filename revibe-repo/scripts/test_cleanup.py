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

sys.dont_write_bytecode = True  # keep __pycache__ out of the skill folder
import measure
import scope
import state
import tools
from tools import vulture
from tools.base import Result, Step

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
    check(scope.matches("src/tests/fake_db.py", "**/tests/**"), "'**/tests/**' matches a nested tests/ folder")
    check(scope.matches("a/b/test_x.py", "**/test_*.py") and scope.matches("test_x.py", "**/test_*.py"),
          "'**/' matches any depth, including none")
    created = {tool.name: tool for tool in tools.ALL}
    check(not created["betterleaks"].tarball_ok(b"tampered", "darwin", "arm64"), "a tampered Betterleaks download is refused")

    rows = tools.policies()
    check(all(sev in state.Severity.__members__ and when for sev, when in rows.values()), "every policy row has a severity")
    check(len(rows) == sum(len(t.policy) for t in tools.ALL), "no policy key is set by two tools")
    check(all(isinstance(t, Step) for t in tools.MEASURE), "every measure step is a Step")
    pinned = set(re.findall(r'"([\w.-]+==[\w.]+)"', CLEANUP.read_text().split("# ///")[1]))
    check(pinned == {f"{t.package}=={t.version}" for t in tools.ANALYZERS},
          "cleanup.py's header pins exactly the analyzers, at the versions their classes name")
    try:
        vulture.Vulture(severity={"dead-large": "high"})
        refused = False
    except ValueError:
        refused = True
    check(refused, "a tool created with severities that don't match its policy is refused")

    def cx(symbol, policy="complex", **kw):
        return created["scb-check"].issue("CX", "complexity", "cc", policy, f"`{symbol}` is complex", file="a.py",
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

    class Ok(Step):
        name = "a"
        policy = {"x": "a test row"}

        def measure(self, scope):
            return Result([self.issue("T", "bug", "x", "x", "found by a")], {"m": 1})

    class NothingToRead(Step):
        name = "b"

        def measure(self, scope):
            return Result([], {}, skipped="no lockfile")

    class Broken(Step):
        name = "c"

        def measure(self, scope):
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
        none = created["deptry"].measure(scope.Scope(Path(tmp), []))
        Path(tmp, "tool.py").write_text("# /// script\n# dependencies = []\n# ///\n")
        inline = created["deptry"].measure(scope.Scope(Path(tmp), ["tool.py"]))
        Path(tmp, "setup.py").write_text("from setuptools import setup\n")
        unreadable = created["deptry"].measure(scope.Scope(Path(tmp), ["tool.py"]))
        check(none.skipped and {i.issues[0]["severity"] for i in (none, inline, unreadable)} == {"high"}
              and "setup.py" in unreadable.issues[0]["summary"],
              "no pyproject.toml dependencies or requirements.txt at the root is high, with PEP 723 blocks or a setup.py")

    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "mod.py").write_text("import x\n\n\ndef f(a):\n    return a\n\n\nclass C:\n    X = 1\n\n"
                                       "    def m(self):\n        return 2\n")
        index = scope.Scope(Path(tmp), ["mod.py"])
        check(vulture.runtime_lines(index, "mod.py", 4) == {5} and vulture.runtime_lines(index, "mod.py", 8) == {12},
              "coverage counts function and method bodies, not def or class-body lines that run on import")

    text = ('"""Module docstring,\n\nover three lines."""\n'
            "import os\n"
            "from pathlib import (\n    Path,\n)\n"
            "\n"
            "# a comment\n"
            "X = 1  # code with a comment\n"
            'HELP = """\n# not a comment: this string is code\n"""\n'
            "\n\n"
            "def f(a):\n"
            '    """Docstring."""\n'
            "    import json\n"
            "    return json.dumps(a)\n")
    check(scope.code_lines(text, ast.parse(text)) == 6,
          "file length counts code lines: not blank lines, comments, docstrings or imports")

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
        Ok({"x": "low"}), NothingToRead(), Broken()))
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
               "allowed": "ghp" + "_" + "".join(rnd.choice(alnum) for _ in range(36)),
               "between": "ghp" + "_" + "".join(rnd.choice(alnum) for _ in range(36))}  # committed after a scan
    # split so this file scans clean
    sbatch = "#SB" + "ATCH"
    account = "rrg" + "-someprof"
    mila = "/net" + "work/scratch/s/student/data"
    cluster = "/wo" + "rk1/somelab/student/data"
    drac = "def" + "-someprof"
    email = "selftest@" + "mila" + ".quebec"

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
    write("README.md", f"# demo\n\nContact: {email}\n")
    write("pyproject.toml", f'[project]\nname = "demo"\nauthors = [{{name = "Demo", email = "{email}"}}]\n')
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
    write("pkg/vendored.py", 'def borrowed(x):\n    return x\n\n\nPATTERN = "a\\.b"  # noqa: W605\n'  # an invalid escape
          'HINT = "add # noqa to hide it"\n')
    write("pkg/main.py", "import pkg.constants\nimport pkg.trainer\n"
          "from pkg.model import route, normalize_a, normalize_b\nfrom pkg.shared import dispatch\n"
          "from pkg.vendored import borrowed\n\n\n"
          "def main():\n    if True:\n" + "        print(route('k1', 1))\n" * 6 +
          "    print(normalize_a([1]), normalize_b([2]), borrowed(3), dispatch('k2', 2), pkg.trainer.bucket(40))\n\n\n"
          "if __name__ == '__main__':\n    main()\n")
    write("pkg/legacy.py", "def old_route(x):\n" + "    x += 1\n" * 25 + "    return x\n")
    write("pkg/tests/helpers.py", "def fixture_rows():\n" + "    rows = []\n" * 6 + "    return rows\n")
    write("pkg/launch.py", f'import argparse\n\nparser = argparse.ArgumentParser()\n'
          f'parser.add_argument("--account", default="{drac}")\n')
    write("scripts/sync.sh", f"#!/bin/bash\n{sbatch} --account={{account}}\nrsync -a {cluster} .\n")
    write("PLAN.md", "# plan\n")
    write("Untitled.ipynb", "{}\n")
    (repo / "loss_curve.png").write_bytes(b"\x89PNG")
    (root / "shared").mkdir()  # a tracked symlink to a module outside the repo
    (root / "shared/dispatch.py").write_text(f"def dispatch(kind, x):\n{branches}\n    return x\n")
    (repo / "pkg/shared.py").symlink_to("../../shared/dispatch.py")
    write("configs/model.yaml", "model:\n  target: pkg.model.ConfigOnlyModel\n  params:\n    width: 4\n")
    write("pkg/constants.py", "".join(f"C{n} = {n}\n" for n in range(1001)))
    write("pkg/trainer.py", "import os\n\n\ndef train(model, steps):\n    for _ in range(steps):\n        model.step()\n"
          "    model.save(os.path.join('runs', 'last.pt'))\n    return model\n\n\n"
          "def bucket(n_atoms):\n    if n_atoms > 38:\n        return 'large'\n    if n_atoms > 12:\n"
          "        return 'medium'\n    return 'small' if n_atoms > 5 else 'tiny'\n")
    write("tests/test_model.py", "def test_mode(model):\n    model.train()\n")
    write("config.py", f'GITHUB_TOKEN = "{secrets["github"]}"\n')
    write("settings.py", f'TOKEN = "{secrets["allowed"]}"  # gitleaks' + ':allow\n')
    write("train.sh", f"#!/bin/bash\n{sbatch} --account={account}\npython -m pkg.main --data {mila}\n")
    (repo / "data").mkdir()
    (repo / "data/blob.bin").write_bytes(rnd.randbytes(6_000_000))
    sh("add", "-A")
    sh("commit", "-qm", "demo repo")
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

        def at_head() -> dict:
            commit = subprocess.run(["git", "-C", r, "rev-parse", "HEAD"], capture_output=True, text=True).stdout
            folders = list((repo / ".revibe").glob(f"*-{commit.strip()[:12]}"))
            return json.loads((folders[0] / "metrics.json").read_text())

        check(cli("measure", r).returncode != 0, "measure refuses to run before the leak scan")
        p = cli("leaks", r)
        check(p.returncode == 0, f"leaks runs ({p.stderr.strip()[-200:]})")
        p = cli("measure", r, "--label", "baseline")
        check(p.returncode == 0, f"measure runs ({p.stderr.strip()[-200:]})")
        check("SyntaxWarning" not in p.stderr, "the repo's invalid escapes print no warnings")
        data = json.loads((repo / ".revibe/issues.json").read_text())
        scanned = {i["id"] for i in data["issues"] if i["category"] in ("leak", "internal-ref")}
        history_commit = data["leak_scan"]["history"]
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
        check(one("internal-ref", "internal-host-or-email", file="README.md")
              and not one("internal-ref", "internal-host-or-email", file="pyproject.toml"),
              "a lab email is an internal reference, except as an author in pyproject.toml")
        check(one("internal-ref", "internal-slurm-account", file="pkg/launch.py", severity="high"),
              "a DRAC account in an argparse default -> high")
        check(not one("internal-ref", "internal-slurm-account", file="scripts/sync.sh"),
              "an `--account={account}` template is not an account")
        check(one("internal-ref", "internal-cluster-path", file="scripts/sync.sh", severity="high"),
              "a /work1/<lab>/<user> path -> high")
        check(len(by.get(("internal-ref", "internal-cluster-path"), [])) == 1,
              "a Mila /network/scratch path is not also a cluster path")
        check(one("large-file", "large-file", severity="high", file="data/weights.bin"),
              "6 MB file at HEAD -> high, at its path there")
        check(one("license", "missing-license", severity="high"), "missing LICENSE -> high")
        check(one("dead-code", "function", symbol="unused_helper", status="open"), "dead function -> open")
        check(one("dead-code", "class", symbol="ConfigOnlyModel", status="dismissed"),
              "class named only in yaml `target:` -> dismissed")
        check(one("dead-code", "function", symbol="train", status="open"),
              "dead `train()` stays open although a test calls `model.train()`")
        check(one("dead-code", "unused-module", file="pkg/legacy.py", severity="high")
              and not one("dead-code", "function", symbol="old_route"),
              "a module nothing imports -> one high issue, not one per function in it")
        check(not one("dead-code", "unused-module", file="pkg/main.py"), "a script with a __main__ guard is not dead")
        check(not one("dead-code", "unreachable", file="pkg/main.py"), "the body of `if True:` is not dead code")
        check(not any(i["file"].startswith("pkg/tests/") for i in data["issues"]), "nested tests/ folders are out of scope")
        check(one("readiness", "clutter-output", severity="high") and one("readiness", "clutter-scratch", severity="high")
              and one("readiness", "clutter-notes", severity="medium"),
              "a root image, an Untitled notebook and a PLAN.md -> clutter issues")
        check(one("complexity", "cc", symbol="route", source="scb-check"), "CC > 10 -> complexity issue")
        check(one("complexity", "cc", symbol="dispatch", file="pkg/shared.py"),
              "a tracked symlink to a module outside the repo is measured")
        check(one("complexity", "cc", symbol="legacy_route", severity="low")
              and one("dead-code", "function", symbol="legacy_route", severity="high"),
              "complex dead function -> delete it (high), don't refactor it (low)")
        check(one("red-flag", "repetition"), "duplicated block -> repetition issue")
        check(one("readiness", "DEP001", symbol="numpy", severity="high"), "undeclared numpy -> high")
        metrics = at_head()["metrics"]
        check(one("file-length", "long-file", file="pkg/constants.py", severity="medium")
              and metrics["files_over_1000_code_lines"] == 1 and metrics["files_over_500_code_lines"] == 1
              and metrics["avg_code_lines"] > 0,
              "a module of 1001 code lines -> medium 'split it' issue, counted with the average file length")
        check(one("red-flag", "magic-numbers", symbol="bucket", severity="medium") and metrics["magic_comparisons"] > 0
              and not one("red-flag", "magic-numbers", symbol="route"),
              "comparisons against unnamed numbers -> medium magic-numbers issue; arguments and indices don't count")
        check(one("pathlib", "string-paths", severity="low") and metrics["string_paths"] > 0,
              "os.path calls -> one low string-paths issue, and a scorecard count")
        check(metrics["suppressions"] == 1, "a `# noqa` comment is a suppression; the same text in a string is not")
        check(one("globals", "module-globals", severity="low") and metrics["globals"] >= 1001,
              "module-level globals -> one low issue, and a scorecard count")
        if "pip-audit" in at_head()["not_measured"]:
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
        (repo / ".revibe/issues.json").write_text(json.dumps(data))
        flags = at_head()["metrics"]["red_flags_open"]
        cli("report", r)
        check(at_head()["metrics"]["red_flags_open"]
              == flags + 1, "report counts red flags added in review into the snapshot taken at HEAD")
        loc = at_head()["metrics"]["loc"]
        head_commit = subprocess.run(["git", "-C", r, "rev-parse", "HEAD"], capture_output=True, text=True).stdout
        legacy = repo / ".revibe" / head_commit[:12]  # how older versions named a commit's folder
        legacy.mkdir()
        (legacy / "metrics.json").write_text(json.dumps(at_head()))
        (repo / "scratch.py").write_text("def scratch(x):\n" + "    x += 1\n" * 40 + "    return x\n")  # untracked
        cli("measure", r, "--label", "baseline")
        (repo / "scratch.py").unlink()
        check(at_head()["metrics"]["loc"] == loc,
              "untracked files are not measured")
        folders = [f.name for f in (repo / ".revibe").iterdir() if f.is_dir() and f.name != "scripts"]
        check(not legacy.exists() and all(re.fullmatch(r"\d{10}-[0-9a-f]{12}", f) for f in folders),
              "metrics folders are named <commit epoch>-<commit>, and an older <commit> folder is replaced")
        data = json.loads((repo / ".revibe/issues.json").read_text())
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
        train = repo / "scripts/train.sh"
        train.write_text(train.read_text().replace("/net" + "work/scratch/s/student/data", "data/"))
        (repo / "pkg/config.py").write_text('"""Config."""\n\n' + (repo / "config.py").read_text())
        subprocess.run(["git", "-C", r, "rm", "-q", "config.py"], check=True)
        subprocess.run(["git", "-C", r, "add", "pkg/config.py"], check=True)
        subprocess.run(["git", "-C", r, "commit", "-qam", "delete dead code"], check=True)
        (repo / "notes.py").write_text(f'KEY = "{secrets["between"]}"\n')
        subprocess.run(["git", "-C", r, "add", "notes.py"], check=True)
        subprocess.run(["git", "-C", r, "commit", "-qm", "add a key"], check=True)
        subprocess.run(["git", "-C", r, "rm", "-q", "notes.py"], check=True)
        subprocess.run(["git", "-C", r, "commit", "-qm", "and remove it"], check=True)
        check(cli("measure", r, "--label", "after").returncode != 0, "a new commit needs a new leak scan")
        cli("leaks", r)
        cli("measure", r, "--label", "after")
        data = json.loads((repo / ".revibe/issues.json").read_text())
        check(any(i["symbol"] == "unused_helper" and i["status"] == "fixed" for i in data["issues"]),
              "deleted dead code -> fixed")
        check(len(list((repo / ".revibe").glob("*/metrics.json"))) == 2,
              "each measured commit keeps its metrics in its own folder; re-measuring one replaces them")
        check(next(i for i in data["issues"] if i["id"] == leak["id"])["status"] == "open",
              "the repo's own .betterleaksignore doesn't hide a secret from the audit")
        rescanned = {i["id"]: i for i in data["issues"] if i["category"] in ("leak", "internal-ref")}
        new = [rescanned[i] for i in set(rescanned) - scanned]
        check(data["leak_scan"]["history"] == history_commit and scanned <= set(rescanned),
              "a later scan reads only the new commits, and keeps the IDs earlier scans gave")
        check([(i["file"], i["severity"], i["evidence"]["at_head"]) for i in new] == [("notes.py", "critical", False)],
              "a secret committed and removed between two scans is found: critical, only in history")
        moved = rescanned[leak["id"]]
        check(moved["evidence"]["at_head"] and (moved["file"], moved["line"]) == ("pkg/config.py", 3),
              "a secret moved to another file, lower down, keeps its ID")
        token = next(i for i in rescanned.values() if i["rule"] == "huggingface-access-token")
        check(token["status"] == "open" and token["severity"] == "critical" and not token["evidence"]["at_head"],
              "a secret only in the commits read before stays open and critical")
        mila = next(i for i in rescanned.values() if i["rule"] == "internal-mila-path")
        slurm = next(i for i in rescanned.values() if i["rule"] == "internal-slurm-account")
        check(mila["status"] == "open" and mila["severity"] == "low" and slurm["severity"] == "high",
              "an internal path taken out of HEAD drops to history-only (low), not fixed; one still there stays high")
        p = cli("compare", r, "baseline")
        check(p.returncode == 0 and "Dead code LOC" in p.stdout, "compare prints before/after")
        p = cli("report", r)
        report = (repo / ".revibe/report.md").read_text()
        check(p.returncode == 0 and "**BLOCKED**" in report, "report renders and says BLOCKED")
        check("## Severity rules" in report and "- **critical:** credential" in report,
              "report ends with the severity rules")
        p = subprocess.run([sys.executable, str(repo / ".revibe/scripts/cleanup.py"), "compare", r, "baseline"],
                           capture_output=True, text=True)
        check(p.returncode == 0 and "Dead code LOC" in p.stdout, "the copy report leaves in .revibe/scripts/ runs on its own")
        cli("leaks", r, "--history")
        check(json.loads((repo / ".revibe/issues.json").read_text())["leak_scan"]["history"] != history_commit,
              "leaks --history reads every commit again")
        subprocess.run(["git", "-C", r, "commit", "-q", "--amend", "-m", "and remove it (amended)"], check=True)
        cli("leaks", r)
        amended = subprocess.run(["git", "-C", r, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        check(json.loads((repo / ".revibe/issues.json").read_text())["leak_scan"]["history"] == amended,
              "a commit read before that has left the history (amended) makes the next scan read every commit")
        check(not any(f.name == "__pycache__" for f in (repo / ".revibe").rglob("*")) and
              not any(CLEANUP.parent.rglob("__pycache__")), "no __pycache__ in .revibe/ or the skill folder")
        files = [f.read_text(errors="replace") for f in (repo / ".revibe").rglob("*") if f.is_file()]
        leaked = [k for k, v in secrets.items() if any(v in t for t in files + transcript)]
        check(not leaked, f"no plaintext secret in output or .revibe/ {leaked or ''}")


def main():
    offline()
    if "--offline" not in sys.argv[1:]:
        end_to_end()
    print("tests:", f"FAILED {len(failures)}" if failures else "passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
