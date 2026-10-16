"""deptry: imports that aren't declared dependencies, and declared dependencies nobody imports. Its Python API
exits the process, so it runs as `python -m deptry` from this script's environment and reports in JSON."""

import json
import sys
import tempfile
from pathlib import Path

from utils.paths import rel
from utils.process import run
from utils.pyproject import read_pyproject
from tools.base import Metric, Result, Step, ToolError


class Deptry(Step):
    """Undeclared, unused and dev-only dependencies."""

    name = "deptry"
    package = "deptry"
    version = "0.25.1"
    policy = {
        "deps-none": "no pyproject.toml dependencies or requirements.txt at the repo root",
        "deps-undeclared": "an import that isn't a declared dependency, or only a dev one",
        "deps-unused": "declared dependency that is never imported",
    }
    metrics = (Metric("deps_missing", "Dependency problems",
                      description="imports not declared as dependencies, and declared ones never imported"),)

    def no_dependency_list(self, repo: Path) -> dict:
        """No pyproject.toml dependencies or requirements.txt at the repo root. A list anywhere else (setup.py,
        requirements-dev.txt, env/, environment.yml, PEP 723 script blocks) doesn't count; one at the root is named."""
        elsewhere = [n for n in ("setup.py", "setup.cfg") if (repo / n).exists()] + \
            sorted(p.name for p in repo.glob("requirements?*.txt"))
        found = f"; found only {', '.join(elsewhere)}" if elsewhere else ""
        return self.issue(
            "DEPS", "readiness", "no-dependency-list", "deps-none",
            f"No pyproject.toml dependencies or requirements.txt at the repo root{found}",
            key="DEPS|declare", effort="S",
            fix=("Declare the runtime dependencies in pyproject.toml ([project].dependencies), or in a requirements.txt "
                 "at the repo root. Move a list kept elsewhere (env/, environment.yml, a script's PEP 723 block) "
                 "there."),
            done_when="`cleanup.py measure` runs deptry without 'not measured'.")

    def measure(self, scope) -> Result:
        """Undeclared, unused and dev-only dependencies, or one issue saying there is no list deptry can read."""
        repo = scope.repo
        source = deptry_source(repo)
        if source is None:
            return Result([self.no_dependency_list(repo)], {},
                          skipped="no pyproject.toml dependencies or requirements.txt")
        # packages whose import name deptry can't guess when they aren't installed where it runs
        name_map = ("scikit-learn=sklearn,pillow=PIL,opencv-python=cv2,opencv-python-headless=cv2,pyyaml=yaml,"
                    "hydra-core=hydra,pytorch-lightning=pytorch_lightning,scikit-image=skimage,beautifulsoup4=bs4,"
                    "protobuf=google.protobuf,huggingface-hub=huggingface_hub,faiss-cpu=faiss,faiss-gpu=faiss,"
                    "python-dateutil=dateutil,attrs=attr,tensorboardx=tensorboardX,ftfy=ftfy")
        with tempfile.TemporaryDirectory(prefix="cleanup-deptry-") as tmp:
            report = Path(tmp) / "deptry.json"
            # -I: a deptry.py in the working directory (this module's, or the repo's) must not shadow deptry
            p = run([sys.executable, "-I", "-m", "deptry", repo, *source, "--package-module-name-map", name_map,
                     "--json-output", report])
            if not report.exists():
                raise ToolError((p.stderr or p.stdout).strip()[-300:])
            violations = json.loads(report.read_text())
        fixes = {
            "DEP001": "Declare the package that provides `{}` as a dependency.",
            "DEP002": "Remove `{}` from the dependencies if it is really unused.",
            "DEP004": "`{}` is a dev dependency but the code imports it: move it to the main dependencies.",
        }
        out = []
        for v in violations:
            code = v["error"]["code"]
            module = v["module"]
            message = v["error"]["message"]
            # DEP003, "installed only as a transitive dependency", describes the environment deptry runs in: this
            # script's, never the project's. So it is an undeclared import like DEP001.
            if code == "DEP003":
                code = "DEP001"
                message = f"'{module}' imported but missing from the dependency definitions"
            out.append(self.issue(
                "DEP", "readiness", code, "deps-unused" if code == "DEP002" else "deps-undeclared",
                message, file=rel(repo, v["location"]["file"]), symbol=module, line=v["location"]["line"],
                fix=fixes[code].format(module) if code in fixes else message, effort="S",
                done_when="deptry no longer reports it."))
        return Result(out, {"deps_missing": sum(v["error"]["code"] != "DEP002" for v in violations)})


def deptry_source(repo: Path) -> list | None:
    """deptry's arguments for the dependency list it can read: pyproject.toml, then requirements.txt."""
    project = read_pyproject(repo)
    if project.get("project", {}).get("dependencies") is not None or \
            project.get("tool", {}).get("poetry", {}).get("dependencies"):
        return ["--config", repo / "pyproject.toml"]
    if (repo / "requirements.txt").exists():
        return ["--requirements-files", repo / "requirements.txt"]
    return None
