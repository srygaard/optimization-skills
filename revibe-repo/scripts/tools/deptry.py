"""deptry: imports that aren't declared dependencies, and declared dependencies nobody imports. Its Python API
exits the process, so it runs as `python -m deptry` from this script's environment and reports in JSON."""

import json
import sys
import tempfile
from pathlib import Path

from common import pyproject, rel, run
from tools.base import Result, Tool, ToolError

TOOL = Tool(
    name="deptry",
    package="deptry",
    policy={
        "deps-none": ("high", "no dependency list at all: nobody can install it"),
        "deps-undeclared": ("high", "an import that isn't a declared dependency, or only a dev one"),
        "deps-unused": ("low", "declared dependency that is never imported"),
        "deps-unreadable": ("low", "dependencies listed only where deptry can't read them (environment.yml, "
                                   "setup.py, PEP 723)"),
    },
)


def deptry_source(repo: Path) -> list | None:
    """deptry's arguments for the dependency list it can read: pyproject.toml, then requirements.txt."""
    project = pyproject(repo)
    if project.get("project", {}).get("dependencies") is not None or \
            project.get("tool", {}).get("poetry", {}).get("dependencies"):
        return ["--config", repo / "pyproject.toml"]
    if (repo / "requirements.txt").exists():
        return ["--requirements-files", repo / "requirements.txt"]
    return None


def no_dependency_list(repo: Path, files: list) -> dict:
    """High when nothing lists the dependencies; low when they are listed only where deptry can't read them."""
    elsewhere = [n for n in ("environment.yml", "environment.yaml", "setup.py", "setup.cfg", "Pipfile")
                 if (repo / n).exists()] + sorted(p.name for p in repo.glob("requirements?*.txt"))
    if any("# /// script" in (repo / f).read_text(errors="replace") for f in files):
        elsewhere.append("PEP 723 script blocks")
    return TOOL.issue(
        "DEPS", "readiness", "no-dependency-list", "deps-unreadable" if elsewhere else "deps-none",
        f"Dependencies are declared only in {', '.join(elsewhere)}, which deptry can't check" if elsewhere
        else "No dependency list (pyproject.toml, requirements.txt, environment.yml): nobody can install it",
        key="DEPS|declare", effort="S",
        fix=("Declare the runtime dependencies in pyproject.toml ([project].dependencies)."
             + (" Self-contained PEP 723 scripts can dismiss this." if elsewhere else "")),
        done_when="`cleanup.py measure` runs deptry without 'not measured'.")


def measure(repo, files, index) -> Result:
    """Undeclared, unused and dev-only dependencies, or one issue saying there is no list deptry can read."""
    source = deptry_source(repo)
    if source is None:
        return Result([no_dependency_list(repo, files)], {},
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
        out.append(TOOL.issue(
            "DEP", "readiness", code, "deps-unused" if code == "DEP002" else "deps-undeclared",
            message, file=rel(repo, v["location"]["file"]), symbol=module, line=v["location"]["line"],
            fix=fixes[code].format(module) if code in fixes else message, effort="S",
            done_when="deptry no longer reports it."))
    return Result(out, {"deps_missing": sum(v["error"]["code"] != "DEP002" for v in violations)})
