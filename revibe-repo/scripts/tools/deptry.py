"""deptry: imports that aren't declared dependencies, and declared dependencies nobody imports. Its Python API
exits the process, so it runs as `python -m deptry` from this script's environment and reports in JSON."""

import json
import re
import sys
import tomllib
import tempfile
from pathlib import Path

from utils.git import git
from utils.paths import rel
from utils.process import run
from utils.pyproject import read_pyproject
from tools.base import Metric, Result, Step, ToolError

SETTINGS_FILE = Path(__file__).parent / "deptry.toml"


class Deptry(Step):
    """Undeclared, unused and dev-only dependencies."""

    name = "deptry"
    package = "deptry"
    version = "0.25.1"
    policy = {
        "deps-none": "no pyproject.toml dependencies or requirements.txt at the repo root",
        "deps-undeclared": "an import that isn't a declared dependency, or only a dev one",
        "deps-unused": "declared dependency that is never imported, the project itself and dev-only tooling aside",
    }
    metrics = (Metric("deps_missing", "Dependency problems",
                      description="imports in the code in scope (tests aside) not declared as dependencies, and "
                                  "declared ones it never imports"),)

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
        """Undeclared, unused and dev-only dependencies of the code in scope, or one issue saying there is no list
        deptry can read. Tests are out of scope, so a test importing a dev dependency such as pytest is fine."""
        repo = scope.repo
        source = deptry_source(repo)
        if source is None:
            return Result([self.no_dependency_list(repo)], {},
                          skipped="no pyproject.toml dependencies or requirements.txt")
        with tempfile.TemporaryDirectory(prefix="cleanup-deptry-") as tmp:
            report = Path(tmp) / "deptry.json"
            # In the repo, on ".": deptry matches its excludes against the paths it walks, so they only work when
            # those are relative. -I: a deptry.py in the repo must not shadow deptry.
            p = run([sys.executable, "-I", "-m", "deptry", ".", *source, *deptry_excludes(scope),
                     *deptry_first_party(scope),
                     "--package-module-name-map", deptry_name_map(repo), "--json-output", report], cwd=repo)
            if not report.exists():
                raise ToolError((p.stderr or p.stdout).strip()[-300:])
            violations = json.loads(report.read_text())
        fixes = {
            "DEP001": "Declare the package that provides `{}` as a dependency.",
            "DEP002": "Remove `{}` from the dependencies if it is really unused.",
            "DEP004": "`{}` is a dev dependency but the code imports it: move it to the main dependencies.",
        }
        exempt = unused_exempt(repo)
        violations = [v for v in violations
                      if v["error"]["code"] != "DEP002" or v["module"].lower() not in exempt]
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


def unused_exempt(repo: Path) -> set:
    """Packages that are declared but, rightly, imported nowhere, so "unused" says nothing about them: the tooling in
    a dev extra or a PEP 735 dependency group (tests are out of scope, and a formatter, docs builder or pytest plugin
    is run as a command, never imported), and the project itself, which an `all` extra usually re-lists. A package
    also declared as a runtime dependency keeps its finding."""
    project = read_pyproject(repo).get("project", {})
    extras = project.get("optional-dependencies", {})
    dev_extras = set(tomllib.loads(SETTINGS_FILE.read_text()).get("dev_extras", []))
    groups = read_pyproject(repo).get("dependency-groups", {})
    requirements = [r for name, extra in extras.items() if name.lower() in dev_extras for r in extra]
    requirements += [r for group in groups.values() for r in group if isinstance(r, str)]
    exempt = {normalize_package(r) for r in requirements}
    exempt.add(normalize_package(project.get("name", "")))
    return exempt - {normalize_package(r) for r in project.get("dependencies", [])} - {""}


def normalize_package(requirement: str) -> str:
    """The distribution name a requirement starts with, lowercased, as deptry names it in a finding."""
    return re.split(r"[\[<>=!;~\s]", requirement.strip(), maxsplit=1)[0].strip().lower()


def declared_import_names(repo: Path) -> set:
    """The import names the declared dependencies provide. A local module of the same name must not be passed off
    as first-party, or it would hide a real finding about the package."""
    project = read_pyproject(repo).get("project", {})
    requirements = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        requirements.extend(extra)
    if (repo / "requirements.txt").exists():
        requirements.extend((repo / "requirements.txt").read_text().splitlines())
    name_map = builtin_import_names()
    names = set()
    for requirement in requirements:
        package = normalize_package(requirement)
        if not package or package.startswith(("#", "-")):
            continue
        modules = name_map.get(package, package)
        if isinstance(modules, str):
            modules = [modules]
        names.update(module.replace("-", "_") for module in modules)
    return names


def deptry_first_party(scope) -> list:
    """deptry's --known-first-party arguments: the repo's own top-level modules, which it can't always detect. It
    counts a directory as local only when it holds an `__init__.py`, so a flat folder of scripts importing its
    siblings (`figures/plot_a.py` importing `plot_b`) looks to it like a missing dependency. A name a declared
    dependency also provides is left out, so a local `trainers/torch.py` can't hide an undeclared torch."""
    declared = declared_import_names(scope.repo)
    local = {Path(f).stem for f in scope.files if f.endswith(".py")}
    local |= {Path(f).parent.name for f in scope.files if Path(f).name == "__init__.py"}
    names = sorted(n for n in local - declared if n.isidentifier() and n != "__init__")
    return [arg for name in names for arg in ("--known-first-party", name)]


def builtin_import_names() -> dict:
    """The import name of each package whose own name doesn't give it away, from the deptry.toml next to this
    module. Edit that file to teach deptry another package."""
    return tomllib.loads(SETTINGS_FILE.read_text()).get("package_module_name_map", {})


def deptry_name_map(repo: Path) -> str:
    """deptry's --package-module-name-map, built from the deptry.toml next to this module. The repo's own [tool.deptry]
    package_module_name_map is merged in and wins, since the command line would otherwise replace it. A package
    naming several modules joins them with `|`."""
    own = read_pyproject(repo).get("tool", {}).get("deptry", {}).get("package_module_name_map", {})
    if not isinstance(own, dict):
        own = {}
    pairs = []
    for package, modules in {**builtin_import_names(), **own}.items():
        if not isinstance(modules, str):
            modules = "|".join(modules)
        pairs.append(f"{package}={modules}")
    return ",".join(pairs)


def deptry_excludes(scope) -> list:
    """deptry's --extend-exclude arguments: the repo's own, which the command line would otherwise replace, and the
    tracked Python files outside the scope (tests, and whatever else measure leaves out), each matched exactly."""
    own = read_pyproject(scope.repo).get("tool", {}).get("deptry", {}).get("extend_exclude", [])
    tracked = {f for f in git(scope.repo, "ls-files", "-z", "--", "*.py").split("\0") if f}
    outside = sorted(tracked - set(scope.files))
    patterns = [*own, "(?:" + "|".join(map(re.escape, outside)) + ")$"] if outside else own
    return [arg for pattern in patterns for arg in ("--extend-exclude", pattern)]


def deptry_source(repo: Path) -> list | None:
    """deptry's arguments for the dependency list it can read: pyproject.toml, then requirements.txt."""
    project = read_pyproject(repo)
    if project.get("project", {}).get("dependencies") is not None or \
            project.get("tool", {}).get("poetry", {}).get("dependencies"):
        return ["--config", repo / "pyproject.toml"]
    if (repo / "requirements.txt").exists():
        return ["--requirements-files", repo / "requirements.txt"]
    return None
