"""pip-audit, in process: known vulnerabilities in the pinned dependencies. Its API is private (`pip_audit._*`),
so check this file when the pin in cleanup.py changes."""

import re
import shutil
import tempfile
from pathlib import Path

from common import run
from tools.base import Result, Tool, ToolError

TOOL = Tool(
    name="pip-audit",
    package="pip-audit",
    policy={
        "vuln-fixable": ("high", "dependency with a known vulnerability and a fixed release"),
        "vuln-unfixed": ("low", "known vulnerability without a fixed release"),
        "pins-none": ("low", "no lockfile or pinned requirements: can't audit, installs aren't reproducible"),
    },
)


def _fully_pinned(path: Path) -> bool:
    """Whether every requirement in `path` pins an exact version, so it can be audited without resolving."""
    pinned = r"^[A-Za-z0-9][\w.\-\[\], ]*==\s*[\w.+!-]+\s*(?:;.*)?(?:\\)?$"
    lines = [ln.split(" #")[0].strip() for ln in path.read_text().splitlines()]
    reqs = [ln for ln in lines if ln and not ln.startswith(("#", "--hash"))]
    return bool(reqs) and all(re.match(pinned, ln) for ln in reqs)


def audit_source(repo: Path, tmp: Path) -> tuple:
    """pip-audit's dependency source for the first pinned one, and its name: uv.lock, pylock.toml, fully
    pinned requirements, .venv. (None, "") when there is none: resolving unpinned requirements can take as
    long as installing them."""
    from pip_audit._dependency_source import PipSource, PyLockSource, RequirementSource

    if (repo / "uv.lock").exists() and (uv := shutil.which("uv")):
        req = tmp / "requirements.txt"
        p = run([uv, "export", "--project", repo, "--frozen", "--format", "requirements.txt",
                 "--no-emit-project", "--quiet", "-o", req])
        if p.returncode == 0:
            return RequirementSource([req], require_hashes=True, disable_pip=True), "uv.lock"
    if pylocks := sorted(repo.glob("pylock*.toml")):
        return PyLockSource(pylocks), "pylock.toml"
    for req in [repo / "requirements.txt", *sorted(repo.glob("requirements*.txt"))]:
        if req.exists() and _fully_pinned(req):
            return RequirementSource([req], no_deps=True, disable_pip=True), req.name
    if sites := sorted(repo.glob(".venv/lib/python*/site-packages")):
        return PipSource(paths=[sites[-1]]), ".venv"
    return None, ""


def vuln_issue(name: str, version: str, vulns: list, label: str) -> dict:
    """A vulnerable dependency, with the lowest release that fixes all of its known vulnerabilities."""
    vulns = {v.id: v for v in vulns}
    fixed = [min(v.fix_versions) for v in vulns.values() if v.fix_versions]
    target = str(max(fixed)) if fixed else None
    ids = sorted(vulns)
    return TOOL.issue(
        "VULN", "vulnerable-dependency", "pip-audit", "vuln-fixable" if target else "vuln-unfixed",
        f"{name} {version}: {len(ids)} known vulnerabilities ({', '.join(ids[:3])}"
        f"{', ...' if len(ids) > 3 else ''})" + (f"; fixed in {target}" if target else "; no fix yet"),
        key=f"VULN|{name.lower()}", file=label, symbol=name,
        evidence={"version": version, "vulns": ids, "fix_to": target,
                  "aliases": sorted({a for v in vulns.values() for a in v.aliases})[:10]},
        fix=(f"Bump {name} to >= {target} in the dependency spec, then run the safety net."
             if target else "No fixed release yet: note it, or drop the dependency if it is easy."),
        done_when=f"pip-audit reports no vulnerabilities for {name}.")


def measure(repo, files, index) -> Result:
    from pip_audit._audit import Auditor
    from pip_audit._dependency_source import DependencySourceError
    from pip_audit._service import PyPIService, ServiceError

    with tempfile.TemporaryDirectory(prefix="cleanup-audit-") as tmp:
        source, label = audit_source(repo, Path(tmp))
        if source is None:
            return Result([TOOL.issue(
                "DEPS", "readiness", "no-pinned-dependencies", "pins-none",
                "No lockfile or pinned requirements: dependencies can't be audited and installs aren't "
                "reproducible", key="DEPS|pin", effort="S",
                fix="Commit a lockfile (uv.lock) or a fully pinned requirements.txt.",
                done_when="`cleanup.py measure` runs pip-audit without 'not measured'.")], {},
                skipped="no pinned source (uv.lock, pylock, pinned requirements, .venv)")
        try:
            audited = list(Auditor(PyPIService(timeout=15)).audit(source))
        except (DependencySourceError, ServiceError) as e:
            raise ToolError(str(e)) from e
    out = [vuln_issue(dep.canonical_name, str(dep.version), vulns, label) for dep, vulns in audited if vulns]
    return Result(out, {"pip_audit_source": label, "vulnerable_deps": len(out)})
