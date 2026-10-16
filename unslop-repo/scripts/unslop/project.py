"""Project-level checks: dependencies (deptry, pip-audit), license and README, file length."""

import json
import re
import shutil
import tempfile
import tomllib
from pathlib import Path

from .common import (
    LONG_FILE_LINES, Result, TOOLS, ToolError, WARN_FILE_LINES, git, rel, run, uvx, version_key,
)
from .state import issue


# --- measure: dependencies -------------------------------------------------------------------------

DEPTRY_MAP = ("scikit-learn=sklearn,pillow=PIL,opencv-python=cv2,opencv-python-headless=cv2,pyyaml=yaml,"
              "hydra-core=hydra,pytorch-lightning=pytorch_lightning,scikit-image=skimage,beautifulsoup4=bs4,"
              "protobuf=google.protobuf,huggingface-hub=huggingface_hub,faiss-cpu=faiss,faiss-gpu=faiss,"
              "python-dateutil=dateutil,attrs=attr,tensorboardx=tensorboardX,ftfy=ftfy")


def _pyproject(repo: Path) -> dict:
    try:
        return tomllib.loads((repo / "pyproject.toml").read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def deptry_deps(repo, files, index) -> Result:
    pyproject = _pyproject(repo)
    if pyproject.get("project", {}).get("dependencies") is not None or \
            pyproject.get("tool", {}).get("poetry", {}).get("dependencies"):
        source = ["--config", repo / "pyproject.toml"]
    elif (repo / "requirements.txt").exists():
        source = ["--requirements-files", repo / "requirements.txt"]
    else:
        elsewhere = [n for n in ("environment.yml", "environment.yaml", "setup.py", "setup.cfg", "Pipfile")
                     if (repo / n).exists()] + sorted(p.name for p in repo.glob("requirements?*.txt"))
        if any("# /// script" in (repo / f).read_text(errors="replace") for f in files):
            elsewhere.append("PEP 723 script blocks")
        return Result([issue(
            "DEPS", "deptry", "readiness", "no-dependency-list", "deps-unreadable" if elsewhere else "deps-none",
            f"Dependencies are declared only in {', '.join(elsewhere)}, which deptry can't check" if elsewhere
            else "No dependency list (pyproject.toml, requirements.txt, environment.yml): nobody can install it",
            key="DEPS|declare", effort="S",
            fix=("Declare the runtime dependencies in pyproject.toml ([project].dependencies)."
                 + (" Self-contained PEP 723 scripts can dismiss this." if elsewhere else "")),
            done_when="`cleanup.py measure` runs deptry without 'not measured'.")], {},
            skipped="no pyproject.toml dependencies or requirements.txt")
    with tempfile.TemporaryDirectory(prefix="cleanup-deptry-") as tmp:
        report = Path(tmp) / "deptry.json"
        p = run([uvx(), f"deptry@{TOOLS['deptry']}", repo, *source, "--package-module-name-map", DEPTRY_MAP,
                 "--json-output", report])
        if not report.exists():
            raise ToolError((p.stderr or p.stdout).strip()[-300:])
        violations = json.loads(report.read_text())
    out = []
    for v in violations:
        code = v["error"]["code"]
        module = v["module"]
        file = rel(repo, v["location"]["file"])
        fixes = {"DEP001": f"Declare the package that provides `{module}` as a dependency.",
                 "DEP002": f"Remove `{module}` from the dependencies if it is really unused.",
                 "DEP003": f"`{module}` is only installed transitively: declare it directly.",
                 "DEP004": f"`{module}` is a dev dependency but the code imports it: move it to the main dependencies."}
        out.append(issue(
            "DEP", "deptry", "readiness", code, "deps-unused" if code == "DEP002" else "deps-undeclared",
            v["error"]["message"], file=file, symbol=module, line=v["location"]["line"],
            fix=fixes.get(code, v["error"]["message"]), effort="S",
            done_when="deptry no longer reports it."))
    return Result(out, {"deps_missing": sum(v["error"]["code"] != "DEP002" for v in violations)})


PINNED = re.compile(r"^[A-Za-z0-9][\w.\-\[\], ]*==\s*[\w.+!-]+\s*(?:;.*)?(?:\\)?$")


def _fully_pinned(path: Path) -> bool:
    lines = [ln.split(" #")[0].strip() for ln in path.read_text().splitlines()]
    reqs = [ln for ln in lines if ln and not ln.startswith(("#", "--hash"))]
    return bool(reqs) and all(PINNED.match(ln) for ln in reqs)


def audit_source(repo: Path, tmp: Path) -> tuple:
    """pip-audit's arguments for the first pinned source, and its name: uv.lock, pylock.toml, fully pinned
    requirements, .venv. (None, "") when there is none: resolving unpinned requirements can take as long as
    installing them."""
    if (repo / "uv.lock").exists() and (uv := shutil.which("uv")):
        req = tmp / "requirements.txt"
        p = run([uv, "export", "--project", repo, "--frozen", "--format", "requirements.txt",
                 "--no-emit-project", "--quiet", "-o", req])
        if p.returncode == 0:
            return ["-r", req, "--require-hashes", "--disable-pip"], "uv.lock"
    if list(repo.glob("pylock*.toml")):
        return ["--locked", repo], "pylock.toml"
    for req in [repo / "requirements.txt", *sorted(repo.glob("requirements*.txt"))]:
        if req.exists() and _fully_pinned(req):
            return ["-r", req, "--no-deps", "--disable-pip"], req.name
    if sites := sorted(repo.glob(".venv/lib/python*/site-packages")):
        return ["--path", sites[-1]], ".venv"
    return None, ""


def vuln_issue(dep: dict, label: str) -> dict:
    """A vulnerable dependency, with the lowest release that fixes all of its known vulnerabilities."""
    vulns = {v["id"]: v for v in dep["vulns"]}
    fixed = [min(v["fix_versions"], key=version_key) for v in vulns.values() if v["fix_versions"]]
    target = max(fixed, key=version_key) if fixed else None
    ids = sorted(vulns)
    return issue(
        "VULN", "pip-audit", "vulnerable-dependency", "pip-audit", "vuln-fixable" if target else "vuln-unfixed",
        f"{dep['name']} {dep['version']}: {len(ids)} known vulnerabilities ({', '.join(ids[:3])}"
        f"{', ...' if len(ids) > 3 else ''})" + (f"; fixed in {target}" if target else "; no fix yet"),
        key=f"VULN|{dep['name'].lower()}", file=label, symbol=dep["name"],
        evidence={"version": dep["version"], "vulns": ids, "fix_to": target,
                  "aliases": sorted({a for v in vulns.values() for a in v.get("aliases", [])})[:10]},
        fix=(f"Bump {dep['name']} to >= {target} in the dependency spec, then run the safety net."
             if target else "No fixed release yet: note it, or drop the dependency if it is easy."),
        done_when=f"pip-audit reports no vulnerabilities for {dep['name']}.")


def pip_audit_vulns(repo, files, index) -> Result:
    with tempfile.TemporaryDirectory(prefix="cleanup-audit-") as tmp:
        source, label = audit_source(repo, Path(tmp))
        if source is None:
            return Result([issue(
                "DEPS", "pip-audit", "readiness", "no-pinned-dependencies", "pins-none",
                "No lockfile or pinned requirements: dependencies can't be audited and installs aren't "
                "reproducible", key="DEPS|pin", effort="S",
                fix="Commit a lockfile (uv.lock) or a fully pinned requirements.txt.",
                done_when="`cleanup.py measure` runs pip-audit without 'not measured'.")], {},
                skipped="no pinned source (uv.lock, pylock, pinned requirements, .venv)")
        report = Path(tmp) / "audit.json"
        p = run([uvx(), f"pip-audit@{TOOLS['pip-audit']}", *source, "--format", "json", "-o", report,
                 "--progress-spinner", "off"])
        if p.returncode not in (0, 1) or not report.exists():
            raise ToolError(p.stderr.strip()[-300:])
        deps = json.loads(report.read_text()).get("dependencies", [])
    out = [vuln_issue(dep, label) for dep in deps if dep.get("vulns")]
    return Result(out, {"pip_audit_source": label, "vulnerable_deps": len(out)})


# --- measure: license and readme -------------------------------------------------------------------

HEADER_LICENSES = (
    ("Apache-2.0", r"apache license|apache-2\.0"), ("AGPL", r"affero general public|agpl"),
    ("LGPL", r"lesser general public|lgpl"), ("GPL", r"gnu general public|\bgpl"),
    ("MPL-2.0", r"mozilla public|mpl-2\.0"), ("CC-BY-NC", r"noncommercial|cc-by-nc"),
    ("BSD", r"redistribution and use in source and binary|\bbsd"), ("MIT", r"permission is hereby granted|\bmit\b"),
)


def detect_license(text: str) -> str | None:
    t = " ".join(text.split()).lower()
    if "apache license" in t and "version 2.0" in t:
        return "Apache-2.0"
    if "mozilla public license" in t:
        return "MPL-2.0"
    for family, words in (("AGPL", "gnu affero general public license"), ("LGPL", "gnu lesser general public"),
                          ("GPL", "gnu general public license")):
        if words in t:
            return family + ("-3.0" if "version 3" in t else "-2.0" if "version 2" in t else "")
    if "permission is hereby granted, free of charge" in t:
        return "MIT"
    if "redistribution and use in source and binary forms" in t:
        return "BSD-3-Clause" if "neither the name" in t else "BSD-2-Clause"
    if "permission to use, copy, modify, and/or distribute this software" in t:
        return "ISC"
    if "free and unencumbered software released into the public domain" in t:
        return "Unlicense"
    if "attribution-noncommercial" in t:
        return "CC-BY-NC"
    return None


def family(spdx: str | None) -> str:
    """Compare licenses by family: GPL-3.0-only and GPL-3.0 match, MIT and Apache-2.0 don't."""
    if not spdx:
        return ""
    s = spdx.lower().replace("-only", "").replace("-or-later", "").replace("+", "")
    for fam, pattern in HEADER_LICENSES:
        if re.search(pattern, s):
            return fam
    return s


LICENSE_FILE = re.compile(r"(?i)(?:licen[cs]e|copying)(?:\.(?:md|txt|rst))?")
SOURCE_SUFFIXES = {".py", ".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".sh", ".js", ".ts"}


def license_file_issues(repo: Path, lic_file: str | None, repo_license: str | None) -> list:
    """A missing or unrecognized LICENSE file, or one that pyproject.toml contradicts."""
    if not lic_file:
        return [issue(
            "LIC", "license", "license", "missing-license", "license-missing",
            "No LICENSE file: without one, nobody may legally use, copy or modify the code",
            key="LIC|missing", effort="S",
            fix=("Ask the user which license to use (MIT and Apache-2.0 are common for research code; "
                 "Apache-2.0 adds a patent grant). Add LICENSE at the repo root and set `license` in "
                 "pyproject.toml."),
            done_when="A LICENSE file is at the root and `cleanup.py measure` recognizes it.")]
    out = []
    if repo_license is None:
        out.append(issue(
            "LIC", "license", "license", "unrecognized-license", "license-unrecognized",
            f"{lic_file} doesn't match a common license (MIT, Apache-2.0, BSD, GPL, MPL)",
            key="LIC|unrecognized", file=lic_file, effort="S",
            fix="Check the text; a standard license text lets people and tools recognize it.",
            done_when="`cleanup.py measure` recognizes the license."))
    declared = _pyproject(repo).get("project", {}).get("license")
    declared = declared.get("text") if isinstance(declared, dict) else declared
    if repo_license and isinstance(declared, str) and family(declared) != family(repo_license):
        out.append(issue(
            "LIC", "license", "license", "license-mismatch", "license-mismatch",
            f"pyproject.toml declares `{declared}` but {lic_file} is {repo_license}",
            key="LIC|mismatch", file="pyproject.toml", effort="S",
            fix="Make pyproject's `license` match the LICENSE file.",
            done_when="Both name the same license."))
    return out


def header_license(top: str) -> str | None:
    """The license family named in a file's first lines (lowercased): an SPDX identifier, or a license's words
    in a header that mentions a license. MIT's words are too common to count without SPDX."""
    if spdx := re.search(r"spdx-license-identifier:\s*([\w.+-]+)", top):
        return family(spdx[1])
    if "license" not in top and "licence" not in top:
        return None
    return next((fam for fam, pattern in HEADER_LICENSES if fam != "MIT" and re.search(pattern, top)), None)


def third_party_issues(repo: Path, repo_license: str | None) -> list:
    """Source files whose header names another license: usually vendored code, whose notice must stay."""
    vendored = {}
    for name in git(repo, "ls-files", "-z").split("\0"):
        if Path(name).suffix not in SOURCE_SUFFIXES:
            continue
        try:
            top = "\n".join((repo / name).read_text(errors="replace").splitlines()[:40]).lower()
        except OSError:
            continue
        found = header_license(top)
        if found and found != family(repo_license):
            vendored.setdefault(found, []).append(name)
    out = []
    for fam, paths in sorted(vendored.items()):
        out.append(issue(
            "LIC", "license", "license", "third-party-license", "license-third-party",
            f"{len(paths)} file(s) carry a {fam} license header but the repo is "
            f"{repo_license or 'unlicensed'}: {', '.join(paths[:4])}{', ...' if len(paths) > 4 else ''}",
            key=f"LIC|vendored|{fam}", file=paths[0], evidence={"files": paths[:50]}, effort="S",
            fix=(f"Keep these notices. Check that {fam} code may ship under the repo license (GPL/AGPL code "
                 "can't be relicensed as MIT or Apache-2.0; non-commercial code isn't open source) and "
                 "credit the original project in the README or a NOTICE file."),
            done_when="The user has confirmed the licenses are compatible (or the files are replaced)."))
    return out


def license_readme(repo, files, index) -> Result:
    """The LICENSE file, files under another license, and the README."""
    names = git(repo, "ls-tree", "--name-only", "HEAD").splitlines()
    lic_file = next((n for n in names if LICENSE_FILE.fullmatch(n)), None)
    repo_license = detect_license((repo / lic_file).read_text(errors="replace")) if lic_file else None
    out = license_file_issues(repo, lic_file, repo_license) + third_party_issues(repo, repo_license)
    if not any(re.fullmatch(r"(?i)readme(?:\.\w+)?", n) for n in names):
        out.append(issue(
            "DOC", "license", "readiness", "missing-readme", "readme-missing", "No README at the repo root",
            key="DOC|readme", effort="M",
            fix="Add a README: what the code does, how to install it, a minimal quickstart, how to cite it.",
            done_when="README exists and its quickstart runs."))
    return Result(out, {"license": repo_license or ("unrecognized" if lic_file else "missing")})


def count_suppressions(repo: Path, files: list) -> int:
    pattern = re.compile(r"#\s*(?:noqa|scbc\s+ignore)", re.I)
    return sum(len(pattern.findall((repo / f).read_text(errors="replace"))) for f in files)


def file_lengths(repo, files, index) -> Result:
    """Average length of the source files, how many pass 500 and 1000 lines, and an issue for the latter."""
    lines = index.lines
    out = [issue(
        "SIZE", "file-length", "file-length", "long-file", "long-file", f"`{f}` has {n:,} lines",
        file=f, evidence={"lines": n, "weight": n}, effort="M",
        fix=("Split it into modules of a few hundred lines, each owning one concept (what it hides, not when it "
             "runs). Move the code unchanged in its own commit, then refactor; if other code imports from this "
             "module, keep that import path working."),
        done_when=f"It is under {LONG_FILE_LINES} lines, and the import test and safety net pass.")
        for f, n in sorted(lines.items()) if n > LONG_FILE_LINES]
    return Result(out, {"avg_file_lines": round(sum(lines.values()) / len(lines)) if lines else 0,
                        "files_over_500": sum(n > WARN_FILE_LINES for n in lines.values()),
                        "files_over_1000": len(out)})
