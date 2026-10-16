"""Pinned tools, thresholds, the severity POLICY, and helpers shared by every step."""

import datetime as dt
import re
import shutil
import subprocess
from pathlib import Path
from typing import NamedTuple


# --- pinned tools ----------------------------------------------------------------------------------

BETTERLEAKS_VERSION = "1.9.0"
BETTERLEAKS_SHA256 = {  # checksums.txt of the v1.9.0 release, signature verified with sigstore
    ("darwin", "arm64"): "83dd7eaab13d44a1e8e347a17064cdef153a1478db473276d41b38d55613d3f2",
    ("darwin", "x64"): "68dfd83458d9e7f90663daa7a81c7f944083b170dae82e4f5f8d0581ed519ef1",
    ("linux", "arm64"): "1d39116e0a58dc94574715e2aa12a2dbd5062f193eee3fec011fef6ba06bd13b",
    ("linux", "x64"): "f8b185a39ffcece2a1ca82bf3a4e7435cd81963ffd16b7a9128daf75f35f6de7",
}
TOOLS = {"scb-check": "0.2.0", "vulture": "2.16", "deptry": "0.25.1", "pip-audit": "2.10.1"}
RETIRED_SOURCES = {"ruff": "scb-check reports complexity now"}  # their open issues get dismissed

# --- thresholds ------------------------------------------------------------------------------------

CC_THRESHOLD = 10  # scb-check 0.2.0's threshold: Erosion is the share of mass in functions above it
MEDIUM_MASS = 100  # complexity mass CC x sqrt(SLOC) from which a function is medium
MEDIUM_DUP_LINES = 20  # duplicate-group weight lines x (copies - 1) from which it is medium
MIN_CLONE_LINES = 10  # smaller duplicate blocks still count in Clone Ratio but get no issue
MIN_DEAD_LINES = 5  # smaller unused symbols still count in dead-code LOC but get no issue
DEAD_HIGH_LINES = 20  # unused symbols at least this long are high: remove them before release
LARGE_FILE_MB = 5
LONG_FILE_LINES, WARN_FILE_LINES = 1000, 500  # source files longer than this should be split / are counted
DEFAULT_EXCLUDES = (
    "tests/**", "test/**", "**/test_*.py", "**/*_test.py", "**/conftest.py", "docs/**", ".cleanup/**",
)
# SlopCodeBench (arXiv 2603.24755): maintained human Python repos vs agent-written code. Only these two
# are comparable with scb-check 0.2.0; its Violation % and Clone Ratio are measured differently.
REFERENCE = {"verbosity": (0.15, 0.33), "erosion": (0.31, 0.68)}

SEVERITIES = ("critical", "high", "medium", "low")
# The one place severities are set: every issue the tools find names a row, and `report` prints this
# table into report.md, so the docs point here instead of restating it. Agents set the severity of the
# red flags and bugs they add.
POLICY = {
    "credential": ("critical", "credential matched by a confident rule, at HEAD or only in history"),
    "possible-credential": ("high", "possible credential (low-confidence rule or sensitive file name)"),
    "internal-head": ("high", "internal path, host, email or account at HEAD"),
    "large-file-head": ("high", f"file over {LARGE_FILE_MB} MB at HEAD"),
    "dead-large": ("high", f"unused function or class of {DEAD_HIGH_LINES}+ lines"),
    "license-missing": ("high", "no LICENSE file: nobody may use the code"),
    "license-mismatch": ("high", "pyproject.toml's license differs from the LICENSE file"),
    "license-third-party": ("high", "files under another license: keep their notices, check compatibility"),
    "readme-missing": ("high", "no README"),
    "deps-none": ("high", "no dependency list at all: nobody can install it"),
    "deps-undeclared": ("high", "an import that isn't a declared dependency, or only a dev one"),
    "vuln-fixable": ("high", "dependency with a known vulnerability and a fixed release"),
    "large-file-history": ("medium", f"file over {LARGE_FILE_MB} MB only in history: every clone downloads it"),
    "complex": ("medium", f"function with CC > {CC_THRESHOLD} and CC x sqrt(SLOC) >= {MEDIUM_MASS} "
                          "(e.g. CC 15 over 45 lines)"),
    "repetition": ("medium", f"duplicate group with lines x (copies - 1) >= {MEDIUM_DUP_LINES}"),
    "wrapper": ("medium", "trivial wrapper or shallow function"),
    "long-file": ("medium", f"source file over {LONG_FILE_LINES} lines: split it into modules"),
    "internal-history": ("low", "internal reference only in history: a fresh squashed history drops it"),
    "dead-small": ("low", f"unused function or class of {MIN_DEAD_LINES}-{DEAD_HIGH_LINES - 1} lines"),
    "complex-small": ("low", f"function with CC > {CC_THRESHOLD} and a smaller mass"),
    "repetition-small": ("low", f"smaller duplicate group of {MIN_CLONE_LINES}+ lines"),
    "inside-dead-code": ("low", "complexity or a red flag inside a dead-code candidate: delete that instead"),
    "deps-unused": ("low", "declared dependency that is never imported"),
    "deps-unreadable": ("low", "dependencies listed only where deptry can't read them (environment.yml, "
                               "setup.py, PEP 723)"),
    "vuln-unfixed": ("low", "known vulnerability without a fixed release"),
    "pins-none": ("low", "no lockfile or pinned requirements: can't audit, installs aren't reproducible"),
    "license-unrecognized": ("low", "LICENSE text not recognized"),
}
STATUSES = ("open", "fixed", "dismissed")
CATEGORIES = ("leak", "internal-ref", "large-file", "license", "vulnerable-dependency", "readiness", "dead-code",
              "complexity", "red-flag", "file-length", "bug")
ISSUE_DEFAULTS = {"source": "agent", "rule": "", "file": "", "line": None, "end_line": None, "symbol": "",
                  "evidence": {}, "fix": "", "done_when": "", "effort": "S", "status": "open", "note": ""}
AGENT_FIELDS = ("status", "severity", "fix", "done_when", "note", "effort")
SCRIPTS = Path(__file__).resolve().parent.parent  # holds cleanup.py, leak_rules.toml and unslop/
RULES = SCRIPTS / "leak_rules.toml"

# --- small helpers ---------------------------------------------------------------------------------


class ToolError(Exception):
    """A tool could not measure: it failed, or printed something this script doesn't recognize."""


class CleanupError(Exception):
    """Stops the command; main() prints the message."""


class Result(NamedTuple):
    """What a measure step found. A step with nothing to read sets `skipped`: it measured nothing, so its
    earlier issues stay open instead of being closed as fixed."""

    issues: list
    metrics: dict
    skipped: str = ""
    notes: dict | None = None  # metric -> why it is incomplete


def die(message: str):
    raise CleanupError(message)


def run(cmd, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True, **kwargs)


def git(repo: Path, *args: str) -> str:
    p = run(["git", "-C", repo, *args])
    if p.returncode:
        die(f"git {' '.join(args)} failed: {p.stderr.strip()}")
    return p.stdout


def repo_root(path: str) -> Path:
    p = run(["git", "-C", path, "rev-parse", "--show-toplevel"])
    if p.returncode:
        die(f"{path} is not inside a git repository")
    return Path(p.stdout.strip()).resolve()


def rel(base: Path, path) -> str:
    """`path` relative to `base`, as posix. A tracked symlink can point outside the repo, so the path as
    written is tried before the resolved one; a path outside `base` comes back unchanged."""
    p = Path(path)
    if not p.is_absolute():
        return p.as_posix()
    for candidate in (p, p.resolve()):
        try:
            return candidate.relative_to(base).as_posix()
        except ValueError:
            pass
    return p.as_posix()


def head(repo: Path) -> str:
    return git(repo, "rev-parse", "HEAD").strip()


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def uvx() -> str:
    exe = shutil.which("uvx")
    if not exe:
        die("uvx not found: install uv (https://docs.astral.sh/uv/)")
    return exe


def version_key(v: str) -> tuple:
    return tuple(int(n) for n in re.findall(r"\d+", v))
