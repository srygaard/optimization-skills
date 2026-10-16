"""The severity scale, the skill's own paths, and helpers shared by every command. Severities are set by
each tool's `TOOL` in tools/."""

import datetime as dt
import subprocess
import tomllib
from enum import IntEnum
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parent  # holds cleanup.py, the modules it imports, and tools/


class Severity(IntEnum):
    """How bad an issue is: a higher value is more severe. Members are named as issues.json spells them, most
    severe first."""

    critical = 4
    high = 3
    medium = 2
    low = 1


# --- small helpers ---------------------------------------------------------------------------------


def run(cmd, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True, **kwargs)


def git(repo: Path, *args: str) -> str:
    p = run(["git", "-C", repo, *args])
    if p.returncode:
        raise SystemExit(f"git {' '.join(args)} failed: {p.stderr.strip()}")
    return p.stdout


def repo_root(path: str) -> Path:
    p = run(["git", "-C", path, "rev-parse", "--show-toplevel"])
    if p.returncode:
        raise SystemExit(f"{path} is not inside a git repository")
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


def pyproject(repo: Path) -> dict:
    try:
        return tomllib.loads((repo / "pyproject.toml").read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return {}
