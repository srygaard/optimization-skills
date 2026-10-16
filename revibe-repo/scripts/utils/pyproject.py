"""The repo's pyproject.toml."""

import tomllib
from pathlib import Path


def read_pyproject(repo: Path) -> dict:
    """The repo's pyproject.toml, or {} when it is missing or broken."""
    try:
        return tomllib.loads((repo / "pyproject.toml").read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return {}
