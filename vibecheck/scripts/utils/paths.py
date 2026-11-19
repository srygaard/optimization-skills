"""Paths as issues.json writes them."""

from pathlib import Path


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
