"""git in the repo under cleanup."""

from pathlib import Path

from utils.process import run


def git(repo: Path, *args: str) -> str:
    """The output of `git -C repo args`; stops the command if git fails."""
    p = run(["git", "-C", repo, *args])
    if p.returncode:
        raise SystemExit(f"git {' '.join(args)} failed: {p.stderr.strip()}")
    return p.stdout


def repo_root(path: str) -> Path:
    """The top of the git repository that holds `path`."""
    p = run(["git", "-C", path, "rev-parse", "--show-toplevel"])
    if p.returncode:
        raise SystemExit(f"{path} is not inside a git repository")
    return Path(p.stdout.strip()).resolve()


def head(repo: Path) -> str:
    """The commit HEAD points to."""
    return git(repo, "rev-parse", "HEAD").strip()
