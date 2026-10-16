"""Large files anywhere in the git history, from git itself."""

from pathlib import Path

from ..common import git, run
from .base import Tool

TOOL = Tool(
    name="git-history",
    policy={
        "large-file-head": ("high", "file over {large_file_mb} MB at HEAD"),
        "large-file-history": ("medium", "file over {large_file_mb} MB only in history: every clone downloads it"),
    },
    settings={"large_file_mb": 5},
)


def scan(repo: Path) -> list:
    paths = {}
    for row in git(repo, "rev-list", "--objects", "--all").splitlines():
        sha, _, path = row.partition(" ")
        if path:
            paths.setdefault(sha, path)
    p = run(["git", "-C", repo, "cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)"],
            input="\n".join(paths) + "\n")
    at_head = {}  # blob -> its path at HEAD: compare blobs, not paths, so a moved file is still at HEAD
    for row in git(repo, "ls-tree", "-r", "-z", "HEAD").split("\0"):
        if row:
            meta, path = row.split("\t", 1)
            at_head[meta.split()[2]] = path
    sizes = {}  # path -> (largest size, whether a large blob of it is at HEAD)
    for row in p.stdout.splitlines():
        sha, kind, size = (row.split() + ["", ""])[:3]
        if kind == "blob" and int(size) > TOOL.settings["large_file_mb"] * 1_000_000:
            path = at_head.get(sha) or paths[sha]
            old_size, old_here = sizes.get(path, (0, False))
            sizes[path] = (max(old_size, int(size)), old_here or sha in at_head)
    out = []
    for path, (size, here) in sorted(sizes.items(), key=lambda kv: -kv[1][0]):
        out.append(TOOL.issue(
            "FILE", "large-file", "large-file", "large-file-head" if here else "large-file-history",
            f"{size / 1e6:.1f} MB file `{path}` ({'at HEAD' if here else 'only in history'})",
            file=path, evidence={"bytes": size, "at_head": here}, effort="M",
            fix=("Take it out of the repo: host weights or data elsewhere (release asset, Hugging Face Hub, "
                 "Zenodo) and download on demand; check that you may redistribute it." if here else
                 "Only matters if the history is published; a fresh squashed history drops it."),
            done_when="Not in the tree, and not in any history that will be published."))
    return out
