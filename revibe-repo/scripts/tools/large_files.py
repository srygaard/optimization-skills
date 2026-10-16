"""Large files in the repo at HEAD: checkpoints, datasets and other blobs that shouldn't ship."""

from utils.git import git
from tools.base import Result, Tool

TOOL = Tool(
    name="large-files",
    policy={
        "large-file": ("high", "file over {large_file_mb} MB at HEAD"),
    },
    settings={"large_file_mb": 5},
)


def measure(scope) -> Result:
    """One issue per file in the HEAD tree over the size limit, whether or not it is in scope."""
    limit = TOOL.settings["large_file_mb"] * 1_000_000
    out = []
    for row in git(scope.repo, "ls-tree", "-r", "-l", "-z", "HEAD").split("\0"):
        if not row:
            continue
        meta, path = row.split("\t", 1)
        size = meta.split()[3]  # "-" for a submodule
        if size == "-" or int(size) <= limit:
            continue
        out.append(TOOL.issue(
            "FILE", "large-file", "large-file", "large-file", f"{int(size) / 1e6:.1f} MB file `{path}`",
            file=path, evidence={"bytes": int(size)}, effort="M",
            fix=("Take it out of the repo: host weights or data elsewhere (release asset, Hugging Face Hub, "
                 "Zenodo) and download on demand; check that you may redistribute it."),
            done_when="Not in the tree, and not in any history that will be published."))
    return Result(out, {})
