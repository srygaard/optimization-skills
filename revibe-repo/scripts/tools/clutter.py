"""Files at HEAD that shouldn't ship: generated outputs at the repo root, scratch copies and working notes."""

import re
from pathlib import Path, PurePosixPath

from utils.git import git
from tools.base import Result, Step


class Clutter(Step):
    """Tracked files that are by-products of the work, not part of the project."""

    name = "clutter"
    policy = {
        "clutter-output": "generated output at the repo root (images, molecules, logs, arrays) the README doesn't show",
        "clutter-scratch": "scratch and backup files: Untitled notebooks, copies (`x copy.py`, `x_copy.py`), `_old` "
                           "and `.bak` files, old/ and tmp/ folders",
        "clutter-notes": "working notes at the repo root: plans, proposals, TODO and notes files",
    }
    output_suffixes = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".pdf", ".log", ".out", ".err", ".sdf", ".mol",
                       ".mol2", ".pdb", ".xyz", ".pml", ".npy", ".npz", ".pt", ".pth", ".ckpt", ".pkl", ".pickle",
                       ".h5"}
    scratch = re.compile(r"(?i)(?:^|/)untitled[^/]*$|[ _-]copy(?: ?\d+)?\.[^/.]+$|_old\.[^/.]+$|\.(?:bak|orig|swp)$"
                         r"|~$|(?:^|/)(?:old|\w+_old|backup|tmp)/")
    notes = re.compile(r"(?i)^[^/]*(?:plan|proposal|todo)[^/]*\.(?:md|txt)$|^notes?\.(?:md|txt)$")
    # kind -> (what its summary calls the files, fix)
    advice = {
        "clutter-output": ("generated files at the repo root",
                           "`git rm` them and add their patterns to .gitignore. Keep a figure the README should show "
                           "under docs/ and link it."),
        "clutter-scratch": ("scratch or backup files",
                            "Delete them, after checking with the user that none is the only copy of something still "
                            "used."),
        "clutter-notes": ("working notes at the repo root",
                          "Ask the user which ship. Delete the rest, and move what helps users into the README or "
                          "docs/."),
    }

    def measure(self, scope) -> Result:
        """One issue per kind of clutter, listing its files."""
        files = git(scope.repo, "ls-tree", "-r", "--name-only", "-z", "HEAD").strip("\0").split("\0")
        readme = readme_text(scope.repo, files)
        found = {kind: [] for kind in self.policy}
        for f in files:
            if kind := self.kind(f, readme):
                found[kind].append(f)
        return Result([self.clutter_issue(kind, paths) for kind, paths in found.items() if paths], {})

    def kind(self, file: str, readme: str) -> str:
        """The kind of clutter `file` is, or "" when it is none."""
        if self.scratch.search(file):
            return "clutter-scratch"
        if "/" in file:
            return ""
        if self.notes.search(file):
            return "clutter-notes"
        if PurePosixPath(file).suffix.lower() in self.output_suffixes and file not in readme:
            return "clutter-output"
        return ""

    def clutter_issue(self, kind: str, paths: list) -> dict:
        label, fix = self.advice[kind]
        shown = ", ".join(f"`{f}`" for f in paths[:3]) + (f" and {len(paths) - 3} more" if len(paths) > 3 else "")
        return self.issue("CLUT", "readiness", kind, kind, f"{len(paths)} {label}: {shown}", key=f"CLUT|{kind}",
                          evidence={"files": paths}, effort="S", fix=fix,
                          done_when="`cleanup.py measure` no longer lists them.")


def readme_text(repo: Path, files: list) -> str:
    """The text of the READMEs at the repo root, where a figure that belongs in the repo is shown."""
    return " ".join((repo / f).read_text(errors="replace") for f in files if f.lower().startswith("readme"))
