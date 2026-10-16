"""Whether the repo root has a LICENSE file and a README."""

import re

from utils.git import git
from tools.base import Result, Tool

TOOL = Tool(
    name="license",
    policy={
        "license-missing": ("high", "no LICENSE file: nobody may use the code"),
        "readme-missing": ("high", "no README"),
    },
)


def measure(repo, files, index) -> Result:
    """An issue for a missing LICENSE file, and one for a missing README, at the repo root."""
    names = git(repo, "ls-tree", "--name-only", "HEAD").splitlines()
    out = []
    if not any(re.fullmatch(r"(?i)(?:licen[cs]e|copying)(?:\.(?:md|txt|rst))?", n) for n in names):
        out.append(TOOL.issue(
            "LIC", "license", "missing-license", "license-missing",
            "No LICENSE file: without one, nobody may legally use, copy or modify the code",
            key="LIC|missing", effort="S",
            fix=("Ask the user which license to use (MIT and Apache-2.0 are common for research code; "
                 "Apache-2.0 adds a patent grant). Add LICENSE at the repo root and set `license` in "
                 "pyproject.toml."),
            done_when="A LICENSE file is at the repo root."))
    if not any(re.fullmatch(r"(?i)readme(?:\.\w+)?", n) for n in names):
        out.append(TOOL.issue(
            "DOC", "readiness", "missing-readme", "readme-missing", "No README at the repo root",
            key="DOC|readme", effort="M",
            fix="Add a README: what the code does, how to install it, a minimal quickstart, how to cite it.",
            done_when="README exists and its quickstart runs."))
    return Result(out, {})
