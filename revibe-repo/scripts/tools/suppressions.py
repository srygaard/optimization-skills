"""Suppression comments (noqa, scbc ignore): they hide findings instead of fixing them."""

import re

from tools.base import Result, Tool

TOOL = Tool(name="suppressions", policy={})  # a scorecard count, without issues


def measure(scope) -> Result:
    """How many suppression comments the files in scope hold."""
    pattern = re.compile(r"#\s*(?:noqa|scbc\s+ignore)", re.I)
    count = sum(len(pattern.findall((scope.repo / f).read_text(errors="replace"))) for f in scope.files)
    return Result([], {"suppressions": count})
