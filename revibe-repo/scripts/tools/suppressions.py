"""Suppression comments (noqa, scbc ignore): they hide findings instead of fixing them."""

import io
import re
import tokenize

from tools.base import Metric, Result, Step


class Suppressions(Step):
    """A scorecard count, without issues."""

    name = "suppressions"
    metrics = (Metric("suppressions", "Suppressions", warn="suppressions",
                      description="`# noqa` and `# scbc ignore` comments: findings hidden, not fixed"),)

    def measure(self, scope) -> Result:
        """How many suppression comments the files in scope hold: comments only, not text in strings."""
        pattern = re.compile(r"#\s*(?:noqa|scbc\s+ignore)", re.I)
        count = 0
        for f in scope.defs:  # the files that parse
            source = (scope.repo / f).read_text(errors="replace")
            count += sum(len(pattern.findall(c)) for c in comments(source))
        return Result([], {"suppressions": count})


def comments(text: str) -> list:
    """The comments in a module's source."""
    return [t.string for t in tokenize.generate_tokens(io.StringIO(text).readline) if t.type == tokenize.COMMENT]
