"""Source file length in code lines, from the scope's index."""

from tools.base import Metric, Result, Step


class FileLength(Step):
    """How long the source files are, in code lines (comments, docstrings and imports aside), and which to split."""

    name = "file-length"
    policy = {
        "long-file": "source file over {long_file_lines} code lines (comments, docstrings and imports aside): "
                     "split it into modules",
    }
    metrics = (
        Metric("avg_code_lines", "Avg code lines per file"),
        Metric("files_over_500_code_lines", "Files > {warn_file_lines} code lines"),
        Metric("files_over_1000_code_lines", "Files > {long_file_lines} code lines",
               warn="files over {long_file_lines} code lines"),
    )

    def measure(self, scope) -> Result:
        """Average code lines per source file, how many files pass 500 and 1000, and an issue for the latter."""
        long_file = self.settings["long_file_lines"]
        warn = self.settings["warn_file_lines"]
        lines = scope.code_lines
        out = [self.issue(
            "SIZE", "file-length", "long-file", "long-file", f"`{f}` has {n:,} code lines",
            file=f, evidence={"code_lines": n, "weight": n}, effort="M",
            fix=("Split it into modules of a few hundred lines, each owning one concept (what it hides, not when it "
                 "runs). Move the code unchanged in its own commit, then refactor; if other code imports from this "
                 "module, keep that import path working."),
            done_when=f"It is under {long_file} code lines, and the import test and safety net pass.")
            for f, n in sorted(lines.items()) if n > long_file]
        return Result(out, {"avg_code_lines": round(sum(lines.values()) / len(lines)) if lines else 0,
                            "files_over_500_code_lines": sum(n > warn for n in lines.values()),
                            "files_over_1000_code_lines": len(out)})
