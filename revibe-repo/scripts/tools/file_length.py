"""Source file length, from the scope's index."""

from tools.base import Metric, Result, Step


class FileLength(Step):
    """How long the source files are, and which to split."""

    name = "file-length"
    policy = {
        "long-file": "source file over {long_file_lines} lines: split it into modules",
    }
    metrics = (
        Metric("avg_file_lines", "Avg file length"),
        Metric("files_over_500", "Files > {warn_file_lines} lines"),
        Metric("files_over_1000", "Files > {long_file_lines} lines", warn="files over {long_file_lines} lines"),
    )

    def measure(self, scope) -> Result:
        """Average length of the source files, how many pass 500 and 1000 lines, and an issue for the latter."""
        long_file = self.settings["long_file_lines"]
        lines = scope.lines
        out = [self.issue(
            "SIZE", "file-length", "long-file", "long-file", f"`{f}` has {n:,} lines",
            file=f, evidence={"lines": n, "weight": n}, effort="M",
            fix=("Split it into modules of a few hundred lines, each owning one concept (what it hides, not when it "
                 "runs). Move the code unchanged in its own commit, then refactor; if other code imports from this "
                 "module, keep that import path working."),
            done_when=f"It is under {long_file} lines, and the import test and safety net pass.")
            for f, n in sorted(lines.items()) if n > long_file]
        return Result(out, {"avg_file_lines": round(sum(lines.values()) / len(lines)) if lines else 0,
                            "files_over_500": sum(n > self.settings["warn_file_lines"] for n in lines.values()),
                            "files_over_1000": len(out)})
