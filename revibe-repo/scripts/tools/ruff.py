"""ruff, as `python -m ruff`: magic numbers in comparisons (PLR2004) and string paths where pathlib reads better (PTH).

Only comparisons count as magic: positional arguments and indices (`torch.randn(3, 224)`, `x[:, 3]`) are normal in
scientific code, while `if n_atoms > 38:` hides a threshold someone chose."""

import json
import re
import sys
from pathlib import Path

from utils.process import run
from tools.base import Metric, Result, Step, ToolError


class Ruff(Step):
    """Magic comparisons per function and as one count, and string-path calls as one count."""

    name = "ruff"
    package = "ruff"
    version = "0.15.7"
    policy = {
        "magic-function": "function or method comparing against {medium_magic}+ distinct unnamed numbers (PLR2004)",
        "magic-numbers": "other comparisons against unnamed numbers, as one issue",
        "string-paths": "paths built and opened as strings (os.path, open, glob) instead of pathlib (PTH), as one issue",
    }
    metrics = (
        Metric("magic_comparisons", "Magic comparisons", warn="comparisons against unnamed numbers",
               description="comparisons against unnamed numbers, such as `if n > 38:`"),
        Metric("string_paths", "String paths", warn="string path calls",
               description="os.path, open() and glob calls on string paths instead of pathlib"),
    )

    def measure(self, scope) -> Result:
        """Run ruff on the files in scope and turn its findings into issues and two counts."""
        if not scope.files:
            return Result([], {"magic_comparisons": 0, "string_paths": 0})
        magic, paths = self.grouped(scope)
        issues = [self.function_issue(file, qualname, found) for (file, qualname), found in magic.items()
                  if qualname != "<module>" and len({v for _, v in found}) >= self.settings["medium_magic"]]
        magic_total = sum(len(found) for found in magic.values())
        if magic_total:
            issues.append(self.magic_issue(magic, magic_total))
        if paths:
            issues.append(self.paths_issue(paths))
        return Result(issues, {"magic_comparisons": magic_total, "string_paths": sum(paths.values())})

    def grouped(self, scope) -> tuple:
        """ruff's findings in scope: ({(file, qualname): [(line, value)]} of magic comparisons, {file: count} of
        string-path calls)."""
        names = {(scope.repo / f).resolve(): f for f in scope.files}
        magic = {}
        paths = {}
        for d in self.findings(scope):
            file = names.get(Path(d["filename"]).resolve())
            if file is None:
                continue
            line = d["location"]["row"]
            if d["code"] == "PLR2004":
                value = re.search(r"`([^`]+)`", d["message"]).group(1)
                magic.setdefault((file, scope.qualname(file, line)), []).append((line, value))
            else:
                paths[file] = paths.get(file, 0) + 1
        return magic, paths

    def findings(self, scope) -> list:
        """ruff's JSON findings for the files in scope, with the repo's own ruff config ignored."""
        p = run([sys.executable, "-I", "-m", "ruff", "check", "--isolated", "--no-cache", "--exit-zero",
                 "--select", "PLR2004,PTH", "--output-format", "json", *scope.files], cwd=scope.repo)
        try:
            found = json.loads(p.stdout)
        except json.JSONDecodeError:
            raise ToolError(f"ruff failed: {p.stderr.strip()[-200:]}") from None
        return [d for d in found if d.get("code")]  # a file that doesn't parse has no code

    def function_issue(self, file: str, qualname: str, found: list) -> dict:
        distinct = sorted({v for _, v in found})
        shown = ", ".join(distinct[:6]) + (" ..." if len(distinct) > 6 else "")
        return self.issue(
            "MAGIC", "red-flag", "magic-numbers", "magic-function",
            f"`{qualname}` compares against {len(distinct)} unnamed numbers: {shown}", file=file, symbol=qualname,
            line=min(line for line, _ in found),
            evidence={"values": distinct[:20], "count": len(found), "weight": len(found)}, effort="S",
            fix=("Name each threshold where it is used: a parameter with a default, a config value, a local, or a "
                 "class attribute. Don't move them into module-level constants."),
            done_when="It compares against fewer unnamed numbers than the threshold, and the safety net passes.")

    def magic_issue(self, magic: dict, total: int) -> dict:
        per_file = {}
        for (file, _), found in magic.items():
            per_file[file] = per_file.get(file, 0) + len(found)
        return self.count_issue("MAGIC", "red-flag", "magic-numbers", "magic-numbers", per_file,
                                f"{total} comparisons against unnamed numbers",
                                fix=("Name the thresholds whose meaning isn't plain, the files with most first, the "
                                     "way the magic-function issues say. Dismiss with that note once only plain "
                                     "ones are left."))

    def paths_issue(self, paths: dict) -> dict:
        return self.count_issue("PATH", "pathlib", "string-paths", "string-paths", paths,
                                f"{sum(paths.values())} string path calls",
                                fix=("Build paths with pathlib (`root / \"data\" / name`) and use `.read_text()`, "
                                     "`.exists()`, `.glob()` and `.open()` in the code you touch anyway; don't "
                                     "rewrite untouched files just for this."))

    def count_issue(self, prefix: str, category: str, rule: str, policy: str, per_file: dict, what: str,
                    fix: str) -> dict:
        """One issue holding a count and the files with the most."""
        most = sorted(per_file.items(), key=lambda kv: (-kv[1], kv[0]))
        return self.issue(
            prefix, category, rule, policy,
            f"{what} in {len(per_file)} files, most in " + ", ".join(f"`{f}` ({n})" for f, n in most[:3]),
            key=f"{prefix}|count", evidence={"count": sum(per_file.values()), "files": dict(most[:20])},
            effort="M", fix=fix, done_when="Only deliberate ones are left (dismissed with a note).")
