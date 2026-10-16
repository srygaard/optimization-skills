"""vulture, in process: unused functions and classes that nothing else names, and modules nothing imports."""

import ast
import json
import re
import warnings
from pathlib import Path

from utils.git import git
from utils.pyproject import read_pyproject
from tools.base import Metric, Result, Step


class Vulture(Step):
    """Unused functions, classes and modules, checked against every file that could name them."""

    name = "vulture"
    package = "vulture"
    version = "2.16"
    policy = {
        "dead-large": "unused function, class or module of {dead_high_lines}+ lines",
        "dead-small": "unused function, class or module of {min_dead_lines}+ lines, under {dead_high_lines}",
        # set by `adjust` on other tools' issues
        "inside-dead-code": "complexity or a red flag inside a dead-code candidate: delete that instead",
    }
    metrics = (Metric("dead_code_loc", "Dead code LOC"),)

    def dead_issue(self, c: dict, with_coverage: bool) -> dict:
        """Open, or dismissed when something names the symbol or its body ran."""
        item = c["item"]
        size = c["lines"]
        note = (f"referenced by name in {', '.join(c['hits'][:3])}" if c["hits"] else
                "executed in the coverage run" if c["ran"] else "")
        return self.issue(
            "DEAD", "dead-code", c["rule"], "dead-large" if size >= self.settings["dead_high_lines"] else "dead-small",
            f"{item.message} ({size} lines, vulture {item.confidence}% confidence)",
            file=c["file"], symbol=c["symbol"], line=c["line"], end_line=c["line"] + size - 1,
            evidence={"lines": size, "confidence": item.confidence, "referenced_in": c["hits"][:5],
                      "mentioned_in": c["mentions"][:5], "executed": c["ran"] if with_coverage else None},
            status="dismissed" if note else "open", note=note, effort="S",
            fix=("Confirm nothing references it (yaml `_target_`/`target:` strings, registries, getattr, "
                 "notebooks, shell scripts; files that mention its bare name are in `evidence.mentioned_in`), "
                 "then delete it."),
            done_when="Deleted, and the import test and safety net pass.")

    def measure(self, scope) -> Result:
        import vulture  # the analyzer, not this module: tools/ is not on sys.path

        if not scope.files:
            return Result([], {"dead_code_loc": 0})
        config = read_pyproject(scope.repo).get("tool", {}).get("vulture", {})  # what the CLI would read from the repo
        v = vulture.Vulture(ignore_names=config.get("ignore_names", []), ignore_decorators=config.get("ignore_decorators", []))
        v.scavenge([scope.repo / f for f in scope.files], exclude=config.get("exclude", []))
        # vulture resolves paths, and a tracked symlink can resolve outside the repo
        names = {(scope.repo / f).resolve(): f for f in scope.files}
        refs = _references(scope.repo, scope.files)
        coverage = executed_lines(scope.repo)
        modules = unimported_modules(scope, refs)
        found = []
        for item in v.get_unused_code(min_confidence=self.settings["min_confidence"], sort_by_size=True):
            if file := reported_file(item, names, modules):
                found.append(dead_candidate(item, file, scope, refs, coverage))
        total = sum(c["lines"] for c in found if c["unused"]) + sum(modules.values())
        issues = [self.dead_issue(c, bool(coverage)) for c in found if c["lines"] >= self.settings["min_dead_lines"]]
        issues += [self.module_issue(file, size) for file, size in modules.items()
                   if size >= self.settings["min_dead_lines"]]
        return Result(issues, {"dead_code_loc": total})

    def module_issue(self, file: str, size: int) -> dict:
        """A module that nothing imports or runs."""
        return self.issue(
            "DEAD", "dead-code", "unused-module", "dead-large" if size >= self.settings["dead_high_lines"] else "dead-small",
            f"module `{file}` is never imported, run or named elsewhere ({size} lines)",
            file=file, line=1, end_line=size, evidence={"lines": size}, effort="S",
            fix=("Confirm nothing loads it (importlib with a built name, a launcher outside the repo, a script that "
                 "reproduces a published result), then delete the file."),
            done_when="Deleted, and the import test and safety net pass.")

    def adjust(self, found: list):
        """Complexity and red-flag issues inside a dead-code candidate drop to low: delete before refactoring."""
        dead = [i for i in found if i["category"] == "dead-code" and i["status"] == "open"]
        for i in found:
            if i["category"] not in ("complexity", "red-flag") or not i.get("line"):
                continue
            host = next((d for d in dead if d["file"] == i["file"] and d["line"] <= i["line"] <= d["end_line"]), None)
            if host:
                i["severity"] = self.severity["inside-dead-code"]
                i["evidence"]["inside_dead_code"] = host["id"]
                i["fix"] = f"Inside dead-code candidate {host['id']}: delete that instead, unless it turns out to be used. " + i["fix"]


def _references(repo: Path, shipped: list) -> list:
    """(path, text, strings_only) of tracked files that can name code without importing it.

    Configs, scripts, notebooks and docs count wholesale, as do Python files outside the scope (tests). In the
    shipped Python files only string literals name a symbol, e.g. {"target": "main.SetupCallback"}; their whole
    text names a module.
    """
    text_suffixes = {".yaml", ".yml", ".json", ".toml", ".cfg", ".ini", ".sh", ".bash", ".slurm", ".sbatch",
                     ".ipynb", ".jinja", ".j2", ".md", ".rst"}
    out = []
    shipped = set(shipped)
    for rel in git(repo, "ls-files", "-z").split("\0"):
        path = repo / rel
        suffix = Path(rel).suffix
        if not rel or not path.is_file() or path.stat().st_size > 5_000_000:
            continue
        if suffix in text_suffixes or (suffix == ".py" and rel not in shipped):
            out.append((rel, path.read_text(errors="replace"), False))
        elif rel in shipped:
            out.append((rel, path.read_text(errors="replace"), True))
    return out


def _named_in(refs: list, name: str, modules: list) -> tuple:
    """(files that name it, files that only mention it). Naming takes its module or a package above it:
    `module.name` or `module:name` (a yaml `target:`, an entry point, a "main.SetupCallback" string in the
    code) or `from ...module import name`. A bare `.name` or "name" (`model.train()`, `split="train"`) is only a
    mention, left for review. Dunders are never references."""
    if name.startswith("__") and name.endswith("__"):
        return [], []
    # `modules`: the defining module and its packages, since `pkg/__init__.py` often re-exports a class that
    # configs then name as `pkg.Name`
    n = re.escape(name)
    m = "(?:" + "|".join(map(re.escape, modules)) + ")"
    qualified = re.compile(rf"\b{m}[.:]{n}\b|\bfrom\s+[\w.]*\b{m}\s+import\s*(?:\([^)]*|[^\n]*)\b{n}\b")
    in_string = re.compile(rf"[\"'][^\"'\n]*\b{m}[.:]{n}[\"']")
    bare = re.compile(rf"\b{n}\b")
    bare_string = re.compile(rf"[\"']{n}[\"']")
    named = []
    mentioned = []
    for rel, text, strings_only in refs:
        if (in_string if strings_only else qualified).search(text):
            named.append(rel)
        elif (bare_string if strings_only else bare).search(text):
            mentioned.append(rel)
    return named, mentioned


def reported_file(item, names: dict, modules: dict) -> str | None:
    """The file in scope of a vulture item worth an issue, or None. Methods and properties are skipped: in ML code
    they are mostly framework hooks (Lightning's training_step, nn.Module overrides) that vulture can't see being
    called. A redundant if-condition is an `if True:` whose body runs. A module nobody imports is one issue of its
    own, not one per definition in it."""
    file = names.get(Path(item.filename))  # None: one of vulture's bundled whitelists
    if file is None or file in modules or item.message == "redundant if-condition":
        return None
    return None if item.typ in ("method", "property", "variable", "attribute", "import") else file


def unimported_modules(scope, refs: list) -> dict:
    """{file: lines} of the modules in scope whose name no other tracked file mentions, leaving out scripts (a
    `__main__` guard or work done at the top level) and modules Python or tools load by name."""
    loaded_by_name = {"__init__", "__main__", "setup", "conftest", "sitecustomize", "usercustomize", "noxfile"}
    out = {}
    for file in scope.files:
        stem = Path(file).stem
        if stem in loaded_by_name or file not in scope.defs:
            continue
        text = (scope.repo / file).read_text(errors="replace")
        if "__main__" in text or _runs_on_import(text):
            continue
        named = re.compile(rf"\b{re.escape(stem)}\b")
        if not any(rel != file and named.search(other) for rel, other, _ in refs):
            out[file] = len(text.splitlines())
    return out


def _runs_on_import(text: str) -> bool:
    """Whether a module does work at the top level, as a script run with `python file.py` does."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        body = ast.parse(text).body
    definitions = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Assign,
                   ast.AnnAssign, ast.Pass)
    for node in body:
        if isinstance(node, definitions):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):  # a docstring
            continue
        if isinstance(node, (ast.If, ast.Try)):  # `if TYPE_CHECKING:` or `try: import x`
            continue
        return True
    return False


def runtime_lines(scope, file: str, line: int) -> set:
    """Lines that only run when the code is used: a function's body, or the bodies of a class's methods.
    Def, decorator and class-body lines run on import, so coverage of them says nothing."""
    d = scope.at_def_line(file, line)
    if not d:
        return set()
    bodies = [d] if d.kind == "function" else [
        x for x in scope.defs.get(file, ()) if x.kind == "function" and d.start <= x.start and x.end <= d.end]
    return {n for b in bodies for n in range(b.def_line + 1, b.end + 1)}


def executed_lines(repo: Path) -> dict:
    """{file: lines that ran}, from an optional coverage.py JSON report at the repo root (from any run)."""
    if not (cov := repo / "coverage.json").exists():
        return {}
    return {Path(name).as_posix(): set(f.get("executed_lines", []))
            for name, f in json.loads(cov.read_text()).get("files", {}).items()}


def dead_candidate(item, file: str, scope, refs: list, coverage: dict) -> dict:
    """A vulture item in `file` with what decides it: the symbol, the files that name or mention it, whether
    it ran. `unused` (nothing names it and it didn't run) is what dead-code LOC counts."""
    line = item.first_lineno
    unreachable = item.typ == "unreachable_code"
    d = scope.at_def_line(file, line)
    if unreachable:
        qual = scope.qualname(file, line)
    else:
        qual = d.qualname if d and d.qualname.endswith(item.name) else item.name
    modules = [p for p in Path(file).with_suffix("").parts[::-1] if p != "__init__"]
    # unreachable code can't be referenced
    hits, mentions = ([], []) if unreachable else _named_in(refs, qual.rsplit(".", 1)[-1], modules)
    ran = bool(coverage.get(file, set()) & runtime_lines(scope, file, line))
    return {"item": item, "rule": "unreachable" if unreachable else item.typ, "file": file, "line": line,
            "lines": item.size, "symbol": qual, "hits": hits, "mentions": mentions, "ran": ran,
            "unused": not hits and not ran}
