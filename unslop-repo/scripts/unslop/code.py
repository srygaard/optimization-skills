"""The code in scope and the analyzers that read it: scb-check (complexity, duplication) and vulture."""

import ast
import json
import math
import os
import re
import shutil
import tempfile
from fnmatch import fnmatch
from pathlib import Path

from .common import (
    CC_THRESHOLD, DEAD_HIGH_LINES, MEDIUM_DUP_LINES, MEDIUM_MASS, MIN_CLONE_LINES, MIN_DEAD_LINES, Result,
    TOOLS, ToolError, git, rel, run, uvx,
)
from .state import issue


# --- code index ------------------------------------------------------------------------------------


def matches(rel: str, pattern: str) -> bool:
    """scb-check's exclude semantics: '/'-separated fnmatch segments, '**' spans directories."""
    def seg(pp, pt):
        if not pt:
            return not pp
        if pt[0] == "**":
            return any(seg(pp[i:], pt[1:]) for i in range(len(pp) + 1))
        return bool(pp) and fnmatch(pp[0], pt[0]) and seg(pp[1:], pt[1:])
    return seg(tuple(p for p in rel.split("/") if p), tuple(p for p in pattern.split("/") if p))


def shipped_files(repo: Path, excludes: list) -> list:
    rels = git(repo, "ls-files", "-z", "--", "*.py").split("\0")
    return [r for r in rels if r and (repo / r).is_file() and not any(matches(r, g) for g in excludes)]


class CodeIndex:
    """Functions and classes per file (qualname, span), the lines that hold code, and each file's length."""

    def __init__(self, repo: Path, files: list):
        self.defs = {}
        self.code = {}
        self.lines = {}
        self.syntax_errors = []
        for rel in files:
            text = (repo / rel).read_text(errors="replace")
            self.lines[rel] = len(text.splitlines())
            self.code[rel] = {n for n, s in enumerate(text.splitlines(), 1)
                              if s.strip() and not s.strip().startswith("#")}
            try:
                tree = ast.parse(text)
            except SyntaxError:
                self.syntax_errors.append(rel)
                continue
            defs = self.defs[rel] = []

            def walk(node, prefix):
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        start = min([child.lineno] + [x.lineno for x in child.decorator_list])
                        defs.append((start, child.end_lineno, prefix + child.name, child.lineno,
                                     "class" if isinstance(child, ast.ClassDef) else "function"))
                        walk(child, prefix + child.name + ".")
                    else:
                        walk(child, prefix)

            walk(tree, "")

    def enclosing(self, rel: str, line: int):
        """Innermost def/class containing `line`: (start, end, qualname, def_line, kind) or None."""
        best = None
        for d in self.defs.get(rel, ()):
            if d[0] <= line <= d[1] and (best is None or d[0] >= best[0]):
                best = d
        return best

    def at_def_line(self, rel: str, line: int):
        return next((d for d in self.defs.get(rel, ()) if d[3] == line), None) or self.enclosing(rel, line)

    def sloc(self, rel: str, start: int, end: int) -> int:
        return sum(start <= n <= end for n in self.code.get(rel, ()))

    def qualname(self, rel: str, line: int) -> str:
        d = self.enclosing(rel, line)
        return d[2] if d else "<module>"


# --- measure: scb-check ----------------------------------------------------------------------------

SCB_HEADER = re.compile(r"^(?:warning\[(?P<a>[\w-]+)\]|(?P<b>[\w-]+)\[warning\]|(?P<c>[\w-]+)):\s+(?P<msg>.+)$")
SCB_LOCATION = re.compile(r"┌─\s+(?P<path>.+?):(?P<line>\d+)(?::\d+)?\s*$")
SCB_CLONE = re.compile(r"duplicated block \((\d+) lines?, (\d+) instances?\)")
SCB_NOTE = re.compile(r"^\s*= (?P<note>.+)$")
SCB_CC = re.compile(r"complexity: (\d+), sloc: (\d+)")
SCB_FUNCTION = re.compile(r"function `([^`]+)`")


def parse_scb(text: str) -> list:
    findings = []
    for row in text.splitlines():
        if m := SCB_HEADER.match(row):
            findings.append({"rule": m["a"] or m["b"] or m["c"], "msg": m["msg"], "locs": [], "notes": []})
        elif (m := SCB_LOCATION.search(row)) and findings:
            findings[-1]["locs"].append((m["path"], int(m["line"])))
        elif (m := SCB_NOTE.match(row)) and findings:
            findings[-1]["notes"].append(m["note"])
    return findings


def scb_check(repo, files, index) -> Result:
    with tempfile.TemporaryDirectory(prefix="cleanup-scb-") as tmp:
        # scb-check walks a directory, so it would also read untracked files and other languages: give it
        # a copy of exactly the files in scope
        src = Path(tmp, "src").resolve()
        src.mkdir()
        for f in files:
            (src / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(repo / f, src / f)
        config = Path(tmp, "scb-check.toml")
        config.write_text("")
        shim = Path(tmp, "bin", "sg")  # a failing `sg` first on PATH makes scb-check use its bundled ast-grep
        shim.parent.mkdir()
        shim.write_text("#!/bin/sh\nexit 1\n")
        shim.chmod(0o755)
        env = {**os.environ, "PATH": f"{shim.parent}{os.pathsep}{os.environ.get('PATH', '')}"}
        base = [uvx(), f"scb-check@{TOOLS['scb-check']}", "check", src, "--config", config]
        p = run([*base, "--output-format", "json"], env=env)
        h = run([*base, "--min-duplicate-lines", MIN_CLONE_LINES], env=env)
        for q in (p, h):
            if q.returncode not in (0, 1):  # 1 only means findings
                raise ToolError(q.stderr.strip()[-300:])
        m = json.loads(p.stdout)
        out = scb_issues(h.stdout, src, index, m["high_cc_functions"])
    loc = m["total_loc"] or 1
    metrics = dict(
        loc=m["total_loc"], cc_over_10=m["high_cc_functions"], verbosity=round(m["verbosity"], 4),
        erosion=round(m["erosion"], 4), cog_erosion=round(m["cog_erosion"], 4),
        violation_pct=round(m["ast_grep_flagged_loc"] / loc, 4), clone_ratio=round(m["clone_loc"] / loc, 4),
        # absolute amounts behind the ratios: deleting clean code raises a ratio without adding slop
        verbosity_flagged_loc=m["verbosity_flagged_loc"], high_cc_mass=round(m["high_cc_mass"], 1))
    # no ast-grep binary worked: the slop rules silently counted nothing
    return Result(out, metrics, notes={"violation_pct": "ast-grep did not run"} if "ast-grep" in p.stderr else None)


LOCATED = ("erosion", "duplicate-structure", "trivial-wrapper", "low-use-short-function")
WRAPPER_FLAGS = {"trivial-wrapper": "pass-through-method", "low-use-short-function": "shallow-module"}


def scb_issues(text: str, src: Path, index, high_cc: int) -> list:
    """Trivial wrappers, duplicate blocks and complex functions from scb-check's human output. Raises
    ToolError when it doesn't parse as expected, so a changed format can't read as "all fixed"."""
    findings = parse_scb(text)
    for f in findings:
        f["locs"] = [(rel(src, path), line) for path, line in f["locs"]]
        if not f["locs"] and f["rule"] in LOCATED:
            raise ToolError(f"no location parsed for {f['rule']}: {f['msg'][:80]}")
    complex_fns = complexity_issues(findings, index)
    if len(complex_fns) != high_cc:
        raise ToolError(f"output not recognized: its JSON counts {high_cc} functions with CC > {CC_THRESHOLD}, "
                        f"{len(complex_fns)} were parsed")
    return wrapper_issues(findings, index) + repetition_issues(findings, index) + complex_fns


def wrapper_issues(findings: list, index) -> list:
    """Ousterhout's pass-through methods and shallow modules, from scb-check's wrapper rules."""
    out = []
    for f in findings:
        if f["rule"] not in WRAPPER_FLAGS:
            continue
        file, line = f["locs"][0]
        out.append(issue(
            "RF", "scb-check", "red-flag", WRAPPER_FLAGS[f["rule"]], "wrapper", f"{f['rule']}: {f['msg']}",
            file=file, symbol=(index.at_def_line(file, line) or (0, 0, "<module>"))[2], line=line,
            evidence={"scb_rule": f["rule"]}, effort="S",
            fix=("Inline it into its callers or remove it. If it is deliberate (a public alias, or an "
                 "override such as `model.train = disabled_train`), dismiss the issue with that reason."),
            done_when="scb-check no longer reports it (or it is dismissed), and the safety net passes."))
    return out


def repetition_issues(findings: list, index) -> list:
    """One issue per duplicate group, named after the functions that hold the copies."""
    out = []
    for f in findings:
        if f["rule"] != "duplicate-structure" or not (c := SCB_CLONE.search(f["msg"])):
            continue
        lines = int(c[1])
        count = int(c[2])
        locs = f["locs"]
        weight = lines * (count - 1)
        members = sorted({(file, index.qualname(file, line)) for file, line in locs})
        places = [f"{file}:{line}" for file, line in locs]
        out.append(issue(
            "RF", "scb-check", "red-flag", "repetition",
            "repetition" if weight >= MEDIUM_DUP_LINES else "repetition-small",
            f"{lines}-line block duplicated {count}x: {', '.join(places[:4])}",
            key="RF|repetition|" + "|".join(f"{f}:{q}" for f, q in members),
            file=locs[0][0], line=locs[0][1], symbol=", ".join(q for _, q in members),
            evidence={"lines": lines, "instances": count, "weight": weight, "locations": places},
            effort="M" if lines >= 30 else "S",
            fix=("Extract the shared block into one function, or delete the copy that is no longer used. "
                 "Pass the real differences as a few parameters."),
            done_when="scb-check no longer reports this duplicate group and the safety net passes."))
    return out


def _sized(findings: list, rule: str) -> list:
    """(finding, complexity, SLOC) for each `rule` finding whose note gives the function's size."""
    out = []
    for f in findings:
        m = next((m for n in f["notes"] if (m := SCB_CC.search(n))), None)
        if f["rule"] == rule and m:
            out.append((f, int(m[1]), int(m[2])))
    return out


def complexity_issues(findings: list, index) -> list:
    """One issue per function over the CC threshold, with its cognitive complexity as evidence."""
    cognitive = {f["locs"][0]: cc for f, cc, _ in _sized(findings, "cog_erosion")}
    out = []
    for f, cc, sloc in _sized(findings, "erosion"):
        file, line = f["locs"][0]
        d = index.at_def_line(file, line)
        name = SCB_FUNCTION.search(f["msg"])
        qual, start, end = (d[2], d[0], d[1]) if d else (name[1] if name else "<function>", line, line)
        mass = cc * math.sqrt(sloc)
        evidence = {"cc": cc, "sloc": sloc, "mass": round(mass, 1), "weight": round(mass, 1)}
        if (file, line) in cognitive:
            evidence["cognitive"] = cognitive[file, line]
        out.append(issue(
            "CX", "scb-check", "complexity", "cc", "complex" if mass >= MEDIUM_MASS else "complex-small",
            f"`{qual}` has cyclomatic complexity {cc} ({sloc} SLOC)",
            file=file, symbol=qual, line=start, end_line=end, evidence=evidence,
            effort="L" if sloc > 150 else "M",
            fix=("Simplify the control flow: drop dead branches and flags, return early, move special cases "
                 "out of the general path. Extracting a step with its own clear purpose is fine; moving "
                 "branches into a helper that only this function calls, with most of its arguments, is not. "
                 "If every branch is needed and it reads clearly, dismiss it with that reason. Add a "
                 "characterization test first."),
            done_when=(f"scb-check no longer lists it (CC <= {CC_THRESHOLD}), or it is gone or dismissed with "
                       "a reason, and the safety net passes.")))
    return out


# --- measure: vulture ----------------------------------------------------------------------------

VULTURE = re.compile(
    r"^(?P<path>.+?):(?P<line>\d+): (?P<msg>unused (?P<kind>function|class|method|property|variable|"
    r"attribute|import) '(?P<name>[^']+)'|unreachable code after '[^']+'|.+?) "
    r"\((?P<conf>\d+)% confidence(?:, (?P<size>\d+) lines?)?\)$")
TEXT_SUFFIXES = {".yaml", ".yml", ".json", ".toml", ".cfg", ".ini", ".sh", ".bash", ".slurm", ".sbatch",
                 ".ipynb", ".jinja", ".j2"}


def _references(repo: Path, shipped: list) -> list:
    """(path, text, strings_only) of tracked files that can name code without importing it.

    Configs, scripts and notebooks count wholesale, as do Python files outside the scope (tests). In the
    shipped Python files only string literals count, e.g. {"target": "main.SetupCallback"}.
    """
    out = []
    shipped = set(shipped)
    for rel in git(repo, "ls-files", "-z").split("\0"):
        path = repo / rel
        suffix = Path(rel).suffix
        if not rel or not path.is_file() or path.stat().st_size > 5_000_000:
            continue
        if suffix in TEXT_SUFFIXES or (suffix == ".py" and rel not in shipped):
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


def runtime_lines(index, file: str, line: int) -> set:
    """Lines that only run when the code is used: a function's body, or the bodies of a class's methods.
    Def, decorator and class-body lines run on import, so coverage of them says nothing."""
    d = index.at_def_line(file, line)
    if not d:
        return set()
    start, end, _, _, kind = d
    bodies = [d] if kind == "function" else [
        x for x in index.defs.get(file, ()) if x[4] == "function" and start <= x[0] and x[1] <= end]
    return {n for _, e, _, def_line, _ in bodies for n in range(def_line + 1, e + 1)}


def vulture_rows(text: str) -> list:
    """vulture's report lines as regex matches. An unrecognized line raises ToolError instead of being
    skipped, so a changed format can't read as "no dead code"."""
    rows = []
    for row in text.splitlines():
        if row.strip():
            if not (m := VULTURE.match(row)):
                raise ToolError(f"output not recognized: {row[:120]}")
            rows.append(m)
    return rows


def executed_lines(repo: Path) -> dict:
    """{file: lines that ran}, from an optional coverage.py JSON report at the repo root (from any run)."""
    if not (cov := repo / "coverage.json").exists():
        return {}
    return {Path(name).as_posix(): set(f.get("executed_lines", []))
            for name, f in json.loads(cov.read_text()).get("files", {}).items()}


def dead_candidate(m, index, refs: list, coverage: dict) -> dict:
    """A vulture row with what decides it: the symbol, the files that name or mention it, whether it ran.
    `unused` (nothing names it and it didn't run) is what dead-code LOC counts."""
    file = Path(m["path"]).as_posix()
    line = int(m["line"])
    d = index.at_def_line(file, line)
    qual = d[2] if d and m["name"] and d[2].endswith(m["name"]) else (m["name"] or index.qualname(file, line))
    modules = [p for p in Path(file).with_suffix("").parts[::-1] if p != "__init__"]
    # unreachable code can't be referenced
    hits, mentions = _named_in(refs, qual.rsplit(".", 1)[-1], modules) if m["kind"] else ([], [])
    ran = bool(coverage.get(file, set()) & runtime_lines(index, file, line))
    return {"row": m, "file": file, "line": line, "lines": int(m["size"] or 1), "symbol": qual, "hits": hits,
            "mentions": mentions, "ran": ran, "unused": not hits and not ran}


def dead_issue(c: dict, with_coverage: bool) -> dict:
    """Open, or dismissed when something names the symbol or its body ran."""
    m = c["row"]
    size = c["lines"]
    note = (f"referenced by name in {', '.join(c['hits'][:3])}" if c["hits"] else
            "executed in the coverage run" if c["ran"] else "")
    return issue(
        "DEAD", "vulture", "dead-code", m["kind"] or "unreachable",
        "dead-large" if size >= DEAD_HIGH_LINES else "dead-small",
        f"{m['msg']} ({size} lines, vulture {m['conf']}% confidence)",
        file=c["file"], symbol=c["symbol"], line=c["line"], end_line=c["line"] + size - 1,
        evidence={"lines": size, "confidence": int(m["conf"]), "referenced_in": c["hits"][:5],
                  "mentioned_in": c["mentions"][:5], "executed": c["ran"] if with_coverage else None},
        status="dismissed" if note else "open", note=note, effort="S",
        fix=("Confirm nothing references it (yaml `_target_`/`target:` strings, registries, getattr, "
             "notebooks, shell scripts; files that mention its bare name are in `evidence.mentioned_in`), "
             "then delete it."),
        done_when="Deleted, and the import test and safety net pass.")


def vulture_dead(repo, files, index) -> Result:
    if not files:
        return Result([], {"dead_code_loc": 0})
    p = run([uvx(), f"vulture@{TOOLS['vulture']}", *files, "--min-confidence", "60", "--sort-by-size"], cwd=repo)
    if p.returncode not in (0, 3):
        raise ToolError((p.stderr or p.stdout).strip()[-300:])
    refs = _references(repo, files)
    coverage = executed_lines(repo)
    # methods and properties are skipped: in ML code they are mostly framework hooks (Lightning's
    # training_step, nn.Module overrides) that vulture can't see being called
    found = [dead_candidate(m, index, refs, coverage) for m in vulture_rows(p.stdout)
             if m["kind"] not in ("method", "property", "variable", "attribute", "import")]
    total = sum(c["lines"] for c in found if c["unused"])
    return Result([dead_issue(c, bool(coverage)) for c in found if c["lines"] >= MIN_DEAD_LINES],
                  {"dead_code_loc": total})
