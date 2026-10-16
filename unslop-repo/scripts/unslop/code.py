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
        self.defs, self.code, self.lines, self.syntax_errors = {}, {}, {}, []
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


def scb_issues(text: str, src: Path, index, high_cc: int) -> list:
    """Duplicate blocks, trivial wrappers and complex functions from scb-check's human output. Raises
    ToolError when it doesn't parse as expected, so a changed format can't read as "all fixed"."""
    clones, complex_fns, cognitive, out = [], [], {}, []
    for f in parse_scb(text):
        locs = [(rel(src, path), line) for path, line in f["locs"]]
        if not locs and f["rule"] in ("erosion", "duplicate-structure", "trivial-wrapper", "low-use-short-function"):
            raise ToolError(f"no location parsed for {f['rule']}: {f['msg'][:80]}")
        size = next((m for n in f["notes"] if (m := SCB_CC.search(n))), None)
        if f["rule"] in ("erosion", "cog_erosion") and size:
            if f["rule"] == "erosion":
                name = SCB_FUNCTION.search(f["msg"])
                complex_fns.append((*locs[0], int(size[1]), int(size[2]), name[1] if name else "<function>"))
            else:
                cognitive[locs[0]] = int(size[1])
        elif f["rule"] == "duplicate-structure" and (c := SCB_CLONE.search(f["msg"])):
            clones.append((int(c[1]) * (int(c[2]) - 1), int(c[1]), int(c[2]), locs))
        elif f["rule"] in ("trivial-wrapper", "low-use-short-function"):
            file, line = locs[0]
            name = (index.at_def_line(file, line) or (0, 0, "<module>"))[2]
            flag ="pass-through-method" if f["rule"] == "trivial-wrapper" else "shallow-module"
            out.append(issue(
                "RF", "scb-check", "red-flag", flag, "wrapper", f"{f['rule']}: {f['msg']}",
                file=file, symbol=name, line=line, evidence={"scb_rule": f["rule"]}, effort="S",
                fix=("Inline it into its callers or remove it. If it is deliberate (a public alias, or an "
                     "override such as `model.train = disabled_train`), dismiss the issue with that reason."),
                done_when="scb-check no longer reports it (or it is dismissed), and the safety net passes."))
    if len(complex_fns) != high_cc:
        raise ToolError(f"output not recognized: its JSON counts {high_cc} functions with CC > {CC_THRESHOLD}, "
                        f"{len(complex_fns)} were parsed")
    for weight, lines, count, locs in clones:
        members = sorted({(file, index.qualname(file, line)) for file, line in locs})
        where = ", ".join(f"{file}:{line}" for file, line in locs[:4])
        out.append(issue(
            "RF", "scb-check", "red-flag", "repetition",
            "repetition" if weight >= MEDIUM_DUP_LINES else "repetition-small",
            f"{lines}-line block duplicated {count}x: {where}",
            key="RF|repetition|" + "|".join(f"{f}:{q}" for f, q in members),
            file=locs[0][0], line=locs[0][1], symbol=", ".join(q for _, q in members),
            evidence={"lines": lines, "instances": count, "weight": weight,
                      "locations": [f"{f}:{n}" for f, n in locs]},
            effort="M" if lines >= 30 else "S",
            fix=("Extract the shared block into one function, or delete the copy that is no longer used. "
                 "Pass the real differences as a few parameters."),
            done_when="scb-check no longer reports this duplicate group and the safety net passes."))
    for file, line, cc, sloc, name in complex_fns:
        d = index.at_def_line(file, line)
        qual, start, end = (d[2], d[0], d[1]) if d else (name, line, line)
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
    out, shipped = [], set(shipped)
    for rel in git(repo, "ls-files", "-z").split("\0"):
        path, suffix = repo / rel, Path(rel).suffix
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
    n, m = re.escape(name), "(?:" + "|".join(map(re.escape, modules)) + ")"
    qualified = re.compile(rf"\b{m}[.:]{n}\b|\bfrom\s+[\w.]*\b{m}\s+import\s*(?:\([^)]*|[^\n]*)\b{n}\b")
    in_string = re.compile(rf"[\"'][^\"'\n]*\b{m}[.:]{n}[\"']")
    bare, bare_string = re.compile(rf"\b{n}\b"), re.compile(rf"[\"']{n}[\"']")
    named, mentioned = [], []
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


def vulture_dead(repo, files, index) -> Result:
    if not files:
        return Result([], {"dead_code_loc": 0})
    p = run([uvx(), f"vulture@{TOOLS['vulture']}", *files, "--min-confidence", "60", "--sort-by-size"], cwd=repo)
    if p.returncode not in (0, 3):
        raise ToolError((p.stderr or p.stdout).strip()[-300:])
    refs = _references(repo, files)
    coverage = {}
    if (cov := repo / "coverage.json").exists():  # optional: a coverage.py JSON report from any run
        for name, f in json.loads(cov.read_text()).get("files", {}).items():
            coverage[Path(name).as_posix()] = set(f.get("executed_lines", []))
    total, out = 0, []
    for m in vulture_rows(p.stdout):
        # methods and properties are skipped: in ML code they are mostly framework hooks (Lightning's
        # training_step, nn.Module overrides) that vulture can't see being called
        if m["kind"] in ("method", "property", "variable", "attribute", "import"):
            continue
        file, line, size = Path(m["path"]).as_posix(), int(m["line"]), int(m["size"] or 1)
        d = index.at_def_line(file, line)
        qual = d[2] if d and m["name"] and d[2].endswith(m["name"]) else (m["name"] or index.qualname(file, line))
        name = qual.rsplit(".", 1)[-1]
        modules = [p for p in Path(file).with_suffix("").parts[::-1] if p != "__init__"]
        # unreachable code can't be referenced
        hits, mentions = _named_in(refs, name, modules) if m["kind"] else ([], [])
        ran = bool(coverage.get(file, set()) & runtime_lines(index, file, line))
        if not hits and not ran:
            total += size
        if size < MIN_DEAD_LINES:
            continue
        note = (f"referenced by name in {', '.join(hits[:3])}" if hits else
                "executed in the coverage run" if ran else "")
        out.append(issue(
            "DEAD", "vulture", "dead-code", m["kind"] or "unreachable",
            "dead-large" if size >= DEAD_HIGH_LINES else "dead-small",
            f"{m['msg']} ({size} lines, vulture {m['conf']}% confidence)",
            file=file, symbol=qual, line=line, end_line=line + size - 1,
            evidence={"lines": size, "confidence": int(m["conf"]), "referenced_in": hits[:5],
                      "mentioned_in": mentions[:5], "executed": ran if coverage else None},
            status="dismissed" if note else "open", note=note, effort="S",
            fix=("Confirm nothing references it (yaml `_target_`/`target:` strings, registries, getattr, "
                 "notebooks, shell scripts; files that mention its bare name are in `evidence.mentioned_in`), "
                 "then delete it."),
            done_when="Deleted, and the import test and safety net pass."))
    return Result(out, {"dead_code_loc": total})
