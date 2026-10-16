"""The code in scope: the tracked Python files left after the excludes, and the `Scope` every measure step
reads."""

import ast
from fnmatch import fnmatch
from pathlib import Path
from typing import NamedTuple

from utils.git import git


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


def module_globals(tree: ast.Module) -> list:
    """The names a module binds outside its functions and classes, plus those its functions declare `global`;
    dunders such as `__all__` aside."""
    names = {n for g in ast.walk(tree) if isinstance(g, ast.Global) for n in g.names}
    scopes = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda, ast.ListComp, ast.SetComp,
              ast.DictComp, ast.GeneratorExp)

    def visit(node):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, scopes):
                continue
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                names.add(child.id)
            visit(child)

    visit(tree)
    return sorted(n for n in names if not (n.startswith("__") and n.endswith("__")))


class Def(NamedTuple):
    """A function or class: its span (decorators included), qualified name, `def` line and kind."""

    start: int
    end: int
    qualname: str
    def_line: int
    kind: str  # "function" or "class"


class Scope:
    """What a measure step reads: the repo, the files in scope, and for each file its functions and classes
    (a `Def` each), its length and its module-level globals."""

    def __init__(self, repo: Path, files: list):
        self.repo = repo
        self.files = files
        self.defs = {}
        self.lines = {}
        self.globals = {}
        self.syntax_errors = []
        for rel in files:
            text = (repo / rel).read_text(errors="replace")
            self.lines[rel] = len(text.splitlines())
            try:
                tree = ast.parse(text)
            except SyntaxError:
                self.syntax_errors.append(rel)
                continue
            self.globals[rel] = module_globals(tree)
            defs = self.defs[rel] = []

            def walk(node, prefix):
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        start = min([child.lineno] + [x.lineno for x in child.decorator_list])
                        defs.append(Def(start, child.end_lineno, prefix + child.name, child.lineno,
                                        "class" if isinstance(child, ast.ClassDef) else "function"))
                        walk(child, prefix + child.name + ".")
                    else:
                        walk(child, prefix)

            walk(tree, "")

    def enclosing(self, rel: str, line: int) -> Def | None:
        """The innermost function or class whose span holds `line`."""
        best = None
        for d in self.defs.get(rel, ()):
            if d.start <= line <= d.end and (best is None or d.start >= best.start):
                best = d
        return best

    def at_def_line(self, rel: str, line: int) -> Def | None:
        """The function or class defined on `line`, else the one that holds it."""
        return next((d for d in self.defs.get(rel, ()) if d.def_line == line), None) or self.enclosing(rel, line)

    def qualname(self, rel: str, line: int) -> str:
        """The qualified name of what holds `line`, or "<module>"."""
        d = self.enclosing(rel, line)
        return d.qualname if d else "<module>"
