"""Magic numbers: numeric literals whose meaning no name gives, such as `x[:, 3]`, `lr * 0.1` or `torch.randn(3, 224)`."""

import ast

from tools.base import Metric, Result, Step


class MagicNumbers(Step):
    """Unnamed numeric literals, per function and as one count."""

    name = "magic-numbers"
    policy = {
        "magic-function": "function or method with {medium_magic}+ distinct unnamed numbers",
        "magic-numbers": "unnamed numbers in the rest of the code, as one issue",
    }
    metrics = (Metric("magic_numbers", "Magic numbers", warn="unnamed numeric literals"),)
    plain = {-1, 0, 1, 2}  # their meaning is usually plain where they stand

    def measure(self, scope) -> Result:
        """A medium issue for each function with many distinct magic numbers, and one low issue counting them all."""
        per_def = {}  # (file, qualname) -> [(line, value)]
        for rel in scope.defs:
            tree = ast.parse((scope.repo / rel).read_text(errors="replace"))
            for line, value in self.magic_numbers(tree):
                per_def.setdefault((rel, scope.qualname(rel, line)), []).append((line, value))
        total = sum(len(found) for found in per_def.values())
        issues = [self.function_issue(file, qualname, found) for (file, qualname), found in per_def.items()
                  if qualname != "<module>" and len({v for _, v in found}) >= self.settings["medium_magic"]]
        if total:
            issues.append(self.count_issue(per_def, total))
        return Result(issues, {"magic_numbers": total})

    def magic_numbers(self, tree: ast.Module) -> list:
        """(line, value) of each numeric literal that isn't plain and that nothing names: not the value of an
        assignment, a default argument, a keyword argument or a dict entry with a string key. A literal inside a
        tuple or list, or negated, takes its meaning from the same place."""
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        out = []
        for node in ast.walk(tree):
            if not is_number(node):
                continue
            negated = isinstance(parents.get(node), ast.UnaryOp) and isinstance(parents[node].op, ast.USub)
            value = -node.value if negated else node.value
            top = outermost(node, parents)
            if value not in self.plain and not named(top, parents.get(top)):
                out.append((node.lineno, value))
        return out

    def function_issue(self, file: str, qualname: str, found: list) -> dict:
        distinct = sorted({v for _, v in found})
        shown = ", ".join(map(str, distinct[:6])) + (" ..." if len(distinct) > 6 else "")
        return self.issue(
            "MAGIC", "red-flag", "magic-numbers", "magic-function",
            f"`{qualname}` uses {len(distinct)} unnamed numbers: {shown}", file=file, symbol=qualname,
            line=min(line for line, _ in found),
            evidence={"values": distinct[:20], "count": len(found), "weight": len(found)}, effort="S",
            fix=("Name each number where it is used: a keyword argument, a default parameter, a local, a class "
                 "attribute or a config value. Leave a number whose meaning is plain where it stands (an array axis "
                 "next to a named shape, `x ** 3`), and don't move them into module-level constants."),
            done_when="It has fewer unnamed numbers than the threshold, or the rest are plain (dismissed with that "
                      "note), and the safety net passes.")

    def count_issue(self, per_def: dict, total: int) -> dict:
        per_file = {}
        for (file, _), found in per_def.items():
            per_file[file] = per_file.get(file, 0) + len(found)
        most = sorted(per_file.items(), key=lambda kv: (-kv[1], kv[0]))
        return self.issue(
            "MAGIC", "red-flag", "magic-numbers", "magic-numbers",
            f"{total} unnamed numbers in {len(per_file)} files, most in "
            + ", ".join(f"`{f}` ({n})" for f, n in most[:3]),
            key="MAGIC|count", evidence={"count": total, "files": dict(most[:20])}, effort="M",
            fix=("Name the numbers whose meaning isn't plain, in the files with most first, the way the "
                 "magic-function issues say. Leave plain ones; dismiss with that note once only those are left."),
            done_when="Only plain numbers are left (dismissed with a note), and the safety net passes.")


def is_number(node: ast.AST) -> bool:
    """An int or float literal; True and False are ints to Python but not numbers here."""
    return isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool)


def outermost(node: ast.AST, parents: dict) -> ast.AST:
    """The expression a literal takes its meaning from: itself, or the tuple, list or negation around it."""
    while isinstance(parents.get(node), (ast.UnaryOp, ast.Tuple, ast.List)):
        node = parents[node]
    return node


def named(top: ast.AST, parent: ast.AST | None) -> bool:
    """Whether the expression `top` gets its meaning from a name: the value assigned to a target, a default
    argument, a keyword argument, or the value of a dict entry with a string key."""
    if isinstance(parent, (ast.Assign, ast.AnnAssign, ast.keyword, ast.arguments)):
        return True
    if isinstance(parent, ast.Dict) and top in parent.values:
        key = parent.keys[parent.values.index(top)]
        return isinstance(key, ast.Constant) and isinstance(key.value, str)
    return False
