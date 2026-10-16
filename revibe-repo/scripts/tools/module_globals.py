"""Module-level globals, from the code index: constants and state a module keeps outside its functions."""

from tools.base import Result, Tool

TOOL = Tool(
    name="globals",
    policy={
        "globals": ("low", "module-level globals, constants included: each could live next to its only user"),
    },
)


def measure(repo, files, index) -> Result:
    """How many module-level globals the code has, as a metric and as one issue naming the files with most."""
    per_file = {f: names for f, names in index.globals.items() if names}
    total = sum(len(names) for names in per_file.values())
    if not total:
        return Result([], {"globals": 0})
    most = sorted(per_file.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    return Result([TOOL.issue(
        "GLOB", "globals", "module-globals", "globals",
        f"{total} module-level globals in {len(per_file)} files, most in "
        + ", ".join(f"`{f}` ({len(names)})" for f, names in most[:3]),
        key="GLOB|module", evidence={"globals": total, "weight": total,
                                     "files": {f: names[:20] for f, names in most[:20]}},
        effort="M",
        fix=("Move each constant next to its only user (a local, a default argument, or a class attribute), and "
             "pass state as arguments instead of keeping it in the module. Constants that several functions "
             "share, the public API and `logger` objects can stay: dismiss with that reason once only those "
             "are left."),
        done_when="Only deliberate globals are left (dismissed with a note naming them), and the safety net passes.")],
        {"globals": total})
