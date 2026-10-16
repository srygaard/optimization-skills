"""Module-level globals, from the scope's index: constants and state a module keeps outside its functions."""

from tools.base import Metric, Result, Step


class ModuleGlobals(Step):
    """How many globals the modules keep, as one issue and a count."""

    name = "globals"
    policy = {
        "globals": "module-level globals, constants included: each could live next to its only user",
    }
    metrics = (Metric("globals", "Globals", warn="module-level globals"),)

    def measure(self, scope) -> Result:
        """How many module-level globals the code has, as a metric and as one issue naming the files with most."""
        per_file = {f: names for f, names in scope.globals.items() if names}
        total = sum(len(names) for names in per_file.values())
        if not total:
            return Result([], {"globals": 0})
        most = sorted(per_file.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        return Result([self.issue(
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
