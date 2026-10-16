"""scb-check, in process: the SlopCodeBench metrics, complex functions, duplicate blocks and trivial wrappers."""

import contextlib
import io
import math
import os
import shutil
import tempfile
from importlib.metadata import distribution
from pathlib import Path

from utils.paths import rel
from scope import Def
from tools.base import Result, Tool, ToolError

TOOL = Tool(
    name="scb-check",
    package="scb-check",
    policy={
        "complex": ("medium", "function with CC > {cc_threshold} and CC x sqrt(SLOC) >= {medium_mass} "
                              "(e.g. CC 15 over 45 lines)"),
        "repetition": ("medium", "duplicate group with lines x (copies - 1) >= {medium_dup_lines}"),
        "wrapper": ("medium", "trivial wrapper or shallow function"),
        "complex-small": ("low", "function with CC > {cc_threshold} and a smaller mass"),
        "repetition-small": ("low", "smaller duplicate group of {min_clone_lines}+ lines"),
    },
    settings={
        "cc_threshold": 10,  # scb-check 0.2.0's own: Erosion is the share of mass in functions above it
        "medium_mass": 100,  # complexity mass CC x sqrt(SLOC) from which a function is medium
        "medium_dup_lines": 20,  # duplicate-group weight lines x (copies - 1) from which it is medium
        "min_clone_lines": 10,  # smaller duplicate blocks still count in Clone Ratio but get no issue
    },
)


def analyze(root: Path, files: list) -> tuple:
    """(scb-check's flags for `files` under `root`, its warnings). It tries a global `sg` before its own
    ast-grep, and that may be another version or shadow-utils' `sg`, so its own goes first on PATH."""
    from scb_check.logging import configure_logging
    from scb_check.pipeline import analyze_files

    package = distribution("ast-grep-cli")
    sg = next(package.locate_file(f) for f in package.files or () if f.name == "sg")
    path = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{Path(sg).parent}{os.pathsep}{path}"
    log = io.StringIO()
    try:
        with contextlib.redirect_stderr(log):
            configure_logging(0)  # warnings only, to `log`
            flags = analyze_files(tuple(root / f for f in files)).flags
    finally:
        os.environ["PATH"] = path
    return flags, log.getvalue()


def measure(repo, files, index) -> Result:
    from scb_check.reporting.score import compute_report

    if not files:
        return Result([], {}, skipped="no Python files in scope")
    with tempfile.TemporaryDirectory(prefix="cleanup-scb-") as tmp:
        # a copy of the files in scope, where a tracked symlink is a plain file: ast-grep resolves paths, and
        # a symlink resolved outside the repo leaves its matches uncounted
        src = Path(tmp).resolve()
        for f in files:
            (src / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(repo / f, src / f)
        flags, warnings = analyze(src, files)
    if not flags.lines.total_loc_by_file:
        raise ToolError("no file in scope could be parsed")
    m = compute_report(flags).to_dict()
    out = [*wrapper_issues(src, flags, index), *repetition_issues(src, flags, index),
           *complexity_issues(src, flags, index)]
    loc = m["total_loc"] or 1
    metrics = dict(
        loc=m["total_loc"], cc_over_10=m["high_cc_functions"], verbosity=round(m["verbosity"], 4),
        erosion=round(m["erosion"], 4), cog_erosion=round(m["cog_erosion"], 4),
        violation_pct=round(m["ast_grep_flagged_loc"] / loc, 4), clone_ratio=round(m["clone_loc"] / loc, 4),
        # absolute amounts behind the ratios: deleting clean code raises a ratio without adding slop
        verbosity_flagged_loc=m["verbosity_flagged_loc"], high_cc_mass=round(m["high_cc_mass"], 1))
    # no ast-grep binary worked: the slop rules silently counted nothing
    return Result(out, metrics, notes={"violation_pct": "ast-grep did not run"} if "ast-grep" in warnings else None)


def wrapper_issues(src: Path, flags, index) -> list:
    """Ousterhout's pass-through methods and shallow modules, from scb-check's wrapper rules."""
    red_flags = {"trivial-wrapper": "pass-through-method", "low-use-short-function": "shallow-module"}
    out = []
    for f in flags.findings.structural_findings:
        if f.rule_id not in red_flags:
            continue
        file = rel(src, f.file)
        d = index.at_def_line(file, f.start_line)
        out.append(TOOL.issue(
            "RF", "red-flag", red_flags[f.rule_id], "wrapper", f"{f.rule_id}: {f.message}",
            file=file, symbol=d.qualname if d else "<module>", line=f.start_line,
            evidence={"scb_rule": f.rule_id}, effort="S",
            fix=("Inline it into its callers or remove it. If it is deliberate (a public alias, or an "
                 "override such as `model.train = disabled_train`), dismiss the issue with that reason."),
            done_when="scb-check no longer reports it (or it is dismissed), and the safety net passes."))
    return out


def repetition_issues(src: Path, flags, index) -> list:
    """One issue per duplicate group, named after the functions that hold the copies. A copy's length counts
    its code lines, as scb-check prints it."""
    clone_sloc = {e.file: e.lines for e in flags.lines.clone_sloc_lines_by_file}
    groups = {}
    for clone in flags.findings.clones:
        groups.setdefault(clone.group_hash, []).append(clone)
    out = []
    for group in groups.values():
        group.sort(key=lambda c: (rel(src, c.file), c.start_line))
        anchor = group[0]
        span = set(range(anchor.start_line, anchor.end_line + 1))
        lines = len(span & clone_sloc[anchor.file]) if anchor.file in clone_sloc else len(span)
        if lines < TOOL.settings["min_clone_lines"]:
            continue
        count = anchor.instance_count
        locs = [(rel(src, c.file), c.start_line) for c in group]
        weight = lines * (count - 1)
        members = sorted({(file, index.qualname(file, line)) for file, line in locs})
        places = [f"{file}:{line}" for file, line in locs]
        out.append(TOOL.issue(
            "RF", "red-flag", "repetition", "repetition" if weight >= TOOL.settings["medium_dup_lines"] else "repetition-small",
            f"{lines}-line block duplicated {count}x: {', '.join(places[:4])}",
            key="RF|repetition|" + "|".join(f"{f}:{q}" for f, q in members),
            file=locs[0][0], line=locs[0][1], symbol=", ".join(q for _, q in members),
            evidence={"lines": lines, "instances": count, "weight": weight, "locations": places},
            effort="M" if lines >= 30 else "S",
            fix=("Extract the shared block into one function, or delete the copy that is no longer used. "
                 "Pass the real differences as a few parameters."),
            done_when="scb-check no longer reports this duplicate group and the safety net passes."))
    return out


def complexity_issues(src: Path, flags, index) -> list:
    """One issue per function over the CC threshold, with its cognitive complexity as evidence."""
    out = []
    for fn in flags.findings.high_cc_functions:
        file = rel(src, fn.file)
        line = fn.start_line
        cc = fn.cyc_complexity
        sloc = fn.sloc
        d = index.at_def_line(file, line) or Def(line, line, fn.name, line, "function")
        mass = cc * math.sqrt(sloc)
        out.append(TOOL.issue(
            "CX", "complexity", "cc", "complex" if mass >= TOOL.settings["medium_mass"] else "complex-small",
            f"`{d.qualname}` has cyclomatic complexity {cc} ({sloc} SLOC)",
            file=file, symbol=d.qualname, line=d.start, end_line=d.end,
            evidence={"cc": cc, "sloc": sloc, "mass": round(mass, 1), "weight": round(mass, 1),
                      "cognitive": fn.cog_complexity},
            effort="L" if sloc > 150 else "M",
            fix=("Simplify the control flow: drop dead branches and flags, return early, move special cases "
                 "out of the general path. Extracting a step with its own clear purpose is fine; moving "
                 "branches into a helper that only this function calls, with most of its arguments, is not. "
                 "If every branch is needed and it reads clearly, dismiss it with that reason. Add a "
                 "characterization test first."),
            done_when=(f"scb-check no longer lists it (CC <= {TOOL.settings['cc_threshold']}), or it is gone or dismissed with "
                       "a reason, and the safety net passes.")))
    return out
