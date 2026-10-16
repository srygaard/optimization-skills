#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["scb-check==0.2.0", "vulture==2.16", "deptry==0.25.1", "pip-audit==2.10.1"]
# ///
"""Measure, track and report the cleanup of a Python ML repo before open-sourcing.

    cleanup.py leaks   REPO [--internal R] [--history]  secrets and internal refs: each commit once, and HEAD
    cleanup.py measure REPO [--label L] [--exclude G]  this commit's metrics + mechanical issues
    cleanup.py compare REPO [BASE] [--to REF]         before/after table and regression warnings
    cleanup.py report  REPO                           render REPO/.revibe/report.md

Everything is written to REPO/.revibe/, which ignores itself in git: issues.json, report.md, and a folder
per measured commit with its metrics.json. Secret values never reach
stdout or any file: only a known public prefix (`ghp_`, `hf_`) and the length. Tests: test_cleanup.py.

The analyzers are pinned above and run in this script's environment, so `measure` needs `uv run --script`
(the shebang does it); the other commands need only the stdlib. Betterleaks is not on PyPI, so it is
downloaded once as a pinned release binary, checked against SHA-256 values from the release's
sigstore-signed checksums.txt, and cached. Set BETTERLEAKS=/path/to/betterleaks to skip the download.
"""

import argparse
import sys

sys.dont_write_bytecode = True  # keep __pycache__ out of the skill folder and .revibe/
from leaks import cmd_leaks
from measure import cmd_measure
from report import cmd_compare, cmd_report


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("leaks", help="scan for secrets and internal refs: the commits never read, and HEAD")
    p.add_argument("repo")
    p.add_argument("--internal", action="append", default=[], metavar="REGEX",
                   help="extra internal-reference pattern (RE2); repeatable")
    p.add_argument("--history", action="store_true",
                   help="read every commit again (new rules, or a commit read before leaving the history, do this anyway)")
    p = sub.add_parser("measure", help="this commit's metrics and mechanical issues")
    p.add_argument("repo")
    p.add_argument("--label", default="", help="a name for this commit's metrics, e.g. baseline")
    p.add_argument("--exclude", action="append", default=[], metavar="GLOB",
                   help="extra path glob to leave out of the metrics; repeatable")
    p = sub.add_parser("compare", help="two measured commits side by side")
    p.add_argument("repo")
    p.add_argument("base", nargs="?", help="a label, commit or git ref (default: the baseline)")
    p.add_argument("--to", help="a label, commit or git ref (default: the latest measured)")
    p = sub.add_parser("report", help="render .revibe/report.md")
    p.add_argument("repo")
    a = ap.parse_args()
    {"leaks": cmd_leaks, "measure": cmd_measure, "compare": cmd_compare, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
