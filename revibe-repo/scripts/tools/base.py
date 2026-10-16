"""What a tool module provides, its TOOL and a `measure` or `scan` function, and what both return: a `Result`
of issues built by `Tool.issue`."""

import hashlib
from dataclasses import dataclass, field
from typing import NamedTuple, Protocol, runtime_checkable

from scope import Scope


class ToolError(Exception):
    """A tool could not measure: it failed, or returned something this script doesn't recognize."""


class Result(NamedTuple):
    """What a tool found. A step with nothing to read sets `skipped`: it measured nothing, so its
    earlier issues stay open instead of being closed as fixed."""

    issues: list
    metrics: dict
    skipped: str = ""
    notes: dict | None = None  # metric -> why it is incomplete


class Metric(NamedTuple):
    """A number in a tool's Result, as the scorecard shows it and `compare` checks it."""

    key: str  # its key in Result.metrics and the snapshots
    label: str = ""  # its scorecard row; "" keeps it off the scorecard
    form: str = "{:,}"
    reference: tuple | None = None  # (maintained human code, agent code), from SlopCodeBench
    warn: str = ""  # what `compare` calls it when it goes up; "" when a rise is no regression


@dataclass(frozen=True)
class Tool:
    """A tool's name, where it comes from and its pinned version, the skill's settings for it, its severity policy
    and its metrics.

    `policy` is the one place the tool's severities are set: every issue it reports names a row, and `report`
    prints all of them into report.md, so the docs point here instead of restating them. A row's `when`, and a
    metric's label and warning, name the settings they depend on in braces, e.g. "{min_dead_lines}".
    """

    name: str  # the `source` of its issues: merge() closes an open issue its source no longer finds
    policy: dict  # policy key -> (severity, when)
    package: str = ""  # its PyPI distribution, pinned in cleanup.py's script header; "" when not from PyPI
    version: str = ""  # the pinned version its code is written against; the header pins a package at the same one
    settings: dict = field(default_factory=dict)  # thresholds and pins its code reads, by name
    metrics: tuple = ()  # the Metric of each scorecard number its measure step reports, in scorecard order

    def severity_rules(self) -> dict:
        """The policy with each row's `when` filled in from the settings, as report.md prints it."""
        return {key: (severity, when.format(**self.settings)) for key, (severity, when) in self.policy.items()}

    def scorecard_metrics(self) -> list:
        """The metrics with their labels and warnings filled in from the settings."""
        return [m._replace(label=m.label.format(**self.settings), warn=m.warn.format(**self.settings))
                for m in self.metrics]

    def issue(self, prefix: str, category: str, rule: str, policy: str, summary: str, *, file="", symbol="", key="",
              line=None, end_line=None, evidence=None, fix="", done_when="", effort="S", status="open",
              note="") -> dict:
        """An issue this tool reports, with the severity of its `policy` row. IDs hash stable facts (never line
        numbers), so re-runs keep them."""
        basis = key or "|".join((category, rule, file, symbol))
        return {
            "id": f"{prefix}-{hashlib.sha1(basis.encode()).hexdigest()[:8]}",
            "source": self.name, "category": category, "rule": rule, "severity": self.policy[policy][0],
            "file": file, "line": line, "end_line": end_line, "symbol": symbol, "summary": summary,
            "evidence": evidence or {}, "fix": fix, "done_when": done_when, "effort": effort,
            "status": status, "note": note,
        }


@runtime_checkable
class Step(Protocol):
    """A measure step: a tool module with a TOOL and a `measure` function."""

    TOOL: Tool

    def measure(self, scope: Scope) -> Result: ...
