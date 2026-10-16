"""What a tool is: a `Tool` subclass that says what the tool finds, created in tools/__init__.py with the severity
of each finding and the settings they depend on. Every tool returns a `Result` of issues built by `Tool.issue`."""

import hashlib
import string
from abc import ABC, abstractmethod
from typing import NamedTuple

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


class Tool:
    """A tool. Its class says what it is and what it finds; an instance holds the policy it was created with.

    `policy` names each kind of finding and says in words what it covers; the `severity` given at creation says
    how bad each one is. report.md prints both for every tool, so the docs point there instead of restating
    them. A policy row, and a metric's label and warning, name the settings they depend on in braces, e.g.
    "{min_dead_lines}".
    """

    name = ""  # the `source` of its issues: merge() closes an open issue its source no longer finds
    package = ""  # its PyPI distribution, pinned in cleanup.py's script header; "" when not from PyPI
    version = ""  # the pinned version its code is written against; the header pins a package at the same one
    policy: dict = {}  # policy key -> what it covers
    metrics: tuple = ()  # the Metric of each scorecard number its measure step reports, in scorecard order

    def __init__(self, severity: dict | None = None, settings: dict | None = None):
        """Stops at once when `severity` doesn't cover exactly the policy, or a policy row or metric names a
        setting that isn't given."""
        self.severity = severity or {}
        self.settings = settings or {}
        if self.severity.keys() != self.policy.keys():
            raise ValueError(f"{self.name}: severities for {sorted(self.severity)}, but its policy is {sorted(self.policy)}")
        texts = [*self.policy.values(), *(text for m in self.metrics for text in (m.label, m.warn))]
        named = {field for text in texts for _, field, _, _ in string.Formatter().parse(text) if field}
        if missing := named - self.settings.keys():
            raise ValueError(f"{self.name}: no setting for {', '.join(sorted(missing))}")

    def severity_rules(self) -> dict:
        """Each policy row's severity and what it covers, with the settings filled in, as report.md prints it."""
        return {key: (self.severity[key], when.format(**self.settings)) for key, when in self.policy.items()}

    def scorecard_metrics(self) -> list:
        """The metrics with their labels and warnings filled in from the settings."""
        return [m._replace(label=m.label.format(**self.settings), warn=m.warn.format(**self.settings))
                for m in self.metrics]

    def adjust(self, found: list):
        """Change the issues that all the measure steps found, once every step has run. Only vulture does."""

    def issue(self, prefix: str, category: str, rule: str, policy: str, summary: str, *, file="", symbol="", key="",
              line=None, end_line=None, evidence=None, fix="", done_when="", effort="S", status="open",
              note="") -> dict:
        """An issue this tool reports, with the severity of its `policy` row. IDs hash stable facts (never line
        numbers), so re-runs keep them."""
        basis = key or "|".join((category, rule, file, symbol))
        return {
            "id": f"{prefix}-{hashlib.sha1(basis.encode()).hexdigest()[:8]}",
            "source": self.name, "category": category, "rule": rule, "severity": self.severity[policy],
            "file": file, "line": line, "end_line": end_line, "symbol": symbol, "summary": summary,
            "evidence": evidence or {}, "fix": fix, "done_when": done_when, "effort": effort,
            "status": status, "note": note,
        }


class Step(Tool, ABC):
    """A measure step: a Tool that measures the code in scope."""

    @abstractmethod
    def measure(self, scope: Scope) -> Result:
        """What the tool finds in `scope`."""
