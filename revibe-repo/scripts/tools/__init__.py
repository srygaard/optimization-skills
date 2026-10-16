"""The tools, each created with its policy: the severity of each kind of finding it reports, and the settings
those depend on. This is the one place both are set; each tool's module says what its findings cover and how it
finds them. Agents set the severity of the red flags and bugs they add.

The measure steps are `Step`s. The leak scan has a `scan` method instead, which `cleanup.py leaks` calls.
"""

from tools.base import Step
from tools.betterleaks import Betterleaks
from tools.clutter import Clutter
from tools.deptry import Deptry
from tools.file_length import FileLength
from tools.large_files import LargeFiles
from tools.license import License
from tools.module_globals import ModuleGlobals
from tools.pip_audit import PipAudit
from tools.ruff import Ruff
from tools.scb_check import ScbCheck
from tools.suppressions import Suppressions
from tools.vulture import Vulture

MEASURE: tuple[Step, ...] = (
    ScbCheck(
        severity={"complex": "medium", "repetition": "medium", "wrapper": "medium", "complex-small": "low",
                  "repetition-small": "low"},
        settings={
            "medium_mass": 100,  # complexity mass CC x sqrt(SLOC) from which a function is medium
            "medium_dup_lines": 20,  # duplicate-group weight lines x (copies - 1) from which it is medium
            "min_clone_lines": 10,  # smaller duplicate blocks still count in Clone Ratio but get no issue
        },
    ),
    Vulture(
        severity={"dead-large": "high", "dead-small": "low", "inside-dead-code": "low"},
        settings={
            "min_confidence": 60,  # vulture's confidence below which a candidate isn't reported
            "min_dead_lines": 5,  # smaller unused symbols still count in dead-code LOC but get no issue
            "dead_high_lines": 20,  # unused symbols at least this long are high: remove them before release
        },
    ),
    Deptry(severity={"deps-none": "high", "deps-undeclared": "high", "deps-unused": "low"}),
    PipAudit(severity={"vuln-fixable": "high", "vuln-unfixed": "low", "pins-none": "low"}),
    License(severity={"license-missing": "high", "readme-missing": "high"}),
    LargeFiles(severity={"large-file": "high"}, settings={"large_file_mb": 5}),
    Clutter(severity={"clutter-output": "high", "clutter-scratch": "high", "clutter-notes": "medium"}),
    FileLength(
        severity={"long-file": "medium"},
        settings={
            "long_file_lines": 1000,  # source files with more code lines than this should be split
            "warn_file_lines": 500,  # files with more code lines than this are counted
        },
    ),
    ModuleGlobals(severity={"globals": "low"}),
    Ruff(
        severity={"magic-function": "medium", "magic-numbers": "low", "string-paths": "low"},
        settings={"medium_magic": 3},  # distinct unnamed numbers a function compares against from which it is medium
    ),
    Suppressions(),
)
LEAK_SCAN = Betterleaks(severity={"credential": "critical", "possible-credential": "high", "internal-head": "high",
                                  "internal-history": "low"})
ANALYZERS = tuple(t for t in MEASURE if t.package)  # the tools cleanup.py's script header pins
ALL = (LEAK_SCAN, *MEASURE)


def scorecard_metrics() -> list:
    """Every measure step's metrics, in step order, with their settings filled in."""
    return [m for step in MEASURE for m in step.scorecard_metrics()]


def policies() -> dict:
    """Every tool's policy rows in one dict, with their severities and settings filled in."""
    return {key: row for tool in ALL for key, row in tool.severity_rules().items()}
