"""One module per tool: its TOOL (name, PyPI package, settings, severity policy), and the code that runs the tool and turns
what it finds into issues (`Tool.issue`). Agents set the severity of the red flags and bugs they add.

The measure steps satisfy `Step`. The leak steps, `betterleaks` and `large_files`, have a TOOL too, and a `scan`
function that `cleanup.py leaks` calls.
"""

from tools import betterleaks, deptry, file_length, large_files, license, module_globals, pip_audit, scb_check, vulture
from tools.base import Step

MEASURE: tuple[Step, ...] = (scb_check, vulture, deptry, pip_audit, license, file_length, module_globals)
ANALYZERS = tuple(m for m in MEASURE if m.TOOL.package)  # the tools cleanup.py's script header pins
ALL = (betterleaks, large_files, *MEASURE)


def policies() -> dict:
    """Every tool's policy rows in one dict, with their settings filled in."""
    return {key: row for m in ALL for key, row in m.TOOL.severity_rules().items()}
