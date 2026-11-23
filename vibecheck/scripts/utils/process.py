"""Running a command."""

import subprocess


def run(cmd, **kwargs) -> subprocess.CompletedProcess:
    """Run `cmd` (its parts may be paths) and capture its output as text."""
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True, **kwargs)
