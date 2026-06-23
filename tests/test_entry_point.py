"""Tests for module execution entry points."""

import subprocess
import sys


def test_python_module_help() -> None:
    """Show help through python -m nanovt."""
    result = subprocess.run(
        [sys.executable, "-m", "nanovt", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "usage:" in result.stdout
    assert "nanovt input.mp4" in result.stdout
