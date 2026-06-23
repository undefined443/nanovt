"""Tests for module execution entry points."""

import subprocess
import sys


def test_python_module_help() -> None:
    """Show help through python -m video_transcriber."""
    result = subprocess.run(
        [sys.executable, "-m", "video_transcriber", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "usage:" in result.stdout
    assert "vt input.mp4" in result.stdout
