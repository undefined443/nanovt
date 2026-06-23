"""Tests for CLI argument parsing and package exports."""

from pathlib import Path

import video_transcriber
from video_transcriber import main
from video_transcriber.cli import parse_args


def test_parse_args_accepts_input_and_chunk_seconds() -> None:
    """Parse a minimal command with an explicit chunk duration."""
    args = parse_args(["input.mp4", "--chunk-seconds", "60"])

    assert args.input == Path("input.mp4")
    assert args.chunk_seconds == 60


def test_package_exports_version_and_main() -> None:
    """Expose package version and callable main entry point."""
    assert isinstance(video_transcriber.__version__, str)
    assert main is video_transcriber.main
