"""Tests for CLI argument parsing and package exports."""

from pathlib import Path

import nanovt
from nanovt import main
from nanovt.cli import (
    DEFAULT_DIARIZATION_MODEL,
    DEFAULT_TRANSCRIPTION_MODEL,
    parse_args,
)


def test_parse_args_accepts_input_and_chunk_seconds() -> None:
    """Parse a minimal command with an explicit chunk duration."""
    args = parse_args(["input.mp4", "--chunk-seconds", "60"])

    assert args.input == Path("input.mp4")
    assert args.chunk_seconds == 60
    assert args.model == DEFAULT_TRANSCRIPTION_MODEL
    assert not args.diarize


def test_parse_args_uses_diarization_model_when_requested() -> None:
    """Use the diarization model when diarization is requested."""
    args = parse_args(["input.mp4", "--diarize"])

    assert args.diarize
    assert args.model == DEFAULT_DIARIZATION_MODEL


def test_parse_args_keeps_explicit_model_with_diarization() -> None:
    """Keep an explicit model when diarization is requested."""
    args = parse_args(["input.mp4", "--diarize", "--model", "custom-model"])

    assert args.model == "custom-model"


def test_parse_args_defaults_to_openai_provider() -> None:
    """Select the OpenAI provider unless another one is requested."""
    args = parse_args(["input.mp4"])

    assert args.provider == "openai"


def test_parse_args_uses_qwen_model_for_qwen_provider() -> None:
    """Use qwen3-asr-flash when the qwen provider is requested."""
    args = parse_args(["input.mp4", "--provider", "qwen"])

    assert args.provider == "qwen"
    assert args.model == "qwen3-asr-flash"


def test_parse_args_uses_volc_model_for_volc_provider() -> None:
    """Use the Volcengine model when the Volcengine provider is requested."""
    args = parse_args(["input.mp4", "--provider", "volc"])

    assert args.provider == "volc"
    assert args.model == "bigmodel"


def test_parse_args_keeps_explicit_model_with_volc_provider() -> None:
    """Keep an explicit model with the Volcengine provider."""
    args = parse_args(["input.mp4", "--provider", "volc", "--model", "custom-model"])

    assert args.model == "custom-model"


def test_package_exports_version_and_main() -> None:
    """Expose package version and callable main entry point."""
    assert isinstance(nanovt.__version__, str)
    assert main is nanovt.main
