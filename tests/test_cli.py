"""Tests for CLI argument parsing and package exports."""

from pathlib import Path
from typing import BinaryIO, Literal, cast

import nanovt
from nanovt import main
from nanovt.cli import (
    DEFAULT_DIARIZATION_MODEL,
    DEFAULT_TRANSCRIPTION_MODEL,
    parse_args,
)
from nanovt.transcription import _OpenAIClient, _transcribe_chunk


class _Segment:
    def __init__(self, speaker: str, text: str) -> None:
        self.speaker = speaker
        self.text = text


class _DiarizedResponse:
    def __init__(self, segments: list[_Segment]) -> None:
        self.segments = segments


class _TextResponse:
    text = "plain text"


class _Transcriptions:
    def __init__(self, response: _DiarizedResponse) -> None:
        self.response = response
        self.model: str | None = None
        self.response_format: str | None = None

    def create(
        self,
        *,
        file: BinaryIO,
        model: str,
        response_format: Literal["json", "diarized_json"],
        language: str | None = None,
        chunking_strategy: Literal["auto"] | None = None,
    ) -> _TextResponse | _DiarizedResponse:
        self.model = model
        self.response_format = response_format
        self.chunking_strategy = chunking_strategy
        if response_format == "json":
            return _TextResponse()
        return self.response


class _Audio:
    def __init__(self, transcriptions: _Transcriptions) -> None:
        self.transcriptions = transcriptions


class _Client:
    def __init__(self, transcriptions: _Transcriptions) -> None:
        self.audio = _Audio(transcriptions)


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


def test_transcribe_chunk_formats_diarized_dialogue(tmp_path: Path) -> None:
    """Format diarized segments as two-speaker dialogue lines."""
    chunk_path = tmp_path / "chunk.wav"
    chunk_path.write_bytes(b"audio")
    response = _DiarizedResponse(
        [_Segment("speaker_0", "Hello."), _Segment("speaker_1", "Hi there.")]
    )
    transcriptions = _Transcriptions(response)
    client = cast(_OpenAIClient, _Client(transcriptions))

    text = _transcribe_chunk(
        chunk_path,
        client,
        DEFAULT_DIARIZATION_MODEL,
        None,
        0,
        True,
        {},
    )

    assert text == "A: Hello.\nB: Hi there."
    assert transcriptions.model == DEFAULT_DIARIZATION_MODEL
    assert transcriptions.response_format == "diarized_json"
    assert transcriptions.chunking_strategy == "auto"


def test_transcribe_chunk_uses_json_for_default_model(tmp_path: Path) -> None:
    """Use JSON transcription output for the default GPT-4o model."""
    chunk_path = tmp_path / "chunk.wav"
    chunk_path.write_bytes(b"audio")
    transcriptions = _Transcriptions(_DiarizedResponse([]))
    client = cast(_OpenAIClient, _Client(transcriptions))

    text = _transcribe_chunk(
        chunk_path,
        client,
        DEFAULT_TRANSCRIPTION_MODEL,
        None,
        0,
        False,
        {},
    )

    assert text == "plain text"
    assert transcriptions.model == DEFAULT_TRANSCRIPTION_MODEL
    assert transcriptions.response_format == "json"
    assert transcriptions.chunking_strategy is None


def test_package_exports_version_and_main() -> None:
    """Expose package version and callable main entry point."""
    assert isinstance(nanovt.__version__, str)
    assert main is nanovt.main
