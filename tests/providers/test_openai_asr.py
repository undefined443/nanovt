"""Tests for the OpenAI speech-to-text provider."""

from pathlib import Path
from typing import BinaryIO, Literal, cast

from nanovt.cli import DEFAULT_DIARIZATION_MODEL, DEFAULT_TRANSCRIPTION_MODEL
from nanovt.providers.openai_asr import (
    _OpenAIClient,
    _should_retry_openai_error,
    _transcribe_chunk,
)


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


def test_transcribe_chunk_formats_diarized_dialogue(tmp_path: Path) -> None:
    """Format diarized segments as speaker-labeled dialogue lines."""
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


def test_transcribe_chunk_formats_more_than_two_speakers(tmp_path: Path) -> None:
    """Keep transcribing when diarization returns extra speakers."""
    chunk_path = tmp_path / "chunk.wav"
    chunk_path.write_bytes(b"audio")
    response = _DiarizedResponse(
        [
            _Segment("speaker_0", "Hello."),
            _Segment("speaker_1", "Hi."),
            _Segment("speaker_2", "Question."),
        ]
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

    assert text == "A: Hello.\nB: Hi.\nC: Question."


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


class _StatusError(Exception):
    """Test exception with an OpenAI-style status code."""

    def __init__(self, status_code: int) -> None:
        """Initialize the fake status error."""
        self.status_code = status_code


class APIConnectionError(Exception):
    """Test exception matching the OpenAI connection error class name."""


class APITimeoutError(Exception):
    """Test exception matching the OpenAI timeout error class name."""


def test_retries_transient_status_codes() -> None:
    """Retry rate-limit, timeout, conflict, and server errors."""
    assert _should_retry_openai_error(_StatusError(408))
    assert _should_retry_openai_error(_StatusError(409))
    assert _should_retry_openai_error(_StatusError(429))
    assert _should_retry_openai_error(_StatusError(500))


def test_does_not_retry_non_transient_status_code() -> None:
    """Do not retry non-transient client errors."""
    assert not _should_retry_openai_error(_StatusError(400))


def test_retries_openai_connection_error_names() -> None:
    """Retry OpenAI SDK connection and timeout error class names."""
    assert _should_retry_openai_error(APIConnectionError())
    assert _should_retry_openai_error(APITimeoutError())
