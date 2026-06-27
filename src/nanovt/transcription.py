"""Transcribe audio chunks with OpenAI speech-to-text models."""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import BinaryIO, Literal, Protocol, cast, overload

_SPEAKER_LABELS = tuple(chr(code) for code in range(ord("A"), ord("Z") + 1))


class _DiarizedSegment(Protocol):
    speaker: str
    text: str


class _TextTranscription(Protocol):
    text: str


class _DiarizedTranscription(Protocol):
    segments: Sequence[_DiarizedSegment]


class _TranscriptionsClient(Protocol):
    """Protocol for the OpenAI audio transcriptions API used by this command."""

    @overload
    def create(
        self,
        *,
        file: BinaryIO,
        model: str,
        language: str,
        response_format: Literal["json"],
    ) -> _TextTranscription: ...

    @overload
    def create(
        self,
        *,
        file: BinaryIO,
        model: str,
        response_format: Literal["json"],
    ) -> _TextTranscription: ...

    @overload
    def create(
        self,
        *,
        file: BinaryIO,
        model: str,
        language: str,
        response_format: Literal["diarized_json"],
        chunking_strategy: Literal["auto"],
    ) -> _DiarizedTranscription: ...

    @overload
    def create(
        self,
        *,
        file: BinaryIO,
        model: str,
        response_format: Literal["diarized_json"],
        chunking_strategy: Literal["auto"],
    ) -> _DiarizedTranscription: ...


class _AudioClient(Protocol):
    """Protocol for the OpenAI audio client namespace."""

    transcriptions: _TranscriptionsClient


class _OpenAIClient(Protocol):
    """Protocol for the subset of the OpenAI client used here."""

    audio: _AudioClient


def _build_openai_client(api_key: str) -> _OpenAIClient:
    """Build an OpenAI SDK client.

    Args:
        api_key: OpenAI API key.

    Returns:
        OpenAI client with the audio transcription API.

    Raises:
        SystemExit: If the OpenAI SDK is not installed.
    """
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("OpenAI SDK is not installed. Reinstall nanovt.") from exc

    return cast(_OpenAIClient, OpenAI(api_key=api_key))


def _transcribe_chunk(
    chunk_path: Path,
    client: _OpenAIClient,
    model: str,
    language: str | None,
    retries: int,
    diarize: bool,
    speaker_labels: dict[str, str],
) -> str:
    """Transcribe one audio chunk with retry for transient failures.

    Args:
        chunk_path: WAV chunk path.
        client: OpenAI SDK client.
        model: Transcription model name.
        language: Optional input language code.
        retries: Number of retries after the first attempt.
        diarize: Whether to request speaker diarization.
        speaker_labels: Stable speaker label mapping across chunks.

    Returns:
        Transcribed text.

    Raises:
        RuntimeError: If transcription fails after retries.
    """
    for attempt in range(1, retries + 2):
        try:
            with chunk_path.open("rb") as audio_file:
                if diarize:
                    transcript = _create_diarized_transcription(
                        audio_file, client, model, language
                    )
                    return _format_diarized_transcript(transcript, speaker_labels)

                transcript = _create_text_transcription(
                    audio_file, client, model, language
                )
                return transcript.strip()
        except Exception as exc:
            if not _should_retry_openai_error(exc) or attempt > retries:
                raise RuntimeError(str(exc)) from exc

            wait_seconds = min(30, 2**attempt)
            print(f"  {exc}; retrying in {wait_seconds}s")
            time.sleep(wait_seconds)

    raise RuntimeError("Transcription failed after retries.")


def _create_text_transcription(
    audio_file: BinaryIO,
    client: _OpenAIClient,
    model: str,
    language: str | None,
) -> str:
    if language:
        transcript = client.audio.transcriptions.create(
            file=audio_file,
            model=model,
            language=language,
            response_format="json",
        )
        return transcript.text

    transcript = client.audio.transcriptions.create(
        file=audio_file,
        model=model,
        response_format="json",
    )
    return transcript.text


def _create_diarized_transcription(
    audio_file: BinaryIO,
    client: _OpenAIClient,
    model: str,
    language: str | None,
) -> _DiarizedTranscription:
    if language:
        return client.audio.transcriptions.create(
            file=audio_file,
            model=model,
            language=language,
            response_format="diarized_json",
            chunking_strategy="auto",
        )

    return client.audio.transcriptions.create(
        file=audio_file,
        model=model,
        response_format="diarized_json",
        chunking_strategy="auto",
    )


def _format_diarized_transcript(
    transcript: _DiarizedTranscription,
    speaker_labels: dict[str, str],
) -> str:
    lines: list[str] = []
    for segment in transcript.segments:
        speaker = segment.speaker
        if speaker not in speaker_labels:
            label_index = len(speaker_labels)
            speaker_labels[speaker] = (
                _SPEAKER_LABELS[label_index]
                if label_index < len(_SPEAKER_LABELS)
                else speaker
            )

        text = segment.text.strip()
        if text:
            lines.append(f"{speaker_labels[speaker]}: {text}")
    return "\n".join(lines)


def _should_retry_openai_error(exc: Exception) -> bool:
    """Return whether an OpenAI SDK error should be retried.

    Args:
        exc: Exception raised by the OpenAI SDK.

    Returns:
        True for transient connection, timeout, rate-limit, and server errors.
    """
    status_code = getattr(exc, "status_code", None)
    if status_code in {408, 409, 429}:
        return True
    if isinstance(status_code, int) and status_code >= 500:
        return True
    return exc.__class__.__name__ in {"APIConnectionError", "APITimeoutError"}


def _transcribe_chunks(
    chunks: list[Path],
    client: _OpenAIClient,
    model: str,
    language: str | None,
    retries: int,
    diarize: bool,
) -> list[str]:
    """Transcribe chunks sequentially.

    Args:
        chunks: Ordered chunk paths.
        client: OpenAI SDK client.
        model: Transcription model name.
        language: Optional input language code.
        retries: Number of retries per chunk.
        diarize: Whether to request speaker diarization.

    Returns:
        Transcript text for each chunk.
    """
    transcripts: list[str] = []
    speaker_labels: dict[str, str] = {}
    for index, chunk in enumerate(chunks, start=1):
        print(f"Transcribing {index}/{len(chunks)}: {chunk.name}")
        text = _transcribe_chunk(
            chunk, client, model, language, retries, diarize, speaker_labels
        )
        if not text:
            print(f"  Warning: empty transcript for {chunk.name}")
        transcripts.append(text)
    return transcripts
