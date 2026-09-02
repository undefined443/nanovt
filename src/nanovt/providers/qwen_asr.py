"""Transcribe audio chunks with Alibaba DashScope ``qwen3-asr-flash``.

``qwen3-asr-flash`` is reached through the DashScope OpenAI-compatible chat
endpoint: each request carries one ``input_audio`` part and the transcript comes
back as the assistant message. The OpenAI SDK is reused with a DashScope base URL
and the ``DASHSCOPE_API_KEY`` credential. The model has no speaker diarization and
caps each request near five minutes of audio, so the shared chunking pipeline
feeds it fixed-duration WAV segments.
"""

from __future__ import annotations

import base64
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, cast

from nanovt.providers.openai_asr import _should_retry_openai_error

_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class _Message(Protocol):
    content: str | None


class _Choice(Protocol):
    message: _Message


class _ChatCompletion(Protocol):
    choices: list[_Choice]


class _CompletionsClient(Protocol):
    """Protocol for the OpenAI-compatible chat completions API."""

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, object]],
        extra_body: Mapping[str, object] | None,
    ) -> _ChatCompletion: ...


class _ChatClient(Protocol):
    """Protocol for the OpenAI chat client namespace."""

    completions: _CompletionsClient


class _QwenClient(Protocol):
    """Protocol for the subset of the OpenAI client used here."""

    chat: _ChatClient


def _build_qwen_client(api_key: str) -> _QwenClient:
    """Build an OpenAI SDK client aimed at the DashScope compatible endpoint.

    Args:
        api_key: DashScope API key.

    Returns:
        OpenAI client whose base URL targets DashScope.

    Raises:
        SystemExit: If the OpenAI SDK is not installed.
    """
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("OpenAI SDK is not installed. Reinstall nanovt.") from exc

    return cast(_QwenClient, OpenAI(api_key=api_key, base_url=_BASE_URL))


def _audio_message(chunk_path: Path) -> dict[str, object]:
    """Wrap one base64-encoded audio chunk in an OpenAI-style user message.

    Args:
        chunk_path: WAV chunk path.

    Returns:
        A message with a single ``input_audio`` content part.
    """
    data = base64.b64encode(chunk_path.read_bytes()).decode("ascii")
    return {
        "role": "user",
        "content": [
            {
                "type": "input_audio",
                "input_audio": {
                    "data": f"data:audio/wav;base64,{data}",
                    "format": "wav",
                },
            }
        ],
    }


def _transcribe_chunk(
    chunk_path: Path,
    client: _QwenClient,
    model: str,
    language: str | None,
    retries: int,
) -> str:
    """Transcribe one audio chunk with retry for transient failures.

    Args:
        chunk_path: WAV chunk path.
        client: OpenAI SDK client targeting DashScope.
        model: Transcription model name.
        language: Optional input language code sent as an ASR option.
        retries: Number of retries after the first attempt.

    Returns:
        Transcribed text.

    Raises:
        RuntimeError: If transcription fails after retries.
    """
    messages = [_audio_message(chunk_path)]
    extra_body = {"asr_options": {"language": language}} if language else None
    for attempt in range(1, retries + 2):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                extra_body=extra_body,
            )
            return (completion.choices[0].message.content or "").strip()
        except Exception as exc:
            if not _should_retry_openai_error(exc) or attempt > retries:
                raise RuntimeError(str(exc)) from exc

            wait_seconds = min(30, 2**attempt)
            print(f"  {exc}; retrying in {wait_seconds}s")
            time.sleep(wait_seconds)

    raise RuntimeError("Transcription failed after retries.")


def _transcribe_chunks(
    chunks: list[Path],
    client: _QwenClient,
    model: str,
    language: str | None,
    retries: int,
) -> list[str]:
    """Transcribe chunks sequentially.

    Args:
        chunks: Ordered chunk paths.
        client: OpenAI SDK client targeting DashScope.
        model: Transcription model name.
        language: Optional input language code.
        retries: Number of retries per chunk.

    Returns:
        Transcript text for each chunk.
    """
    transcripts: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        print(f"Transcribing {index}/{len(chunks)}: {chunk.name}")
        text = _transcribe_chunk(chunk, client, model, language, retries)
        if not text:
            print(f"  Warning: empty transcript for {chunk.name}")
        transcripts.append(text)
    return transcripts
