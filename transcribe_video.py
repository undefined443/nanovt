"""Transcribe a video or audio file with OpenAI's speech-to-text API.

The script extracts a normalized WAV audio stream with ffmpeg, splits it into
short chunks, transcribes each chunk, and concatenates the transcript text.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import BinaryIO, Literal, NotRequired, Protocol, TypedDict, cast, overload


class _TranscriptionCreateArgs(TypedDict):
    """Keyword arguments for creating an OpenAI transcription."""

    file: BinaryIO
    model: str
    response_format: Literal["text"]
    language: NotRequired[str]


class _TranscriptionsClient(Protocol):
    """Protocol for the OpenAI audio transcriptions API used by this script."""

    @overload
    def create(
        self,
        *,
        file: BinaryIO,
        model: str,
        language: str,
        response_format: Literal["text"],
    ) -> str: ...

    @overload
    def create(
        self,
        *,
        file: BinaryIO,
        model: str,
        response_format: Literal["text"],
    ) -> str: ...


class _AudioClient(Protocol):
    """Protocol for the OpenAI audio client namespace."""

    transcriptions: _TranscriptionsClient


class _OpenAIClient(Protocol):
    """Protocol for the subset of the OpenAI client used here."""

    audio: _AudioClient


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Extract audio from a media file, chunk it, and transcribe it with OpenAI."
        ),
        epilog="Run with: uv run python transcribe_video.py input.mp4",
    )
    parser.add_argument("input", type=Path, help="Input video or audio file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output transcript path. Defaults to <input_stem>.txt.",
    )
    parser.add_argument(
        "--model", default="gpt-4o-transcribe", help="OpenAI transcription model."
    )
    parser.add_argument(
        "--language",
        help=(
            "Optional input language code, such as zh. Defaults to automatic detection."
        ),
    )
    parser.add_argument(
        "--chunk-seconds",
        type=int,
        default=180,
        help="Chunk duration in seconds. Default: 180.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retries per chunk after a failed API request. Default: 3.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep extracted audio and chunk files after completion.",
    )
    parser.add_argument(
        "--api-key",
        help="OpenAI API key. Defaults to OPENAI_API_KEY or ~/.zshenv.",
    )
    return parser.parse_args()


def _load_api_key(explicit_key: str | None) -> str:
    """Load the OpenAI API key.

    Args:
        explicit_key: API key passed through the command line.

    Returns:
        The resolved API key.

    Raises:
        SystemExit: If no API key is available.
    """
    if explicit_key:
        return explicit_key

    env_key = os.environ.get("OPENAI_API_KEY")
    if env_key:
        return env_key

    zshenv = Path.home() / ".zshenv"
    if zshenv.exists():
        for line in zshenv.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("export OPENAI_API_KEY="):
                return _clean_shell_value(stripped.split("=", 1)[1])
            if stripped.startswith("OPENAI_API_KEY="):
                return _clean_shell_value(stripped.split("=", 1)[1])

    raise SystemExit("OPENAI_API_KEY is not set. Export it or pass --api-key.")


def _clean_shell_value(value: str) -> str:
    """Strip simple shell quoting from a variable value.

    Args:
        value: Raw value read from a shell environment file.

    Returns:
        The unquoted value when simple matching quotes are present.
    """
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _run_ffmpeg(args: list[str]) -> None:
    """Run ffmpeg with quiet logging.

    Args:
        args: Arguments appended after ffmpeg's global quiet options.

    Raises:
        SystemExit: If ffmpeg exits with a non-zero status.
    """
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "ffmpeg failed"
        raise SystemExit(message)


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
        raise SystemExit(
            "OpenAI SDK is not installed. Run with: uv run transcribe_video.py ..."
        ) from exc

    return cast(_OpenAIClient, OpenAI(api_key=api_key))


def _extract_audio(input_path: Path, audio_path: Path) -> None:
    """Extract normalized WAV audio from a media file.

    Args:
        input_path: Source video or audio file.
        audio_path: Destination WAV file path.
    """
    print(f"Extracting audio -> {audio_path}")
    _run_ffmpeg(
        ["-i", str(input_path), "-vn", "-ac", "1", "-ar", "16000", str(audio_path)]
    )


def _split_audio(audio_path: Path, chunk_seconds: int, chunk_dir: Path) -> list[Path]:
    """Split a WAV file into fixed-duration chunks.

    Args:
        audio_path: Source WAV file.
        chunk_seconds: Duration of each chunk in seconds.
        chunk_dir: Directory for generated chunk files.

    Returns:
        Sorted chunk file paths.

    Raises:
        SystemExit: If ffmpeg does not generate any chunks.
    """
    print(f"Splitting audio into {chunk_seconds}s chunks")
    pattern = chunk_dir / "chunk_%04d.wav"
    _run_ffmpeg(
        [
            "-i",
            str(audio_path),
            "-f",
            "segment",
            "-segment_time",
            str(chunk_seconds),
            "-c",
            "copy",
            str(pattern),
        ]
    )
    chunks = sorted(chunk_dir.glob("chunk_*.wav"))
    if not chunks:
        raise SystemExit("No chunks were created.")
    return chunks


def _transcribe_chunk(
    chunk_path: Path,
    client: _OpenAIClient,
    model: str,
    language: str | None,
    retries: int,
) -> str:
    """Transcribe one audio chunk with retry for transient failures.

    Args:
        chunk_path: WAV chunk path.
        client: OpenAI SDK client.
        model: Transcription model name.
        language: Optional input language code.
        retries: Number of retries after the first attempt.

    Returns:
        Transcribed text.

    Raises:
        RuntimeError: If transcription fails after retries.
    """
    for attempt in range(1, retries + 2):
        try:
            with chunk_path.open("rb") as audio_file:
                transcription_args: _TranscriptionCreateArgs = {
                    "file": audio_file,
                    "model": model,
                    "response_format": "text",
                }
                if language:
                    transcription_args["language"] = language

                transcript = client.audio.transcriptions.create(**transcription_args)
            return str(transcript).strip()
        except Exception as exc:
            if not _should_retry_openai_error(exc) or attempt > retries:
                raise RuntimeError(str(exc)) from exc

            wait_seconds = min(30, 2**attempt)
            print(f"  {exc}; retrying in {wait_seconds}s")
            time.sleep(wait_seconds)

    raise RuntimeError("Transcription failed after retries.")


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
) -> list[str]:
    """Transcribe chunks sequentially.

    Args:
        chunks: Ordered chunk paths.
        client: OpenAI SDK client.
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


def _write_transcript(output_path: Path, transcripts: list[str]) -> None:
    """Write the final transcript file.

    Args:
        output_path: Destination text file.
        transcripts: Chunk transcript texts.
    """
    content = "\n\n".join(text for text in transcripts if text)
    output_path.write_text(content + "\n", encoding="utf-8")
    print(f"Wrote transcript -> {output_path}")


def main() -> None:
    """Run the video-to-transcript command-line workflow."""
    args = _parse_args()
    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"Input file does not exist: {input_path}")
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is not installed or not on PATH.")
    if args.chunk_seconds <= 0:
        raise SystemExit("--chunk-seconds must be positive.")

    api_key = _load_api_key(args.api_key)
    client = _build_openai_client(api_key)
    output_path = args.output or input_path.with_suffix(".txt")
    output_path = output_path.expanduser().resolve()

    temp_dir = Path(tempfile.mkdtemp(prefix=f"transcribe_{input_path.stem}_"))
    audio_path = temp_dir / "source.wav"
    chunk_dir = temp_dir / "chunks"
    chunk_dir.mkdir()

    try:
        _extract_audio(input_path, audio_path)
        chunks = _split_audio(audio_path, args.chunk_seconds, chunk_dir)
        transcripts = _transcribe_chunks(
            chunks, client, args.model, args.language, args.retries
        )
        _write_transcript(output_path, transcripts)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print(f"Temporary files kept at: {temp_dir}", file=sys.stderr)
        raise SystemExit(1) from exc
    else:
        if args.keep_temp:
            print(f"Temporary files kept at: {temp_dir}")
        else:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()
