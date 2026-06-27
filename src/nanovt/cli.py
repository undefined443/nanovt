"""Transcribe a video or audio file with OpenAI's speech-to-text API.

The command extracts a normalized WAV audio stream with ffmpeg, splits it into
short chunks, transcribes each chunk, and concatenates the transcript text.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import shutil
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from nanovt.audio import _extract_audio, _split_audio
from nanovt.transcription import _build_openai_client, _transcribe_chunks

DEFAULT_TRANSCRIPTION_MODEL = "gpt-4o-transcribe"
DEFAULT_DIARIZATION_MODEL = "gpt-4o-transcribe-diarize"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument sequence. Defaults to process arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Extract audio from a media file, chunk it, and transcribe it with OpenAI."
        ),
        epilog="Run with: nanovt input.mp4",
    )
    parser.add_argument("input", type=Path, help="Input video or audio file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output transcript path. Defaults to <input_stem>.txt.",
    )
    parser.add_argument("--model", help="OpenAI transcription model.")
    parser.add_argument(
        "--diarize",
        action="store_true",
        help=(
            "Use the diarization transcription model and write speaker-labeled "
            "dialogue lines."
        ),
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
        "-v",
        "--version",
        action="version",
        version=f"nanovt {importlib.metadata.version('nanovt')}",
    )
    args = parser.parse_args(argv)
    if args.model is None:
        args.model = (
            DEFAULT_DIARIZATION_MODEL if args.diarize else DEFAULT_TRANSCRIPTION_MODEL
        )
    return args


def _load_api_key() -> str:
    """Load the OpenAI API key.

    Returns:
        The OpenAI API key from the environment.

    Raises:
        SystemExit: If OPENAI_API_KEY is not set.
    """
    env_key = os.environ.get("OPENAI_API_KEY")
    if env_key:
        return env_key

    raise SystemExit("OPENAI_API_KEY is not set.")


def _write_transcript(output_path: Path, transcripts: list[str]) -> None:
    """Write the final transcript file.

    Args:
        output_path: Destination text file.
        transcripts: Chunk transcript texts.
    """
    content = "\n\n".join(text for text in transcripts if text)
    output_path.write_text(content + "\n", encoding="utf-8")
    print(f"Wrote transcript -> {output_path}")


def main(argv: Sequence[str] | None = None) -> None:
    """Run the video-to-transcript command-line workflow.

    Args:
        argv: Optional argument sequence. Defaults to process arguments.
    """
    args = parse_args(argv)
    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"Input file does not exist: {input_path}")
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is not installed or not on PATH.")
    if args.chunk_seconds <= 0:
        raise SystemExit("--chunk-seconds must be positive.")

    api_key = _load_api_key()
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
            chunks, client, args.model, args.language, args.retries, args.diarize
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
