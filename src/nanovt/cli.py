"""Transcribe a video or audio file with a speech-to-text API.

The OpenAI provider extracts a normalized WAV stream with ffmpeg, splits it into
short chunks, and transcribes each chunk. The Volcengine provider compresses the
stream to MP3 and submits it whole to an asynchronous job.
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

from nanovt.audio import _extract_audio_mp3, _extract_audio_wav, _split_audio
from nanovt.providers.openai_asr import _build_openai_client, _transcribe_chunks
from nanovt.providers.volc_asr import _transcribe_volc

DEFAULT_TRANSCRIPTION_MODEL = "gpt-4o-transcribe"
DEFAULT_DIARIZATION_MODEL = "gpt-4o-transcribe-diarize"
DEFAULT_VOLC_MODEL = "bigmodel"
DEFAULT_VOLC_RESOURCE_ID = "volc.seedasr.auc"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument sequence. Defaults to process arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Extract audio from a media file and transcribe it with OpenAI or "
            "Volcengine."
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
    parser.add_argument(
        "--provider",
        choices=["openai", "volc"],
        default="openai",
        help="Speech-to-text provider. Default: openai.",
    )
    parser.add_argument("--model", help="Provider transcription model.")
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
        if args.provider == "volc":
            args.model = DEFAULT_VOLC_MODEL
        else:
            args.model = (
                DEFAULT_DIARIZATION_MODEL
                if args.diarize
                else DEFAULT_TRANSCRIPTION_MODEL
            )
    return args


def _load_api_key(provider: str) -> str:
    """Load the provider API key from the environment.

    Args:
        provider: Speech-to-text provider name.

    Returns:
        The API key for the selected provider.

    Raises:
        SystemExit: If the provider's API key environment variable is not set.
    """
    env_name = "VOLC_ASR_API_KEY" if provider == "volc" else "OPENAI_API_KEY"
    env_key = os.environ.get(env_name)
    if env_key:
        return env_key

    raise SystemExit(f"{env_name} is not set.")


def _write_transcript(output_path: Path, transcripts: list[str]) -> None:
    """Write the final transcript file.

    Args:
        output_path: Destination text file.
        transcripts: Chunk transcript texts.
    """
    content = "\n\n".join(text for text in transcripts if text)
    output_path.write_text(content + "\n", encoding="utf-8")
    print(f"Wrote transcript -> {output_path}")


def _run_openai(
    input_path: Path,
    temp_dir: Path,
    args: argparse.Namespace,
    api_key: str,
) -> list[str]:
    """Extract WAV audio, split it into chunks, and transcribe with OpenAI.

    Args:
        input_path: Source media file.
        temp_dir: Working directory for extracted audio and chunks.
        args: Parsed command-line arguments.
        api_key: OpenAI API key.

    Returns:
        Chunk transcript texts.
    """
    audio_path = temp_dir / "source.wav"
    chunk_dir = temp_dir / "chunks"
    chunk_dir.mkdir()
    _extract_audio_wav(input_path, audio_path)
    chunks = _split_audio(audio_path, args.chunk_seconds, chunk_dir)
    client = _build_openai_client(api_key)
    return _transcribe_chunks(
        chunks, client, args.model, args.language, args.retries, args.diarize
    )


def _run_volc(
    input_path: Path,
    temp_dir: Path,
    args: argparse.Namespace,
    api_key: str,
) -> list[str]:
    """Extract compressed MP3 audio and transcribe it with Volcengine.

    Args:
        input_path: Source media file.
        temp_dir: Working directory for extracted audio.
        args: Parsed command-line arguments.
        api_key: Volcengine API key.

    Returns:
        A single-element list holding the whole-file transcript.
    """
    audio_path = temp_dir / "source.mp3"
    _extract_audio_mp3(input_path, audio_path)
    resource_id = os.environ.get("VOLC_ASR_RESOURCE_ID", DEFAULT_VOLC_RESOURCE_ID)
    transcript = _transcribe_volc(
        audio_path,
        api_key,
        resource_id,
        args.model,
        args.language,
        args.diarize,
        args.retries,
    )
    return [transcript]


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

    api_key = _load_api_key(args.provider)
    output_path = args.output or input_path.with_suffix(".txt")
    output_path = output_path.expanduser().resolve()

    temp_dir = Path(tempfile.mkdtemp(prefix=f"transcribe_{input_path.stem}_"))
    runner = _run_volc if args.provider == "volc" else _run_openai

    try:
        transcripts = runner(input_path, temp_dir, args, api_key)
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
