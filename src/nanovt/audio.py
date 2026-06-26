"""Extract and split audio streams with ffmpeg."""

from __future__ import annotations

import subprocess
from pathlib import Path


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
