"""Tests for the CLI workflow orchestration."""

from pathlib import Path

from nanovt import cli


def test_main_runs_transcription_workflow(tmp_path: Path, monkeypatch) -> None:
    """Run the main workflow with audio and API work replaced by fakes."""
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"video")
    calls: list[tuple[str, object]] = []

    def fake_extract_audio(source_path: Path, audio_path: Path) -> None:
        calls.append(("extract", (source_path, audio_path.name)))
        audio_path.write_bytes(b"audio")

    def fake_split_audio(
        audio_path: Path,
        chunk_seconds: int,
        chunk_dir: Path,
    ) -> list[Path]:
        calls.append(("split", (audio_path.name, chunk_seconds, chunk_dir.name)))
        first_chunk = chunk_dir / "chunk_0000.wav"
        second_chunk = chunk_dir / "chunk_0001.wav"
        first_chunk.write_bytes(b"first")
        second_chunk.write_bytes(b"second")
        return [first_chunk, second_chunk]

    def fake_transcribe_chunks(
        chunks: list[Path],
        client: object,
        model: str,
        language: str | None,
        retries: int,
        diarize: bool,
    ) -> list[str]:
        calls.append(("transcribe", ([chunk.name for chunk in chunks], model, retries)))
        assert client == "client"
        assert language is None
        assert not diarize
        return ["hello", "world"]

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(cli, "_build_openai_client", lambda api_key: "client")
    monkeypatch.setattr(cli, "_extract_audio", fake_extract_audio)
    monkeypatch.setattr(cli, "_split_audio", fake_split_audio)
    monkeypatch.setattr(cli, "_transcribe_chunks", fake_transcribe_chunks)

    cli.main([str(input_path), "--chunk-seconds", "60"])

    assert input_path.with_suffix(".txt").read_text(encoding="utf-8") == (
        "hello\n\nworld\n"
    )
    assert calls == [
        ("extract", (input_path.resolve(), "source.wav")),
        ("split", ("source.wav", 60, "chunks")),
        (
            "transcribe",
            (["chunk_0000.wav", "chunk_0001.wav"], cli.DEFAULT_TRANSCRIPTION_MODEL, 3),
        ),
    ]
