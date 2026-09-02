"""Tests for the CLI workflow orchestration."""

from pathlib import Path

import pytest

from nanovt import cli


def test_main_runs_transcription_workflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    monkeypatch.setattr(cli.shutil, "which", lambda _name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(cli, "_build_openai_client", lambda _api_key: "client")
    monkeypatch.setattr(cli, "_extract_audio_wav", fake_extract_audio)
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


def test_main_runs_qwen_workflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run the qwen3-asr-flash path: WAV chunks in, joined transcript out."""
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
        chunk = chunk_dir / "chunk_0000.wav"
        chunk.write_bytes(b"first")
        return [chunk]

    def fake_transcribe_chunks(
        chunks: list[Path],
        client: object,
        model: str,
        language: str | None,
        retries: int,
    ) -> list[str]:
        calls.append(
            ("transcribe", ([chunk.name for chunk in chunks], model, language, retries))
        )
        assert client == "client"
        return ["你好世界"]

    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setattr(cli.shutil, "which", lambda _name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(cli, "_build_qwen_client", lambda _api_key: "client")
    monkeypatch.setattr(cli, "_extract_audio_wav", fake_extract_audio)
    monkeypatch.setattr(cli, "_split_audio", fake_split_audio)
    monkeypatch.setattr(cli, "_transcribe_qwen_chunks", fake_transcribe_chunks)

    cli.main([str(input_path), "--provider", "qwen", "--language", "zh"])

    assert input_path.with_suffix(".txt").read_text(encoding="utf-8") == "你好世界\n"
    assert calls == [
        ("extract", (input_path.resolve(), "source.wav")),
        ("split", ("source.wav", 180, "chunks")),
        ("transcribe", (["chunk_0000.wav"], cli.DEFAULT_QWEN_MODEL, "zh", 3)),
    ]


def test_main_rejects_qwen_diarization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """qwen3-asr-flash has no diarization, so --diarize exits with an error."""
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"video")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setattr(cli.shutil, "which", lambda _name: "/usr/bin/ffmpeg")

    with pytest.raises(SystemExit):
        cli.main([str(input_path), "--provider", "qwen", "--diarize"])


def test_main_runs_volc_workflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run the Volcengine path: compressed MP3 in, whole-file transcript out."""
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"video")
    calls: list[tuple[str, object]] = []

    def fake_extract_audio_mp3(source_path: Path, audio_path: Path) -> None:
        calls.append(("extract", (source_path, audio_path.name)))
        audio_path.write_bytes(b"audio")

    def fake_transcribe_volc(
        audio_path: Path,
        api_key: str,
        resource_id: str,
        model: str,
        language: str | None,
        diarize: bool,
        retries: int,
    ) -> str:
        calls.append(("transcribe", (audio_path.name, resource_id, model, language)))
        assert api_key == "test-key"
        assert diarize
        assert retries == 3
        return "A: 你好\nB: 在的"

    monkeypatch.setenv("VOLC_ASR_API_KEY", "test-key")
    monkeypatch.setattr(cli.shutil, "which", lambda _name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(cli, "_extract_audio_mp3", fake_extract_audio_mp3)
    monkeypatch.setattr(cli, "_transcribe_volc", fake_transcribe_volc)

    cli.main([str(input_path), "--provider", "volc", "--language", "zh", "--diarize"])

    assert input_path.with_suffix(".txt").read_text(encoding="utf-8") == (
        "A: 你好\nB: 在的\n"
    )
    assert calls == [
        ("extract", (input_path.resolve(), "source.mp3")),
        (
            "transcribe",
            ("source.mp3", cli.DEFAULT_VOLC_RESOURCE_ID, cli.DEFAULT_VOLC_MODEL, "zh"),
        ),
    ]
