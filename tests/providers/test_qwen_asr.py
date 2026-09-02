"""Tests for the qwen3-asr-flash speech-to-text provider."""

from collections.abc import Mapping
from pathlib import Path
from typing import cast

from nanovt.providers.qwen_asr import _QwenClient, _transcribe_chunk


class _Message:
    def __init__(self, content: str | None) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str | None) -> None:
        self.message = _Message(content)


class _Completion:
    def __init__(self, content: str | None) -> None:
        self.choices = [_Choice(content)]


class _Completions:
    def __init__(self, content: str | None) -> None:
        self._content = content
        self.model: str | None = None
        self.messages: list[dict[str, object]] | None = None
        self.extra_body: Mapping[str, object] | None = None

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, object]],
        extra_body: Mapping[str, object] | None,
    ) -> _Completion:
        self.model = model
        self.messages = messages
        self.extra_body = extra_body
        return _Completion(self._content)


class _Chat:
    def __init__(self, completions: _Completions) -> None:
        self.completions = completions


class _Client:
    def __init__(self, completions: _Completions) -> None:
        self.chat = _Chat(completions)


def test_transcribe_chunk_returns_stripped_message_content(tmp_path: Path) -> None:
    """Return the assistant message content with surrounding whitespace removed."""
    chunk_path = tmp_path / "chunk.wav"
    chunk_path.write_bytes(b"audio")
    completions = _Completions("  你好世界  ")
    client = cast(_QwenClient, _Client(completions))

    text = _transcribe_chunk(chunk_path, client, "qwen3-asr-flash", None, 0)

    assert text == "你好世界"
    assert completions.model == "qwen3-asr-flash"
    assert completions.extra_body is None
    assert completions.messages is not None
    content = completions.messages[0]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "input_audio"
    assert content[0]["input_audio"]["data"].startswith("data:audio/wav;base64,")


def test_transcribe_chunk_forwards_language_as_asr_option(tmp_path: Path) -> None:
    """Pass an explicit language through the DashScope asr_options body."""
    chunk_path = tmp_path / "chunk.wav"
    chunk_path.write_bytes(b"audio")
    completions = _Completions("hi")
    client = cast(_QwenClient, _Client(completions))

    _transcribe_chunk(chunk_path, client, "qwen3-asr-flash", "zh", 0)

    assert completions.extra_body == {"asr_options": {"language": "zh"}}


def test_transcribe_chunk_handles_empty_content(tmp_path: Path) -> None:
    """Return an empty string when the model produces no text."""
    chunk_path = tmp_path / "chunk.wav"
    chunk_path.write_bytes(b"audio")
    client = cast(_QwenClient, _Client(_Completions(None)))

    assert _transcribe_chunk(chunk_path, client, "qwen3-asr-flash", None, 0) == ""
