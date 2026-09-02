"""Tests for the Volcengine speech-to-text provider."""

from pathlib import Path

from nanovt.providers import volc_asr as volc
from nanovt.providers.volc_asr import _format_diarized, _transcribe_volc


def test_format_diarized_labels_speakers_in_order() -> None:
    """Map raw speaker ids to stable A, B, C dialogue labels."""
    result = {
        "utterances": [
            {"additions": {"speaker": "2"}, "text": " 你好。 "},
            {"additions": {"speaker": "5"}, "text": "在的。"},
            {"speaker": "2", "text": "继续。"},
            {"additions": {"speaker": "5"}, "text": "   "},
        ]
    }

    assert _format_diarized(result) == "A: 你好。\nB: 在的。\nA: 继续。"


def test_transcribe_volc_formats_diarized_result(tmp_path: Path, monkeypatch) -> None:
    """Send one synchronous request and label the returned utterances."""
    audio_path = tmp_path / "source.mp3"
    audio_path.write_bytes(b"audio")
    calls: list[str] = []

    def fake_post_json(url: str, headers: dict[str, str], body: dict):
        calls.append(url)
        assert body["audio"]["data"]
        assert body["request"]["enable_speaker_info"] is True
        return volc._STATUS_SUCCESS, {
            "result": {"utterances": [{"additions": {"speaker": "0"}, "text": "嗨"}]}
        }

    monkeypatch.setattr(volc, "_post_json", fake_post_json)

    text = _transcribe_volc(
        audio_path, "key", "volc.seedasr.auc", "bigmodel", "zh", True, 3
    )

    assert text == "A: 嗨"
    assert calls == [volc._RECOGNIZE_URL]


def test_transcribe_volc_returns_plain_text_without_diarization(
    tmp_path: Path, monkeypatch
) -> None:
    """Return the transcript text when diarization is disabled."""
    audio_path = tmp_path / "source.mp3"
    audio_path.write_bytes(b"audio")

    def fake_post_json(url: str, headers: dict[str, str], body: dict):
        assert url == volc._RECOGNIZE_URL
        assert body["request"]["enable_speaker_info"] is False
        return volc._STATUS_SUCCESS, {"result": {"text": " 一段话。 "}}

    monkeypatch.setattr(volc, "_post_json", fake_post_json)

    text = _transcribe_volc(
        audio_path, "key", "volc.seedasr.auc", "bigmodel", None, False, 3
    )

    assert text == "一段话。"


def test_transcribe_volc_retries_transient_network_errors(
    tmp_path: Path, monkeypatch
) -> None:
    """Retry a URLError once, then succeed on the second attempt."""
    import urllib.error

    audio_path = tmp_path / "source.mp3"
    audio_path.write_bytes(b"audio")
    attempts: list[int] = []

    def fake_post_json(url: str, headers: dict[str, str], body: dict):
        attempts.append(1)
        if len(attempts) == 1:
            raise urllib.error.URLError("connection reset")
        return volc._STATUS_SUCCESS, {"result": {"text": "好"}}

    monkeypatch.setattr(volc, "_post_json", fake_post_json)
    monkeypatch.setattr(volc.time, "sleep", lambda _seconds: None)

    text = _transcribe_volc(
        audio_path, "key", "volc.seedasr.auc", "bigmodel", None, False, 3
    )

    assert text == "好"
    assert len(attempts) == 2
