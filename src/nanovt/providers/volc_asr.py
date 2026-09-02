"""Transcribe audio with Volcengine (Doubao/SeedASR) big-model speech-to-text.

Volcengine exposes a synchronous whole-file API: one request returns the full
transcript. The audio is not split into chunks; the compressed stream is sent
inline as base64.
"""

from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from nanovt.providers.speakers import _label_for

_RECOGNIZE_URL = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/recognize/flash"
_STATUS_SUCCESS = "20000000"


def _post_json(url: str, headers: dict[str, str], body: dict) -> tuple[str, dict]:
    """POST a JSON body and return the API status code and parsed response.

    Args:
        url: Endpoint URL.
        headers: Request headers.
        body: JSON-serializable request body.

    Returns:
        The ``X-Api-Status-Code`` response header and the parsed JSON body.
    """
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=600) as response:
        payload = response.read().decode("utf-8") or "{}"
        return response.headers.get("X-Api-Status-Code", ""), json.loads(payload)


def _headers(api_key: str, resource_id: str, request_id: str) -> dict[str, str]:
    """Build the Volcengine ASR request headers."""
    return {
        "Content-Type": "application/json",
        "X-Api-Key": api_key,
        "X-Api-Resource-Id": resource_id,
        "X-Api-Request-Id": request_id,
        "X-Api-Sequence": "-1",
    }


def _recognize(
    headers: dict[str, str],
    audio_b64: str,
    model: str,
    language: str | None,
    diarize: bool,
) -> dict:
    """Run one synchronous recognition request.

    Args:
        headers: Request headers.
        audio_b64: Base64-encoded compressed audio.
        model: Model name, such as ``bigmodel``.
        language: Optional input language code.
        diarize: Whether to request speaker diarization.

    Returns:
        The ``result`` object from the response body.

    Raises:
        RuntimeError: If the request is rejected.
    """
    request_body = {
        "model_name": model,
        "enable_itn": True,
        "enable_punc": True,
        "enable_ddc": True,
        "enable_speaker_info": diarize,
        "show_utterances": diarize,
    }
    if language:
        request_body["language"] = language

    status, payload = _post_json(
        _RECOGNIZE_URL,
        headers,
        {
            "user": {"uid": "nanovt"},
            "audio": {"data": audio_b64, "format": "mp3"},
            "request": request_body,
        },
    )
    if status != _STATUS_SUCCESS:
        raise RuntimeError(f"Volcengine recognize failed: {status} {payload}")
    return payload.get("result", {})


def _format_diarized(result: dict) -> str:
    """Format diarized utterances as speaker-labeled dialogue lines.

    Args:
        result: The API ``result`` object holding an ``utterances`` list.

    Returns:
        One ``"<label>: <text>"`` line per utterance, sharing stable labels.
    """
    speaker_labels: dict[str, str] = {}
    lines: list[str] = []
    for utterance in result.get("utterances", []):
        speaker = utterance.get("speaker") or utterance.get("additions", {}).get(
            "speaker", ""
        )
        text = utterance.get("text", "").strip()
        if text:
            lines.append(f"{_label_for(speaker, speaker_labels)}: {text}")
    return "\n".join(lines)


def _transcribe_volc(
    audio_path: Path,
    api_key: str,
    resource_id: str,
    model: str,
    language: str | None,
    diarize: bool,
    retries: int,
) -> str:
    """Transcribe an audio file with Volcengine big-model ASR.

    Args:
        audio_path: Compressed audio file (mono 16 kHz MP3).
        api_key: Volcengine API key.
        resource_id: Volcengine ASR resource id.
        model: Model name, such as ``bigmodel``.
        language: Optional input language code.
        diarize: Whether to request speaker diarization.
        retries: Number of retries after a failed request.

    Returns:
        The transcript text, speaker-labeled when diarization is requested.
    """
    audio_b64 = base64.b64encode(audio_path.read_bytes()).decode("ascii")
    headers = _headers(api_key, resource_id, str(uuid.uuid4()))

    print(f"Transcribing audio with Volcengine ({model})")
    for attempt in range(1, retries + 2):
        try:
            result = _recognize(headers, audio_b64, model, language, diarize)
            if diarize:
                return _format_diarized(result)
            return result.get("text", "").strip()
        except urllib.error.URLError as exc:
            if attempt > retries:
                raise RuntimeError(str(exc)) from exc
            wait_seconds = min(30, 2**attempt)
            print(f"  {exc}; retrying in {wait_seconds}s")
            time.sleep(wait_seconds)

    raise RuntimeError("Transcription failed after retries.")
