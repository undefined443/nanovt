"""Transcribe audio with Volcengine (Doubao/SeedASR) big-model speech-to-text.

Volcengine exposes an asynchronous whole-file API: submit a job, then poll for
the result. Unlike the OpenAI path, the audio is not split into chunks; the
compressed stream is sent inline as base64.
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

_SUBMIT_URL = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/submit"
_QUERY_URL = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/query"
_STATUS_SUCCESS = "20000000"
_STATUS_PROCESSING = {"20000001", "20000002"}
_POLL_SECONDS = 15


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
    with urllib.request.urlopen(request, timeout=120) as response:
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


def _submit(
    headers: dict[str, str],
    audio_b64: str,
    model: str,
    language: str | None,
    diarize: bool,
) -> None:
    """Submit a transcription job.

    Raises:
        RuntimeError: If the submit request is rejected.
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
        _SUBMIT_URL,
        headers,
        {
            "user": {"uid": "nanovt"},
            "audio": {"data": audio_b64, "format": "mp3"},
            "request": request_body,
        },
    )
    if status != _STATUS_SUCCESS:
        raise RuntimeError(f"Volcengine submit failed: {status} {payload}")


def _poll(headers: dict[str, str]) -> dict:
    """Poll for the transcription result until the job reaches a terminal state.

    Returns:
        The ``result`` object from the response body.

    Raises:
        RuntimeError: If the job ends in a non-success state.
    """
    while True:
        status, payload = _post_json(_QUERY_URL, headers, {})
        if status == _STATUS_SUCCESS:
            return payload.get("result", {})
        if status in _STATUS_PROCESSING:
            time.sleep(_POLL_SECONDS)
            continue
        raise RuntimeError(f"Volcengine query failed: {status} {payload}")


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
        retries: Number of retries after a failed submit request.

    Returns:
        The transcript text, speaker-labeled when diarization is requested.
    """
    audio_b64 = base64.b64encode(audio_path.read_bytes()).decode("ascii")
    headers = _headers(api_key, resource_id, str(uuid.uuid4()))

    print(f"Submitting audio to Volcengine ({model})")
    for attempt in range(1, retries + 2):
        try:
            _submit(headers, audio_b64, model, language, diarize)
            break
        except urllib.error.URLError as exc:
            if attempt > retries:
                raise RuntimeError(str(exc)) from exc
            wait_seconds = min(30, 2**attempt)
            print(f"  {exc}; retrying in {wait_seconds}s")
            time.sleep(wait_seconds)

    print("Waiting for the transcription result")
    result = _poll(headers)
    if diarize:
        return _format_diarized(result)
    return result.get("text", "").strip()
