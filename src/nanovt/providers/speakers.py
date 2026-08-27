"""Shared speaker-label assignment for diarized transcripts."""

from __future__ import annotations

_SPEAKER_LABELS = tuple(chr(code) for code in range(ord("A"), ord("Z") + 1))


def _label_for(speaker: str, labels: dict[str, str]) -> str:
    """Return a stable display label for a diarization speaker id.

    Args:
        speaker: Provider speaker id.
        labels: Mutable mapping from speaker id to display label, reused across
            chunks so the same speaker keeps one label.

    Returns:
        A single-letter label (A, B, C, ...), falling back to the raw speaker id
        once the alphabet is exhausted.
    """
    if speaker not in labels:
        index = len(labels)
        labels[speaker] = (
            _SPEAKER_LABELS[index] if index < len(_SPEAKER_LABELS) else speaker
        )
    return labels[speaker]
