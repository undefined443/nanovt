# nanovt

Transcribe video or audio files with OpenAI speech-to-text.

The script extracts audio with `ffmpeg`, converts it to mono 16 kHz WAV,
splits it into short chunks, transcribes each chunk with OpenAI, and writes a
single text file.

## Requirements

- Python 3.13+
- `ffmpeg`
- `OPENAI_API_KEY` in your environment

## Installation

From PyPI:

```bash
pip install nanovt
```

Or install it as an isolated `uv` tool:

```bash
uv tool install nanovt
```

## Usage

```bash
nanovt input.mp4
```

By default, `input.mp4` writes `input.txt`.

To force a language, pass an optional language code. For example, English:

```bash
nanovt input.mp4 --language en
```

If `--language` is omitted, the model detects the language automatically.

Other useful options:

```bash
nanovt input.mp4 --chunk-seconds 180 --retries 3
nanovt input.mp4 --output transcript.txt
nanovt input.mp4 --keep-temp
```

For speaker-labeled dialogue transcription, enable diarization:

```bash
nanovt input.mp4 --diarize
```

This uses `gpt-4o-transcribe-diarize` by default and writes dialogue lines such
as `A: ...`, `B: ...`, and `C: ...`.

## Development

```bash
uv sync --group dev
uv run pytest
uv run nanovt input.mp4
```
