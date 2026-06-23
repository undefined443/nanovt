# video-transcriber

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
pip install video-transcriber
```

Or install it as an isolated `uv` tool:

```bash
uv tool install video-transcriber
```

## Usage

```bash
vt input.mp4
```

The longer `video-transcriber` command is also available.

By default, `input.mp4` writes `input.txt`.

To force a language, pass an optional language code. For example, English:

```bash
vt input.mp4 --language en
```

If `--language` is omitted, the model detects the language automatically.

Other useful options:

```bash
vt input.mp4 --chunk-seconds 180 --retries 3
vt input.mp4 --output transcript.txt
vt input.mp4 --keep-temp
```

## Development

```bash
uv sync --group dev
uv run pytest
uv run vt input.mp4
```

## Releasing

Before the first release, confirm:

- The PyPI distribution name `video-transcriber` is available.
- The repository, homepage, and issue tracker URLs.

Build and check the distribution locally:

```bash
uv build
uvx twine check dist/*
```

Upload to PyPI after configuring a PyPI API token:

```bash
uvx twine upload dist/*
```
