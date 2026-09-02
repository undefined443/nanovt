# nanovt

Transcribe video or audio files with OpenAI, qwen3-asr-flash, or Volcengine speech-to-text.

The script extracts audio with `ffmpeg`, converts it to mono 16 kHz, transcribes it, and writes a single text file. The OpenAI and qwen3-asr-flash providers split the audio into short chunks; the Volcengine provider sends the whole file in one synchronous request.

## Requirements

- Python 3.13+
- `ffmpeg`
- `OPENAI_API_KEY` in your environment (OpenAI provider, the default)
- `DASHSCOPE_API_KEY` in your environment (qwen3-asr-flash provider)
- `VOLC_ASR_API_KEY` in your environment (Volcengine provider)

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

**qwen3-asr-flash provider**

`--provider qwen` runs Alibaba's `qwen3-asr-flash` through its DashScope OpenAI-compatible chat endpoint (one `input_audio` request per chunk), reusing the same WAV chunking path as the default provider. Set `DASHSCOPE_API_KEY`. Keep `--chunk-seconds` at or below 300; the model rejects longer audio. This model has no speaker diarization, so `--diarize` is rejected.

**Volcengine provider**

Volcengine's big-model ASR handles Chinese and code-switched speech well. Set `VOLC_ASR_API_KEY` and select the provider:

```bash
nanovt input.mp4 --provider volc --language zh --diarize
```

The audio is compressed to a mono 16 kHz MP3 and sent whole in one synchronous request, so no public URL is needed. `--model` defaults to `bigmodel`; override the resource id with `VOLC_ASR_RESOURCE_ID` (default `volc.seedasr.auc`). `--chunk-seconds` is unused for this provider. Diarized output uses the same `A: ...` / `B: ...` lines.

## Development

```bash
uv sync --group dev
uv run pytest
uv run nanovt input.mp4
```
