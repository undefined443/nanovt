# video-transcriber

Transcribe video or audio files with OpenAI speech-to-text.

The script extracts audio with `ffmpeg`, converts it to mono 16 kHz WAV,
splits it into short chunks, transcribes each chunk with OpenAI, and writes a
single text file.

## Requirements

- Python 3.13+
- `uv`
- `ffmpeg`
- `OPENAI_API_KEY` in your environment or `~/.zshenv`

## Usage

```bash
uv run python transcribe_video.py input.mp4
```

By default, `input.mp4` writes `input.txt`.

To force a language, pass an optional language code. For example, English:

```bash
uv run python transcribe_video.py input.mp4 --language en
```

If `--language` is omitted, the model detects the language automatically.

Other useful options:

```bash
uv run python transcribe_video.py input.mp4 --chunk-seconds 180 --retries 3
uv run python transcribe_video.py input.mp4 --output transcript.txt
uv run python transcribe_video.py input.mp4 --keep-temp
```
