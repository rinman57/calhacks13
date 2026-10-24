"""Transcribe saved audio with Cartesia's batch speech-to-text API.

The key is read from CARTESIA_API_KEY, or from the project .env file.
Word timestamps come back from the ink-whisper model and are grouped into
short timed lines for transcript.json and transcript.txt.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path

CARTESIA_URL = "https://api.cartesia.ai/stt"
CARTESIA_VERSION = "2026-08-14"
MODEL = "ink-whisper"
ROOT = Path(__file__).resolve().parents[1]
PAUSE_SECONDS = 0.7
MAX_WORDS = 14


class TranscriptionError(RuntimeError):
    """The audio could not be transcribed."""


def load_project_env(path: Path | None = None) -> None:
    """Load KEY=VALUE lines from .env without overriding real environment variables."""
    env_path = path or (ROOT / ".env")
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def transcript_from_audio(audio_path: Path, *, video_id: str, language: str = "en") -> dict:
    """Return a timestamped transcript payload for one audio file."""
    load_project_env()
    api_key = os.environ.get("CARTESIA_API_KEY", "").strip()
    if not api_key:
        raise TranscriptionError("CARTESIA_API_KEY is not set.")

    language_code = _language_code(language)
    body, boundary = _multipart(
        {
            "model": MODEL,
            "language": language_code,
            "timestamp_granularities[]": "word",
        },
        audio_path,
    )
    request = urllib.request.Request(
        CARTESIA_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Cartesia-Version": CARTESIA_VERSION,
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        message = _api_message(detail) or f"HTTP {exc.code}"
        raise TranscriptionError(f"Cartesia speech-to-text failed ({message}).") from exc
    except urllib.error.URLError as exc:
        raise TranscriptionError(f"Cartesia speech-to-text failed ({exc.reason}).") from exc

    if not isinstance(payload, dict):
        raise TranscriptionError("Cartesia returned an unexpected response.")

    words = _words(payload)
    text = str(payload.get("text") or "").strip()
    if not words and not text:
        raise TranscriptionError("Cartesia returned an empty transcript.")

    segments = _segments_from_words(words)
    if not segments and text:
        segments = [_segment(0.0, 0.0, text)]

    return {
        "video_id": video_id,
        "language": str(payload.get("language") or language_code),
        "language_code": language_code,
        "source": "cartesia",
        "model": MODEL,
        "segment_count": len(segments),
        "segments": segments,
        "words": words,
    }


def save_transcript(output_dir: Path, *, language: str = "en") -> Path:
    """Transcribe audio.* in a saved video folder and write the transcript files."""
    audio_path = _audio_file(output_dir)
    payload = transcript_from_audio(audio_path, video_id=output_dir.name, language=language)
    transcript_path = output_dir / "transcript.json"
    text_path = output_dir / "transcript.txt"
    _write_json(transcript_path, payload)
    text_path.write_text(_render_transcript(payload["segments"]), encoding="utf-8")
    _patch_saved_metadata(output_dir, payload)
    return transcript_path


def _audio_file(output_dir: Path) -> Path:
    matches = sorted(output_dir.glob("audio.*"))
    if not matches:
        raise TranscriptionError(f"No audio file in {output_dir}.")
    return matches[0]


def _patch_saved_metadata(output_dir: Path, payload: dict) -> None:
    metadata_path = output_dir / "metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["transcript_fetched"] = True
        metadata["transcript_source"] = payload["source"]
        metadata["transcript_language_code"] = payload["language_code"]
        metadata["transcript_model"] = payload["model"]
        metadata["transcript_error"] = None
        _write_json(metadata_path, metadata)

    report_path = output_dir / "extraction_report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["transcript"] = "transcript.json"
        report["transcript_text"] = "transcript.txt"
        report["transcript_source"] = payload["source"]
        report["transcript_model"] = payload["model"]
        _write_json(report_path, report)


def _language_code(language: str) -> str:
    code = language.strip().lower().replace("_", "-")
    primary = code.split("-", 1)[0]
    return primary or "en"


def _multipart(fields: dict[str, str], audio_path: Path) -> tuple[bytes, str]:
    boundary = f"----calhacks{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.append(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                f"{value}\r\n"
            ).encode()
        )
    filename = audio_path.name.replace('"', "")
    content_type = "audio/mp4" if audio_path.suffix.lower() in {".m4a", ".mp4"} else "application/octet-stream"
    chunks.append(
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode()
    )
    chunks.append(audio_path.read_bytes())
    chunks.append(f"\r\n--{boundary}--\r\n".encode())
    return b"".join(chunks), boundary


def _words(payload: dict) -> list[dict]:
    words = []
    for item in payload.get("words") or []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("word") or "").strip()
        if not text:
            continue
        start = float(item.get("start") or 0)
        end = float(item.get("end") or start)
        words.append(
            {
                "text": text,
                "start_seconds": round(start, 3),
                "end_seconds": round(max(start, end), 3),
            }
        )
    return words


def _segments_from_words(words: list[dict]) -> list[dict]:
    segments: list[dict] = []
    bucket: list[dict] = []
    for word in words:
        if bucket:
            gap = word["start_seconds"] - bucket[-1]["end_seconds"]
            ended = bucket[-1]["text"].endswith((".", "!", "?"))
            if gap >= PAUSE_SECONDS or ended or len(bucket) >= MAX_WORDS:
                segments.append(_segment_from_bucket(bucket))
                bucket = []
        bucket.append(word)
    if bucket:
        segments.append(_segment_from_bucket(bucket))
    return segments


def _segment_from_bucket(words: list[dict]) -> dict:
    start = float(words[0]["start_seconds"])
    end = float(words[-1]["end_seconds"])
    text = " ".join(str(word["text"]) for word in words)
    return _segment(start, max(0.0, end - start), text)


def _segment(start: float, duration: float, text: str) -> dict:
    return {
        "start_seconds": round(start, 3),
        "duration_seconds": round(duration, 3),
        "end_seconds": round(start + duration, 3),
        "text": text,
    }


def _render_transcript(segments: list[dict]) -> str:
    lines = [f"[{_clock(segment['start_seconds'])}] {segment['text']}" for segment in segments]
    return "\n".join(lines) + ("\n" if lines else "")


def _clock(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    return f"{minutes:02d}:{secs:02d}.{millis:03d}"


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _api_message(detail: str) -> str:
    detail = " ".join(detail.split())
    if not detail:
        return ""
    try:
        parsed = json.loads(detail)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        for key in ("message", "error", "detail"):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                detail = value.strip()
                break
    if len(detail) > 220:
        detail = detail[:217] + "..."
    return detail


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("Usage: python -m extract.transcribe data/<video_id>", file=sys.stderr)
        return 2
    output_dir = Path(args[0])
    try:
        path = save_transcript(output_dir)
    except (TranscriptionError, OSError, json.JSONDecodeError) as exc:
        print(f"Transcription failed: {exc}", file=sys.stderr)
        return 1
    print(f"transcript: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
