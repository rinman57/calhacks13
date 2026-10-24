"""Save the transcript a later checker reads for one YouTube video.

Given a public watch URL, this module saves:

- metadata.json: title, channel, duration, and which caption tracks exist
- transcript.json / transcript.txt: timestamped captions, or a Cartesia transcript of the audio when those captions are missing
- audio.*: the audio file, only when captions are missing and speech-to-text is needed
- claims.json: a first-pass list of claims and opinions from that transcript

It does not download the picture and it does not search for evidence.
"""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo

import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
from yt_dlp.utils import DownloadError

VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
PACIFIC = ZoneInfo("America/Los_Angeles")
TIMESTAMP_RE = re.compile(
    r"(?P<start>(?:\d{2}:)?\d{2}:\d{2}\.\d{3})\s+-->\s+(?P<end>(?:\d{2}:)?\d{2}:\d{2}\.\d{3})"
)

# Audio is saved only when captions are missing, so Cartesia can transcribe it.
AUDIO_FORMAT = "bestaudio[ext=m4a]/bestaudio/best"


@dataclass
class ExtractionResult:
    video_id: str
    output_dir: Path
    metadata_path: Path | None = None
    transcript_path: Path | None = None
    transcript_text_path: Path | None = None
    audio_path: Path | None = None
    video_path: Path | None = None
    claims_path: Path | None = None
    evidence_path: Path | None = None
    verdict_path: Path | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return any(
            path is not None
            for path in (self.transcript_path, self.audio_path, self.video_path)
        )


def video_id_from_folder(name: str) -> str:
    """Return the video id from a folder named either `<id>` or `<id>_<Pacific timestamp>`."""
    if VIDEO_ID_RE.fullmatch(name):
        return name
    prefix = name[:11]
    if len(name) > 11 and name[11] == "_" and VIDEO_ID_RE.fullmatch(prefix):
        return prefix
    return name


def _new_output_dir(output_root: Path, video_id: str) -> Path:
    """Create data/<video_id>_<YYYY-MM-DD>_<HH-MM-SS>_<PDT or PST>."""
    moment = datetime.now(PACIFIC)
    stamp = moment.strftime("%Y-%m-%d_%H-%M-%S")
    zone = moment.tzname() or "PST"
    folder = output_root / f"{video_id}_{stamp}_{zone}"
    suffix = 2
    while folder.exists():
        folder = output_root / f"{video_id}_{stamp}_{zone}_{suffix}"
        suffix += 1
    folder.mkdir(parents=True)
    return folder


def parse_video_id(value: str) -> str:
    """Accept a video id or a youtube.com / youtu.be URL."""
    raw = value.strip()
    if VIDEO_ID_RE.fullmatch(raw):
        return raw

    parsed = urlparse(raw)
    host = (parsed.hostname or "").removeprefix("www.").lower()
    video_id = ""
    if host == "youtu.be":
        video_id = parsed.path.strip("/").split("/")[0]
    elif host in {"youtube.com", "m.youtube.com", "music.youtube.com"}:
        parts = [part for part in parsed.path.split("/") if part]
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [""])[0]
        elif parts and parts[0] in {"shorts", "embed", "live", "v"} and len(parts) > 1:
            video_id = parts[1]

    if not VIDEO_ID_RE.fullmatch(video_id):
        raise ValueError(f"Could not parse a YouTube video id from: {value}")
    return video_id


def extract_youtube(
    url_or_id: str,
    output_root: Path,
    *,
    languages: list[str] | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> ExtractionResult:
    """Save metadata and a timestamped transcript for one video."""
    requested = languages or ["en"]
    video_id = parse_video_id(url_or_id)
    watch_url = f"https://www.youtube.com/watch?v={video_id}"
    output_dir = _new_output_dir(output_root, video_id)
    result = ExtractionResult(video_id=video_id, output_dir=output_dir)

    def progress(stage: str) -> None:
        if on_progress is not None:
            on_progress(stage)

    if shutil.which("node") is None:
        result.warnings.append(
            "node was not found. YouTube may refuse the caption or audio download until a JS runtime is installed."
        )

    progress("metadata")
    info = _fetch_metadata(watch_url)
    listed = sorted(set((info.get("subtitles") or {}) | (info.get("automatic_captions") or {})))
    languages = _expand_languages(requested, listed)
    progress("transcript")
    transcript_payload, transcript_error = _fetch_transcript(video_id, languages, output_dir, watch_url)
    if transcript_error:
        result.warnings.append(transcript_error)

    metadata = _metadata_payload(info, video_id, watch_url, transcript_payload, transcript_error)
    result.metadata_path = output_dir / "metadata.json"
    _write_json(result.metadata_path, metadata)

    speech_error = None
    if transcript_payload is None:
        progress("audio")
        try:
            result.audio_path = _download(watch_url, output_dir, "audio", AUDIO_FORMAT)
        except DownloadError as exc:
            result.warnings.append(f"Audio download failed: {_short_error(exc)}")

    if transcript_payload is None and result.audio_path is not None:
        from extract.transcribe import TranscriptionError, transcript_from_audio

        progress("speech")
        try:
            transcript_payload = transcript_from_audio(
                result.audio_path,
                video_id=video_id,
                language=requested[0],
            )
        except TranscriptionError as exc:
            speech_error = str(exc)
            result.warnings.append(speech_error)

    if transcript_payload is not None:
        result.transcript_path = output_dir / "transcript.json"
        result.transcript_text_path = output_dir / "transcript.txt"
        _write_json(result.transcript_path, transcript_payload)
        result.transcript_text_path.write_text(
            _render_transcript(transcript_payload["segments"]),
            encoding="utf-8",
        )
        metadata["transcript_fetched"] = True
        metadata["transcript_source"] = transcript_payload["source"]
        metadata["transcript_language_code"] = transcript_payload["language_code"]
        metadata["transcript_error"] = None
        if transcript_payload.get("model"):
            metadata["transcript_model"] = transcript_payload["model"]
    elif speech_error:
        metadata["transcript_error"] = " ".join(
            part for part in (transcript_error, speech_error) if part
        )
    _write_json(result.metadata_path, metadata)

    if result.transcript_path is not None:
        from extract.claims import ClaimError, extract_claims

        progress("claims")
        try:
            result.claims_path = extract_claims(output_dir)
        except (ClaimError, OSError, json.JSONDecodeError) as exc:
            result.warnings.append(f"Claim check failed: {exc}")
    else:
        result.warnings.append("Claim check skipped because there is no transcript.")

    if result.claims_path is not None:
        from extract.verify import VerifyConfigError, verify_folder

        progress("research")
        try:
            verified = verify_folder(output_dir, on_progress=progress)
            result.evidence_path = verified.evidence_path
            result.verdict_path = verified.verdict_path
            result.warnings.extend(verified.warnings)
        except VerifyConfigError as exc:
            result.warnings.append(f"Research skipped: {exc}")
        except Exception as exc:
            result.warnings.append(f"Research failed: {_short_error(exc)}")

    progress("done")

    _write_json(
        output_dir / "extraction_report.json",
        {
            "video_id": video_id,
            "watch_url": watch_url,
            "metadata": _relative(result.metadata_path, output_dir),
            "transcript": _relative(result.transcript_path, output_dir),
            "transcript_text": _relative(result.transcript_text_path, output_dir),
            "audio": _relative(result.audio_path, output_dir),
            "video": _relative(result.video_path, output_dir),
            "claims": _relative(result.claims_path, output_dir),
            "evidence": _relative(result.evidence_path, output_dir),
            "verdict": _relative(result.verdict_path, output_dir),
            "warnings": result.warnings,
        },
    )
    return result


class _SilentLogger:
    """Keep yt-dlp's own error lines out of the transcript fallback."""

    def debug(self, message: str) -> None:
        return None

    def info(self, message: str) -> None:
        return None

    def warning(self, message: str) -> None:
        return None

    def error(self, message: str) -> None:
        return None


def _ydl_options(**extra: object) -> dict:
    # deno is yt-dlp's default runtime and is often absent. node is enough.
    options: dict = {
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "logger": _SilentLogger(),
        "js_runtimes": {"node": {}},
    }
    options.update(extra)
    return options


def _fetch_metadata(watch_url: str) -> dict:
    with yt_dlp.YoutubeDL(_ydl_options(skip_download=True)) as downloader:
        info = downloader.extract_info(watch_url, download=False)
    if info is None:
        raise RuntimeError(f"yt-dlp returned no metadata for {watch_url}")
    return info


def _metadata_payload(
    info: dict,
    video_id: str,
    watch_url: str,
    transcript_payload: dict | None,
    transcript_error: str | None,
) -> dict:
    manual = sorted((info.get("subtitles") or {}).keys())
    automatic = sorted((info.get("automatic_captions") or {}).keys())
    description = info.get("description") or ""
    return {
        "video_id": video_id,
        "watch_url": watch_url,
        "title": info.get("title"),
        "channel": info.get("channel") or info.get("uploader"),
        "duration_seconds": info.get("duration"),
        "upload_date": info.get("upload_date"),
        "language": info.get("language"),
        "availability": info.get("availability"),
        "description": description,
        "caption_languages": {
            "manual": manual,
            "automatic": automatic,
        },
        "transcript_fetched": transcript_payload is not None,
        "transcript_source": None if transcript_payload is None else transcript_payload["source"],
        "transcript_language_code": None
        if transcript_payload is None
        else transcript_payload["language_code"],
        "transcript_error": transcript_error,
    }


def _expand_languages(languages: list[str], available: list[str]) -> list[str]:
    """Keep the requested order, and include a regional track such as en-US for en."""
    ordered: list[str] = []
    pool = [code for code in available if isinstance(code, str) and code]
    for requested in languages:
        wanted = requested.lower()
        matches = [code for code in pool if code.lower() == wanted or code.lower().startswith(wanted + "-")]
        matches.sort(key=lambda code: (code.lower() != wanted, code.lower()))
        for code in matches:
            if code not in ordered:
                ordered.append(code)
        if not matches and requested not in ordered:
            ordered.append(requested)
            if "-" not in requested:
                for region in (f"{requested}-US", f"{requested}-GB"):
                    if region not in ordered:
                        ordered.append(region)
    return ordered or list(languages)


def _fetch_transcript(
    video_id: str,
    languages: list[str],
    output_dir: Path,
    watch_url: str,
) -> tuple[dict | None, str | None]:
    """Prefer the transcript API, then caption files from yt-dlp."""
    payload, api_error = _fetch_transcript_api(video_id, languages)
    if payload is not None:
        return payload, None

    payload, caption_error = _fetch_transcript_captions(languages, output_dir, watch_url)
    if payload is not None:
        return payload, None

    parts = [part for part in (api_error, caption_error) if part]
    if not parts:
        return None, "No caption track matched the requested languages."
    return None, " ".join(parts)


def _fetch_transcript_api(video_id: str, languages: list[str]) -> tuple[dict | None, str | None]:
    """Return a timestamped transcript, preferring a manual track over auto captions."""
    try:
        api = YouTubeTranscriptApi()
        listed = api.list(video_id)
        chosen = listed.find_transcript(languages)
        fetched = chosen.fetch()
    except Exception as exc:
        return None, f"Transcript API failed ({_short_error(exc)})."

    segments = []
    for snippet in fetched.to_raw_data():
        start = float(snippet["start"])
        duration = float(snippet["duration"])
        text = _clean_caption_text(str(snippet.get("text", "")))
        if not text:
            continue
        segments.append(_segment(start, duration, text))

    return _transcript_payload(
        video_id,
        language=fetched.language,
        language_code=fetched.language_code,
        source="automatic" if fetched.is_generated else "manual",
        segments=segments,
    ), None


def _fetch_transcript_captions(
    languages: list[str],
    output_dir: Path,
    watch_url: str,
) -> tuple[dict | None, str | None]:
    sub_dir = output_dir / "_captions"
    if sub_dir.exists():
        shutil.rmtree(sub_dir)
    sub_dir.mkdir()
    options = _ydl_options(
        skip_download=True,
        writesubtitles=True,
        writeautomaticsub=True,
        subtitleslangs=languages,
        subtitlesformat="json3/vtt",
        outtmpl=str(sub_dir / "%(id)s.%(ext)s"),
    )
    try:
        with yt_dlp.YoutubeDL(options) as downloader:
            downloader.download([watch_url])
    except DownloadError as exc:
        shutil.rmtree(sub_dir, ignore_errors=True)
        return None, f"Caption download failed ({_short_error(exc)})."

    caption_file = _pick_caption_file(sub_dir, languages)
    if caption_file is None:
        shutil.rmtree(sub_dir, ignore_errors=True)
        return None, "Caption download finished without a subtitle file."

    raw_text = caption_file.read_text(encoding="utf-8")
    kept = output_dir / f"captions{caption_file.suffix}"
    kept.write_text(raw_text, encoding="utf-8")
    shutil.rmtree(sub_dir, ignore_errors=True)

    if caption_file.name.endswith(".json3"):
        segments = _parse_json3(raw_text)
        language_code = _language_from_filename(caption_file.name)
    else:
        segments = _parse_vtt(raw_text)
        language_code = _language_from_filename(caption_file.name)

    if not segments:
        return None, f"Caption file {caption_file.name} did not contain any text."

    return _transcript_payload(
        parse_video_id(watch_url),
        language=language_code,
        language_code=language_code,
        source="captions",
        segments=segments,
    ), None


def _download(watch_url: str, output_dir: Path, stem: str, fmt: str) -> Path:
    for existing in output_dir.glob(f"{stem}.*"):
        existing.unlink()

    options = _ydl_options(
        format=fmt,
        outtmpl=str(output_dir / f"{stem}.%(ext)s"),
        restrictfilenames=True,
        overwrites=True,
    )
    with yt_dlp.YoutubeDL(options) as downloader:
        info = downloader.extract_info(watch_url, download=True)
        if info is None:
            raise RuntimeError(f"yt-dlp returned no file info for {watch_url}")
        saved = Path(downloader.prepare_filename(info))

    if not saved.exists():
        matches = sorted(output_dir.glob(f"{stem}.*"))
        if not matches:
            raise RuntimeError(f"Download reported success but {stem} was not written")
        saved = matches[0]
    return saved


def _transcript_payload(
    video_id: str,
    *,
    language: str,
    language_code: str,
    source: str,
    segments: list[dict],
) -> dict:
    return {
        "video_id": video_id,
        "language": language,
        "language_code": language_code,
        "source": source,
        "segment_count": len(segments),
        "segments": segments,
    }


def _segment(start: float, duration: float, text: str) -> dict:
    return {
        "start_seconds": round(start, 3),
        "duration_seconds": round(duration, 3),
        "end_seconds": round(start + duration, 3),
        "text": text,
    }


def _parse_json3(raw_text: str) -> list[dict]:
    payload = json.loads(raw_text)
    segments = []
    for event in payload.get("events") or []:
        pieces = event.get("segs") or []
        text = _clean_caption_text("".join(str(piece.get("utf8", "")) for piece in pieces))
        if not text:
            continue
        start = float(event.get("tStartMs") or 0) / 1000
        duration = float(event.get("dDurationMs") or 0) / 1000
        segments.append(_segment(start, duration, text))
    return segments


def _parse_vtt(raw_text: str) -> list[dict]:
    segments = []
    lines = raw_text.splitlines()
    index = 0
    while index < len(lines):
        match = TIMESTAMP_RE.search(lines[index])
        if not match:
            index += 1
            continue
        start = _timestamp_to_seconds(match.group("start"))
        end = _timestamp_to_seconds(match.group("end"))
        index += 1
        text_lines = []
        while index < len(lines) and lines[index].strip():
            text_lines.append(lines[index])
            index += 1
        text = _clean_caption_text(" ".join(text_lines))
        if text:
            segments.append(_segment(start, max(0.0, end - start), text))
    return segments


def _timestamp_to_seconds(value: str) -> float:
    parts = value.split(":")
    if len(parts) == 3:
        hours, minutes, seconds = parts
    else:
        hours = "0"
        minutes, seconds = parts
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _language_from_filename(name: str) -> str:
    pieces = name.split(".")
    if len(pieces) >= 3:
        return pieces[-2]
    return "unknown"


def _pick_caption_file(sub_dir: Path, languages: list[str]) -> Path | None:
    files = [path for path in sub_dir.iterdir() if path.suffix in {".json3", ".vtt"}]

    def rank(path: Path) -> tuple:
        name = path.name.lower()
        language_rank = len(languages)
        for index, language in enumerate(languages):
            token = f".{language.lower()}."
            if token in f".{name}":
                language_rank = index
                break
        auto_rank = 1 if ".auto." in name else 0
        ext_rank = 0 if path.suffix == ".json3" else 1
        return (language_rank, auto_rank, ext_rank, name)

    files.sort(key=rank)
    return files[0] if files else None


def _clean_caption_text(text: str) -> str:
    without_tags = re.sub(r"<[^>]+>", "", text)
    return " ".join(without_tags.replace("\n", " ").split())


def _short_error(exc: Exception) -> str:
    text = " ".join(str(exc).split())
    if not text:
        text = type(exc).__name__
    if len(text) > 220:
        text = text[:217] + "..."
    return f"{type(exc).__name__}: {text}"


def _render_transcript(segments: list[dict]) -> str:
    lines = []
    for segment in segments:
        lines.append(f"[{_clock(segment['start_seconds'])}] {segment['text']}")
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


def _relative(path: Path | None, output_dir: Path) -> str | None:
    if path is None:
        return None
    return str(path.relative_to(output_dir))
