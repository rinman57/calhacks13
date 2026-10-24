"""Preliminary claim list for one saved video.

This is step 2 of the pipeline: summarize the frames and the transcript,
separate claims from opinions, and give each statement a first-pass label
with a confidence score. It does not search the web. Later research agents
read claims.json and check the worth_checking items on their own.

The model is Gemini 3.8 Flash. On the VeriTaS multimodal fact-checking
benchmark it is among the strongest video models, and the Gemini API can
watch an uploaded video while also reading our timestamped transcript.
Structured output forces the claims.json shape.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from extract.transcribe import load_project_env

MODEL = "gemini-3.8-flash"
UPLOAD_URL = "https://generativelanguage.googleapis.com/upload/v1beta/files"
API_ROOT = "https://generativelanguage.googleapis.com/v1beta"
LABELS = ("true", "misinfo", "misleading", "unverified", "opinion", "satire")
KINDS = ("claim", "opinion")

SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": "What the timestamped frames and the timestamped transcript are saying together.",
        },
        "statements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": list(KINDS),
                        "description": "claim is a checkable assertion. opinion is a judgment, joke, or preference.",
                    },
                    "text": {
                        "type": "string",
                        "description": "One atomic statement, in the speaker's meaning, not a whole paragraph.",
                    },
                    "start_seconds": {"type": "number"},
                    "end_seconds": {"type": "number"},
                    "visual_note": {
                        "type": "string",
                        "description": "What the frames show at this time. Empty when the point is only in the audio.",
                    },
                    "worth_checking": {
                        "type": "boolean",
                        "description": "True only when a later agent should research this statement on its own.",
                    },
                    "preliminary_label": {"type": "string", "enum": list(LABELS)},
                    "label_confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "How sure this label is, from the video alone. 0 is a guess, 1 is certain.",
                    },
                    "why_this_label": {"type": "string"},
                    "research_question": {
                        "type": "string",
                        "description": "The question for the later research agent. Empty when worth_checking is false.",
                    },
                },
                "required": [
                    "kind",
                    "text",
                    "start_seconds",
                    "end_seconds",
                    "visual_note",
                    "worth_checking",
                    "preliminary_label",
                    "label_confidence",
                    "why_this_label",
                    "research_question",
                ],
            },
        },
    },
    "required": ["summary", "statements"],
}


class ClaimError(RuntimeError):
    """The claim list could not be built."""


def extract_claims(output_dir: Path) -> Path:
    """Watch one saved video folder and write claims.json."""
    load_project_env()
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ClaimError("GEMINI_API_KEY is not set. Add it to .env.")

    video_path = _one_file(output_dir, "video.*", "video")
    transcript = _transcript_text(output_dir)
    metadata = _read_json(output_dir / "metadata.json")
    video_id = str(metadata.get("video_id") or output_dir.name)

    file_uri = _upload_video(video_path, api_key)
    raw = _generate(api_key, file_uri, _prompt(metadata, transcript))
    payload = _normalize(raw, video_id)
    path = output_dir / "claims.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _prompt(metadata: dict, transcript: str) -> str:
    title = metadata.get("title") or "Unknown title"
    channel = metadata.get("channel") or "Unknown channel"
    return f"""You are the multimodal claim extractor for one video. This is a first pass for later research agents. Do not search the web. Do not treat your own memory as a finished fact-check.

You can see the video frames and you have the timestamped transcript below. Use both. A statement can come from what is said, what is shown, or a mismatch between the two.

Title: {title}
Channel: {channel}

Write a short summary of what the frames and the transcript are saying together.

Then list the statements worth separating:
- A claim is one checkable assertion about the world.
- An opinion is a judgment, preference, prediction, or joke that is not a checkable fact.
- Skip lyrics, greetings, and host banter unless they contain a checkable fact.

preliminary_label is only a first guess from this video:
- true: the frames or the rest of the video make this hard to doubt, such as who is visibly on screen
- misinfo: the video states something the frames or the rest of the video clearly contradict
- misleading: the wording is not flatly false, but it leaves out or twists what the frames show
- unverified: a checkable claim this video does not settle
- opinion: not a factual claim
- satire: a sketch, joke, or lyric, not offered as a report

label_confidence is how sure you are of that label, from 0 to 1. Use a low number when you are guessing. If the video does not settle a factual claim, use unverified instead of true or misinfo.

worth_checking is true only for a claim a later agent should research on its own. Those need a specific research_question. Opinions and satire are usually not worth checking.

Use the transcript times. Do not invent a statement that is not in the video.

TRANSCRIPT
{transcript}
"""


def _upload_video(video_path: Path, api_key: str) -> str:
    size = video_path.stat().st_size
    start = urllib.request.Request(
        UPLOAD_URL,
        data=json.dumps({"file": {"display_name": video_path.stem}}).encode(),
        headers={
            "x-goog-api-key": api_key,
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(size),
            "X-Goog-Upload-Header-Content-Type": "video/mp4",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with _open(start, timeout=60) as response:
        upload_url = response.headers.get("x-goog-upload-url")
    if not upload_url:
        raise ClaimError("Gemini did not return an upload URL.")

    upload = urllib.request.Request(
        upload_url,
        data=video_path.read_bytes(),
        headers={
            "Content-Length": str(size),
            "X-Goog-Upload-Offset": "0",
            "X-Goog-Upload-Command": "upload, finalize",
        },
        method="POST",
    )
    info = _json(upload, timeout=180)
    file_info = info.get("file") if isinstance(info.get("file"), dict) else info
    name = str(file_info.get("name") or "")
    uri = str(file_info.get("uri") or "")
    state = str(file_info.get("state") or "")
    if not name or not uri:
        raise ClaimError("Gemini accepted the upload but returned no file URI.")

    deadline = time.time() + 180
    while state == "PROCESSING":
        if time.time() > deadline:
            raise ClaimError("Gemini did not finish processing the video.")
        time.sleep(2)
        status = _json(
            urllib.request.Request(
                f"{API_ROOT}/{name}",
                headers={"x-goog-api-key": api_key},
                method="GET",
            ),
            timeout=30,
        )
        state = str(status.get("state") or "")
    if state not in {"ACTIVE", ""}:
        raise ClaimError(f"Gemini could not process the video ({state or 'unknown state'}).")
    return uri


def _generate(api_key: str, file_uri: str, prompt: str) -> dict:
    body = {
        "model": MODEL,
        "input": [
            {"type": "video", "uri": file_uri, "mime_type": "video/mp4"},
            {"type": "text", "text": prompt},
        ],
        "response_format": {
            "type": "text",
            "mime_type": "application/json",
            "schema": SCHEMA,
        },
    }
    request = urllib.request.Request(
        f"{API_ROOT}/interactions",
        data=json.dumps(body).encode(),
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        method="POST",
    )
    payload = _json(request, timeout=300)
    text = _output_text(payload)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ClaimError("Gemini did not return JSON.") from exc
    if not isinstance(parsed, dict):
        raise ClaimError("Gemini returned JSON that is not an object.")
    return parsed


def _output_text(payload: dict) -> str:
    direct = payload.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct
    for step in reversed(payload.get("steps") or []):
        if not isinstance(step, dict):
            continue
        content = step.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    return part["text"]
        if isinstance(step.get("text"), str):
            return step["text"]
    output = payload.get("output")
    if isinstance(output, list):
        for part in output:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                return part["text"]
    raise ClaimError("Gemini returned no text.")


def _normalize(raw: dict, video_id: str) -> dict:
    summary = str(raw.get("summary") or "").strip()
    statements = []
    for item in raw.get("statements") or []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        kind = item.get("kind") if item.get("kind") in KINDS else "claim"
        label = item.get("preliminary_label")
        if label not in LABELS:
            label = "opinion" if kind == "opinion" else "unverified"
        start = _seconds(item.get("start_seconds"))
        end = max(start, _seconds(item.get("end_seconds")))
        confidence = _confidence(item.get("label_confidence"))
        worth = bool(item.get("worth_checking")) and kind == "claim" and label not in {"opinion", "satire"}
        question = str(item.get("research_question") or "").strip() if worth else ""
        statements.append(
            {
                "kind": kind,
                "text": text,
                "start_seconds": start,
                "end_seconds": end,
                "visual_note": str(item.get("visual_note") or "").strip(),
                "worth_checking": worth,
                "preliminary_label": label,
                "label_confidence": confidence,
                "why_this_label": str(item.get("why_this_label") or "").strip(),
                "research_question": question,
            }
        )
    statements.sort(key=lambda item: (item["start_seconds"], item["end_seconds"], item["text"]))
    for index, item in enumerate(statements, start=1):
        item["id"] = f"c{index}"

    return {
        "video_id": video_id,
        "model": MODEL,
        "role": "preliminary",
        "note": "Labels are a first read of the video. They are not a finished fact-check.",
        "summary": summary,
        "statement_count": len(statements),
        "checkable_count": sum(1 for item in statements if item["worth_checking"]),
        "statements": statements,
    }


def _transcript_text(output_dir: Path) -> str:
    text_path = output_dir / "transcript.txt"
    if text_path.is_file():
        text = text_path.read_text(encoding="utf-8").strip()
        if text:
            return text
    payload = _read_json(output_dir / "transcript.json")
    lines = []
    for segment in payload.get("segments") or []:
        if not isinstance(segment, dict):
            continue
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        lines.append(f"[{_clock(float(segment.get('start_seconds') or 0))}] {text}")
    if not lines:
        raise ClaimError(f"No transcript in {output_dir}.")
    return "\n".join(lines)


def _one_file(output_dir: Path, pattern: str, label: str) -> Path:
    matches = sorted(output_dir.glob(pattern))
    if not matches:
        raise ClaimError(f"No {label} file in {output_dir}.")
    return matches[0]


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _seconds(value: object) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    if number < 0:
        return 0.0
    return round(number, 3)


def _confidence(value: object) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    if number > 1:
        number = number / 100
    return round(min(1.0, max(0.0, number)), 3)


def _clock(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    minutes, remainder = divmod(total_ms, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{minutes:02d}:{secs:02d}.{millis:03d}"


def _json(request: urllib.request.Request, *, timeout: int) -> dict:
    with _open(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ClaimError("Gemini returned an unexpected response.")
    return payload


def _open(request: urllib.request.Request, *, timeout: int):
    try:
        return urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        message = _api_message(detail) or f"HTTP {exc.code}"
        raise ClaimError(f"Gemini request failed ({message}).") from exc
    except urllib.error.URLError as exc:
        raise ClaimError(f"Gemini request failed ({exc.reason}).") from exc


def _api_message(detail: str) -> str:
    detail = " ".join(detail.split())
    if not detail:
        return ""
    try:
        parsed = json.loads(detail)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        error = parsed.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            detail = error["message"].strip()
        else:
            for key in ("message", "detail"):
                value = parsed.get(key)
                if isinstance(value, str) and value.strip():
                    detail = value.strip()
                    break
    if len(detail) > 240:
        detail = detail[:237] + "..."
    return detail


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("Usage: python -m extract.claims data/<video_id>", file=sys.stderr)
        return 2
    try:
        path = extract_claims(Path(args[0]))
    except (ClaimError, OSError, json.JSONDecodeError) as exc:
        print(f"Claim extraction failed: {exc}", file=sys.stderr)
        return 1
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(f"claims: {path}")
    print(f"statements: {payload['statement_count']}")
    print(f"worth_checking: {payload['checkable_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
