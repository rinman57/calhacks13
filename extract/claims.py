"""Preliminary claim list for one saved transcript.

This is step 2 of the pipeline: read the timestamped transcript, separate
claims from opinions, and give each statement a first-pass label with a
confidence score. It does not search the web and it does not watch the video.
Later research agents read claims.json and check the worth_checking items on
their own.

The model is Gemini 3.8 Flash. Structured output forces the claims.json shape.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from extract.transcribe import load_project_env

MODEL = "gemini-3.8-flash"
API_ROOT = "https://generativelanguage.googleapis.com/v1beta"
LABELS = ("true", "misinfo", "misleading", "unverified", "opinion", "satire")
KINDS = ("claim", "opinion")

SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": "What the timestamped transcript is saying.",
        },
        "statements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": list(KINDS),
                        "description": "claim is a checkable assertion the quote itself states. opinion is a judgment, preference, joke, or the speaker's own experience.",
                    },
                    "text": {
                        "type": "string",
                        "description": "An exact contiguous quote copied from the transcript. Do not paraphrase.",
                    },
                    "start_seconds": {"type": "number"},
                    "end_seconds": {"type": "number"},
                    "visual_note": {
                        "type": "string",
                        "description": "Always empty. The extractor reads the transcript and does not watch frames.",
                    },
                    "worth_checking": {
                        "type": "boolean",
                        "description": "True only for an unverified, misinfo, or misleading claim a later agent should research. False for opinions, satire, personal experience, and claims the video itself settles as true.",
                    },
                    "preliminary_label": {
                        "type": "string",
                        "enum": list(LABELS),
                        "description": "true only when the video itself settles the quote. Use unverified when outside knowledge would be required.",
                    },
                    "label_confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "How sure this label is, from the video alone. 0 is a guess, 1 is certain.",
                    },
                    "why_this_label": {"type": "string"},
                    "research_question": {
                        "type": "string",
                        "description": "Asks whether this exact quote is true. Empty when worth_checking is false. Do not ask about a claim the quote does not make.",
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
    """Read one saved transcript and write claims.json."""
    load_project_env()
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ClaimError("GEMINI_API_KEY is not set. Add it to .env.")

    transcript = _transcript_text(output_dir)
    metadata = _read_json(output_dir / "metadata.json")
    video_id = str(metadata.get("video_id") or "")
    if not video_id:
        from extract.youtube import video_id_from_folder

        video_id = video_id_from_folder(output_dir.name)

    raw = _generate(api_key, _prompt(metadata, transcript))
    payload = _normalize(raw, video_id, _transcript_segments(output_dir))
    path = output_dir / "claims.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _prompt(metadata: dict, transcript: str) -> str:
    title = metadata.get("title") or "Unknown title"
    channel = metadata.get("channel") or "Unknown channel"
    return f"""You are the claim extractor for one transcript. This is a first pass for later research agents. Do not search the web. Do not treat your own memory as a finished fact-check.

You have the timestamped transcript below and no video frames. Judge only what is said. Leave visual_note empty.

Title: {title}
Channel: {channel}

Write a short summary of what the transcript is saying.

Then list the statements worth separating:
- A claim is a checkable assertion about the world that the quote itself states.
- An opinion is a judgment, preference, prediction, joke, or the speaker's own experience or memory. A personal remark stays an opinion even when it mentions a number, date, or institution.
- Skip lyrics, greetings, and host banter unless they contain a checkable fact.
- Judge the quote. Do not turn it into a different claim the speaker did not make.

preliminary_label is only a first guess from this transcript. Do not use outside knowledge to choose the label or to decide what the sentence means:
- true: the rest of the transcript settles the quote, such as a name or number the speaker repeats
- misinfo: the transcript states something a later line in the same transcript clearly contradicts
- misleading: the wording is not flatly false, but it leaves out or twists what the rest of the transcript says
- unverified: a checkable claim this transcript does not settle
- opinion: not a factual claim, including personal experience
- satire: a sketch, joke, or lyric, not offered as a report

label_confidence is how sure you are of that label, from 0 to 1. Use a low number when you are guessing. If the transcript does not settle a factual claim, use unverified instead of true or misinfo.

worth_checking is true only for a claim labeled unverified, misinfo, or misleading that a later agent should research. A true label means the video already settled it, so worth_checking is false. Opinions and satire are not worth checking.

research_question asks whether that exact quote is true. It must not ask about a broader claim the speaker did not make. Leave it empty when worth_checking is false.

The text field must be copied character for character from the transcript below. It is the original quote later agents will pass on unchanged. Do not paraphrase, summarize, or correct it. Put interpretation in why_this_label.

Use the transcript times for that exact quote. Do not invent a statement that is not in the transcript. Leave visual_note as an empty string.

TRANSCRIPT
{transcript}
"""


def _generate(api_key: str, prompt: str) -> dict:
    body = {
        "model": MODEL,
        "input": [{"type": "text", "text": prompt}],
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
    payload = _json(request, timeout=120)
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


def _normalize(raw: dict, video_id: str, segments: list | None = None) -> dict:
    summary = str(raw.get("summary") or "").strip()
    statements = []
    for item in raw.get("statements") or []:
        if not isinstance(item, dict):
            continue
        proposed = str(item.get("text") or "").strip()
        if not proposed:
            continue
        kind = item.get("kind") if item.get("kind") in KINDS else "claim"
        label = item.get("preliminary_label")
        if label not in LABELS:
            label = "opinion" if kind == "opinion" else "unverified"
        start = _seconds(item.get("start_seconds"))
        end = max(start, _seconds(item.get("end_seconds")))
        text, start, end = quote_from_transcript(segments or [], start, end, proposed)
        if not text:
            continue
        confidence = _confidence(item.get("label_confidence"))
        worth = (
            bool(item.get("worth_checking"))
            and kind == "claim"
            and label in {"unverified", "misinfo", "misleading"}
        )
        question = str(item.get("research_question") or "").strip() if worth else ""
        statements.append(
            {
                "kind": kind,
                "text": text,
                "start_seconds": start,
                "end_seconds": end,
                "visual_note": "",
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
        "note": (
            "Labels are a first read of the video. They are not a finished fact-check. "
            "Each text field is an exact transcript quote and must be copied unchanged."
        ),
        "summary": summary,
        "statement_count": len(statements),
        "checkable_count": sum(1 for item in statements if item["worth_checking"]),
        "statements": statements,
    }


def quote_from_transcript(
    segments: list,
    start: float,
    end: float,
    proposed: str,
) -> tuple[str, float, float]:
    """Return transcript wording for one statement, plus the times of that quote.

    A proposed string is kept only when it already appears in the transcript.
    Otherwise the quote is the transcript lines that overlap the statement.
    """
    rows = _quote_rows(segments)
    proposed = " ".join(str(proposed or "").split())
    found = _exact_span(rows, proposed)
    if found is not None:
        return found
    chosen = _rows_overlapping(rows, start, end)
    if not chosen:
        alt_start, alt_end = _uncompact_timestamp(start), _uncompact_timestamp(end)
        if (alt_start, alt_end) != (start, end):
            chosen = _rows_overlapping(rows, alt_start, alt_end)
    if not chosen:
        return "", start, end
    return " ".join(text for _, _, text in chosen), chosen[0][0], chosen[-1][1]


def _quote_rows(segments: list) -> list[tuple[float, float, str]]:
    rows = []
    for segment in segments or []:
        if not isinstance(segment, dict):
            continue
        text = " ".join(str(segment.get("text") or "").split())
        if not text:
            continue
        try:
            seg_start = float(segment.get("start_seconds"))
            seg_end = float(segment.get("end_seconds"))
        except (TypeError, ValueError):
            continue
        rows.append((seg_start, max(seg_start, seg_end), text))
    return rows


def _exact_span(
    rows: list[tuple[float, float, str]],
    proposed: str,
) -> tuple[str, float, float] | None:
    if not proposed:
        return None
    for index, (seg_start, seg_end, text) in enumerate(rows):
        if proposed == text or proposed in text:
            return proposed, seg_start, seg_end
        acc = [text]
        for follow_start, follow_end, follow_text in rows[index + 1 :]:
            acc.append(follow_text)
            blob = " ".join(acc)
            if proposed in blob:
                return proposed, seg_start, follow_end
            if len(blob) > len(proposed) + 80:
                break
    return None


def _rows_overlapping(
    rows: list[tuple[float, float, str]],
    start: float,
    end: float,
) -> list[tuple[float, float, str]]:
    return [row for row in rows if row[0] < end and row[1] > start]


def _uncompact_timestamp(value: float) -> float:
    """Turn a misread mm:ss value such as 101.450 back into 61.450 seconds."""
    minutes = int(value // 100)
    rest = value - minutes * 100
    if minutes < 1 or rest >= 60:
        return value
    return minutes * 60 + rest


def _transcript_segments(output_dir: Path) -> list:
    payload = _read_json(output_dir / "transcript.json")
    segments = payload.get("segments") if isinstance(payload, dict) else None
    return segments if isinstance(segments, list) else []


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
