"""Preliminary claim list for one saved transcript.

This is step 2 of the pipeline. A fast model copies checkable quotes out of the
timestamped transcript and separates them from opinions. It does not search the
web, watch the video, or decide whether a quote is true. The researcher and
assessor do that later.

The model is Claude Haiku 5.5. Structured output forces the claims.json shape.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from extract.transcribe import load_project_env

MODEL = "claude-haiku-5-5"
EFFORT = "low"
KINDS = ("claim", "opinion")

# The model returns only this thin object. Python fills the remaining claims.json
# fields so a later stage can judge the quote without a second wait here.
EXTRACT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "summary": {"type": "string"},
        "statements": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "kind": {"type": "string", "enum": list(KINDS)},
                    "text": {"type": "string"},
                    "start_seconds": {"type": "number"},
                    "end_seconds": {"type": "number"},
                },
                "required": ["kind", "text", "start_seconds", "end_seconds"],
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
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise ClaimError("ANTHROPIC_API_KEY is not set. Add it to .env.")

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
    return f"""You extract quotes from one transcript. Work quickly. Do not search the web and do not decide whether any statement is true.

Title: {title}
Channel: {channel}

Write one sentence summarizing the transcript.

Then list the statements worth separating:
- kind "claim" means the quote itself asserts a checkable fact about the world.
- kind "opinion" means a judgment, preference, joke, or the speaker's own experience or memory. A personal remark stays an opinion even when it mentions a number, date, or institution.
- Skip lyrics, greetings, and host banter unless they contain a checkable fact.
- Copy text character for character from the transcript. Do not paraphrase, summarize, or correct it.
- Use that quote's own start and end times.
- Do not turn a quote into a different claim the speaker did not make.

TRANSCRIPT
{transcript}
"""


def _generate(api_key: str, prompt: str) -> dict:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key, max_retries=1, timeout=45.0)
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=2048,
            system="Return only the requested JSON. Copy transcript quotes exactly.",
            messages=[{"role": "user", "content": prompt}],
            output_config={
                "effort": EFFORT,
                "format": {"type": "json_schema", "schema": EXTRACT_SCHEMA},
            },
        )
    except Exception as exc:
        message = str(exc).replace(api_key, "[redacted]")
        raise ClaimError(f"Claim extraction failed ({message[:500]}).") from exc
    texts = []
    for block in getattr(response, "content", []) or []:
        text = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
        if isinstance(text, str) and text.strip():
            texts.append(text)
    if not texts:
        raise ClaimError("Claim extraction returned no text.")
    try:
        parsed = json.loads("\n".join(texts))
    except json.JSONDecodeError as exc:
        raise ClaimError("Claim extraction did not return JSON.") from exc
    if not isinstance(parsed, dict):
        raise ClaimError("Claim extraction returned JSON that is not an object.")
    return parsed


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
        start = _seconds(item.get("start_seconds"))
        end = max(start, _seconds(item.get("end_seconds")))
        text, start, end = quote_from_transcript(segments or [], start, end, proposed)
        if not text:
            continue
        worth = kind == "claim"
        statements.append(
            {
                "kind": kind,
                "text": text,
                "start_seconds": start,
                "end_seconds": end,
                "visual_note": "",
                "worth_checking": worth,
                "preliminary_label": "unverified" if worth else "opinion",
                "label_confidence": None,
                "why_this_label": "",
                "research_question": "",
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
            "The extractor only separates quotes. It does not decide whether a claim is true. "
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


def _clock(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    minutes, remainder = divmod(total_ms, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{minutes:02d}:{secs:02d}.{millis:03d}"


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
