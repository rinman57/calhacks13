"""Preliminary claim list for one saved transcript.

This is step 2 of the pipeline. A fast model copies checkable quotes out of the
timestamped transcript. It keeps only claims worth exploring and leaves the rest out. It does not search the
web, watch the video, or decide whether a quote is true. The researcher and
assessor do that later.

The model is Claude Haiku 5.5. Structured output forces the claims.json shape.
"""

from __future__ import annotations

import json
import os
import re
import sys
import unicodedata
from collections.abc import Callable
from pathlib import Path

from extract.transcribe import load_project_env

MODEL = "claude-haiku-5-5"
EFFORT = "low"
KINDS = ("claim",)
RESEARCH_BASES = (
    "health_safety", "science_environment", "money_law", "public_affairs",
    "product_capability", "other_material_fact",
)

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
                    "research_basis": {"type": "string", "enum": [*RESEARCH_BASES, "skip"]},
                },
                "required": ["kind", "text", "start_seconds", "end_seconds", "research_basis"],
            },
        },
    },
    "required": ["summary", "statements"],
}


class ClaimError(RuntimeError):
    """The claim list could not be built."""


CHUNK_SECONDS = 75.0
OVERLAP_SECONDS = 12.0


def extract_claims(output_dir: Path, on_claim: Callable[[dict], None] | None = None) -> Path:
    """Read one saved transcript and write claims.json.

    Chunks are read in order. Each accepted claim is handed to on_claim as soon
    as it is saved, so research can start before later chunks finish.
    """
    load_project_env()
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise ClaimError("ANTHROPIC_API_KEY is not set. Add it to .env.")

    metadata = _read_json(output_dir / "metadata.json")
    video_id = str(metadata.get("video_id") or "")
    if not video_id:
        from extract.youtube import video_id_from_folder

        video_id = video_id_from_folder(output_dir.name)

    segments = _transcript_segments(output_dir)
    plans = _chunk_plan(segments)
    if not plans:
        raise ClaimError(f"No transcript in {output_dir}.")

    accepted: list[dict] = []
    summaries: list[str] = []
    path = output_dir / "claims.json"
    for index, plan in enumerate(plans):
        last = index == len(plans) - 1
        try:
            raw = _generate(api_key, _prompt(metadata, _render_rows(plan["body"]), _render_rows(plan["context"])))
        except ClaimError:
            if accepted:
                _write_claims(path, video_id, summaries, accepted, complete=False)
            raise
        summaries.append(str(raw.get("summary") or "").strip())
        for item in _statements_from_model(raw, plan["body"]):
            if not _in_primary(float(item["start_seconds"]), plan, last=last):
                continue
            if any(_same_occurrence(item, earlier) for earlier in accepted):
                continue
            item["id"] = f"c{len(accepted) + 1}"
            accepted.append(item)
        _write_claims(path, video_id, summaries, accepted, complete=last)
        if on_claim is not None:
            for item in accepted:
                if item.get("worth_checking") and not item.get("_sent"):
                    item["_sent"] = True
                    on_claim({key: value for key, value in item.items() if key != "_sent"})
    for item in accepted:
        item.pop("_sent", None)
    _write_claims(path, video_id, summaries, accepted, complete=True)
    return path


def _video_blurb(metadata: dict) -> str:
    """First paragraph of the description, without links."""
    raw = re.sub(r"https?://\S+", "", str(metadata.get("description") or ""))
    paragraph: list[str] = []
    for line in raw.splitlines():
        line = " ".join(line.split())
        if not line:
            if paragraph:
                break
            continue
        paragraph.append(line)
        if sum(len(part) for part in paragraph) > 420:
            break
    return " ".join(paragraph)[:480]


def _prompt(metadata: dict, transcript: str, context: str = "") -> str:
    title = metadata.get("title") or "Unknown title"
    channel = metadata.get("channel") or "Unknown channel"
    blurb = _video_blurb(metadata)
    about = f"About: {blurb}\n" if blurb else ""
    context_block = ""
    if context.strip():
        context_block = (
            "CONTEXT\n"
            "These earlier lines are only for understanding. Do not output a statement whose quote begins here.\n"
            f"{context.strip()}\n\n"
        )
    return f"""You extract quotes that would matter if they were false. Work quickly. Do not search the web and do not decide whether any statement is true.

Use the channel, title, and description to judge the kind of video before you keep a line.

Title: {title}
Channel: {channel}
{about}
Write one sentence summarizing this part of the transcript.

Then list only the statements worth separating. Return an empty list when nothing here would matter if it were false.
Research consumes paid web calls. Prefer precision over coverage: a factual-sounding sentence alone is not enough. There is no minimum number of claims to return.
Before including a quote, require ALL THREE:
1. The speaker asserts a complete, specific fact, with enough nearby context to know what it refers to.
2. Independent public evidence could realistically settle it. The video's own footage, private experience, challenge inventory, or scoreboard is not independent public evidence.
3. Being wrong would materially affect a viewer's understanding or decisions about health, safety, science, money, law, products, or public affairs. Trivia and incidental scene-setting do not qualify.
Set research_basis to health_safety, science_environment, money_law, public_affairs, product_capability, or other_material_fact only when these tests pass. other_material_fact must meet the same materiality test; it is not a catch-all for anything factual. Omit rejected lines; if a rejected candidate is returned, set research_basis to skip.
Do not fill a quota, research a fragment by inventing its missing assertion, or collect multiple overlapping quotes for the same assertion within this chunk. Prefer one complete quote with its material qualifiers.
- Skip predictions about this video's outcome, hypothetical examples, promotional promises, rhetorical questions, and broad accusations without a specific checkable assertion.
- Examples to omit: "Produce is the first thing that's gonna go bad" as incidental challenge narration; "This could take a year" as a guess about the stunt; "there's too much money involved with sugar the" as an incomplete accusation.
- Examples to keep: a specific claim about a food's health effects, a product's independently testable capability, or a public official's dated economic statistic. Entertainment context does not disqualify these.
- kind "claim" is a complete quote a viewer could take as a fact about the world outside this video. It is specific enough to check, and it would matter if it were false.
- News, documentaries, explainers, and commentary can contain claims about health, science, money, law, products, public events, or what someone did.
- Entertainment, challenges, comedy, pranks, gameplay, vlogs, and stunts: the story of the video is not a claim. Skip the score, the rules of the bit, how long they have been filming, what they just ate or did, who is ahead, and the prize for this challenge. "The store has 4.3 million calories" and "around 80 calories of blueberries on the floor" are the premise of a stunt, not claims.
- Still keep a claim inside an entertainment video when the line asserts something outside the bit: what a product does, a health or safety fact, where the public's money goes, or a fact about the wider world.
- Leave out judgments, preferences, jokes, and the speaker's own experience or memory, even when they mention a number, date, or institution. Leave out ordinary banter and the plot of the video.
- A fragment is not a claim. A year, date, number, or place with no assertion is not a claim. "back in 2009," is not a claim.
- The speaker timing their own talk is not a claim. "I'm at 44 seconds right now" is not a claim.
- Skip lyrics, greetings, and lines that would not matter if they were false.
- Copy text character for character from the transcript. Do not paraphrase, summarize, or correct it.
- Use that quote's own start and end times.
- Do not turn a quote into a different claim the speaker did not make.

{context_block}TRANSCRIPT
{transcript}
"""


def _generate(api_key: str, prompt: str) -> dict:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key, max_retries=1, timeout=45.0)
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=2048,
            system=(
                "Return only the requested JSON. Copy transcript quotes exactly. "
                "Skip lines that would not matter if they were false."
            ),
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
    statements = _statements_from_model(raw, segments or [])
    statements.sort(key=lambda item: (item["start_seconds"], item["end_seconds"], item["text"]))
    for index, item in enumerate(statements, start=1):
        item["id"] = f"c{index}"
    return _claims_payload(video_id, summary, statements, complete=True)


def _statements_from_model(raw: dict, segments: list) -> list[dict]:
    statements = []
    for item in raw.get("statements") or []:
        if not isinstance(item, dict):
            continue
        proposed = str(item.get("text") or "").strip()
        if not proposed:
            continue
        if item.get("kind") not in KINDS:
            continue
        # Admission is explicit: missing or rejected classifications never launch
        # paid research, even if the returned wording looks like a factual quote.
        if item.get("research_basis") not in RESEARCH_BASES:
            continue
        start = _seconds(item.get("start_seconds"))
        end = max(start, _seconds(item.get("end_seconds")))
        text, start, end = quote_from_transcript(segments, start, end, proposed)
        if not text or not _researchable_quote(text):
            continue
        statements.append(
            {
                "kind": "claim",
                "text": text,
                "start_seconds": start,
                "end_seconds": end,
                "visual_note": "",
                "worth_checking": True,
                "preliminary_label": "unverified",
                "label_confidence": None,
                "why_this_label": "",
                "research_question": "",
                "research_basis": item["research_basis"],
            }
        )
    statements.sort(key=lambda item: (item["start_seconds"], item["end_seconds"], item["text"]))
    return statements


def _claims_payload(video_id: str, summary: str, statements: list[dict], *, complete: bool) -> dict:
    visible = [{key: value for key, value in item.items() if key != "_sent"} for item in statements]
    return {
        "video_id": video_id,
        "model": MODEL,
        "role": "preliminary",
        "note": (
            "The extractor only separates quotes. It does not decide whether a claim is true. "
            "Each text field is an exact transcript quote and must be copied unchanged."
        ),
        "summary": summary,
        "extraction_status": "complete" if complete else "partial",
        "statement_count": len(visible),
        "checkable_count": sum(1 for item in visible if item["worth_checking"]),
        "statements": visible,
    }


def _write_claims(path: Path, video_id: str, summaries: list[str], statements: list[dict], *, complete: bool) -> None:
    summary = " ".join(part for part in summaries if part).strip()
    payload = _claims_payload(video_id, summary, statements, complete=complete)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _chunk_plan(segments: list) -> list[dict]:
    """Split timestamped lines into chronological chunks with a short overlap."""
    rows = []
    for segment in segments or []:
        if not isinstance(segment, dict):
            continue
        text = _plain_caption(str(segment.get("text") or ""))
        if not text:
            continue
        try:
            start = float(segment.get("start_seconds"))
            end = float(segment.get("end_seconds"))
        except (TypeError, ValueError):
            continue
        row = {"start_seconds": start, "end_seconds": max(start, end), "text": text}
        if rows and rows[-1]["text"] == text and rows[-1]["start_seconds"] == row["start_seconds"]:
            continue
        rows.append(row)
    if not rows:
        return []
    duration = max(row["end_seconds"] for row in rows)
    if duration <= CHUNK_SECONDS:
        return [{"primary_start": 0.0, "primary_end": duration, "context": [], "body": rows}]

    plans = []
    primary_start = 0.0
    while primary_start < duration:
        primary_end = min(duration, primary_start + CHUNK_SECONDS)
        context_start = max(0.0, primary_start - OVERLAP_SECONDS)
        look_end = min(duration, primary_end + OVERLAP_SECONDS)
        context = [
            row
            for row in rows
            if row["start_seconds"] < primary_start and row["end_seconds"] > context_start
        ][-3:]
        body = [row for row in rows if row["end_seconds"] > primary_start and row["start_seconds"] < look_end]
        plans.append(
            {
                "primary_start": primary_start,
                "primary_end": primary_end,
                "context": context,
                "body": body or context,
            }
        )
        if primary_end >= duration:
            break
        primary_start = primary_end
    return plans


def _in_primary(start: float, plan: dict, *, last: bool) -> bool:
    if last:
        return plan["primary_start"] <= start <= plan["primary_end"]
    return plan["primary_start"] <= start < plan["primary_end"]


def _same_occurrence(left: dict, right: dict) -> bool:
    if " ".join(str(left.get("text") or "").split()).lower() != " ".join(str(right.get("text") or "").split()).lower():
        return False
    return float(left["start_seconds"]) < float(right["end_seconds"]) and float(right["start_seconds"]) < float(
        left["end_seconds"]
    )


def _render_rows(rows: list[dict]) -> str:
    return "\n".join(f"[{_clock(float(row['start_seconds']))}] {row['text']}" for row in rows)


def _plain_caption(text: str) -> str:
    """Drop invisible caption marks and repeated spacing."""
    visible = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")
    return " ".join(visible.split())


_TALK_ASIDE = re.compile(
    r"\b(?:i'm|i am|we are|we're)\s+at\s+\d+\s+seconds\b"
    r"|\b\d+\s+seconds\s+right\s+now\b"
    r"|\b(?:doing|do)\s+(?:mine|this|it)\s+in\s+under\s+a\s+minute\b",
    re.IGNORECASE,
)


def _researchable_quote(text: str) -> bool:
    """Drop fragments, joke lead-ins, and the speaker timing their own talk.

    A real short claim such as "Inflation was 9 percent." still passes.
    """
    stripped = " ".join(str(text or "").split())
    if not stripped or _TALK_ASIDE.search(stripped):
        return False
    words = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?", stripped)
    if len(words) < 4:
        return False
    if stripped.endswith(",") and len(words) < 6:
        return False
    return True


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
