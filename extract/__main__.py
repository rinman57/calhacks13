"""Command line: python -m extract <youtube-url>"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from extract.youtube import extract_youtube


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Save one YouTube video, its transcript, and a first-pass claim list."
    )
    parser.add_argument("url", help="YouTube watch URL, short URL, or 11-character video id")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data"),
        help="Directory for per-video folders (default: data/)",
    )
    parser.add_argument(
        "--languages",
        default="en",
        help="Comma-separated caption language codes, highest priority first (default: en)",
    )
    args = parser.parse_args(argv)
    languages = [code.strip() for code in args.languages.split(",") if code.strip()]

    try:
        result = extract_youtube(
            args.url,
            args.out,
            languages=languages,
        )
    except Exception as exc:
        print(f"Extraction failed: {exc}", file=sys.stderr)
        return 1

    print(f"video_id: {result.video_id}")
    print(f"output: {result.output_dir}")
    print(f"metadata: {result.metadata_path}")
    print(f"transcript: {result.transcript_path or 'not available'}")
    print(f"audio: {result.audio_path or 'not downloaded'}")
    print(f"claims: {result.claims_path or 'not created'}")
    print(f"evidence: {result.evidence_path or 'not created'}")
    print(f"verdicts: {result.verdict_path or 'not created'}")
    for warning in result.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
