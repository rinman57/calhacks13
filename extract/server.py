"""Local server for the YouTube page panel.

Run from anywhere:

    python -m extract.server

The page panel sends the watch URL here. This process saves that video under data/.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

from extract.youtube import ExtractionResult, extract_youtube, parse_video_id

HOST = "127.0.0.1"
PORT = 8765
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "data"

_jobs: dict[str, dict] = {}
_lock = threading.Lock()


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), ExtractHandler)
    print(f"Extractor listening on http://{HOST}:{PORT}")
    print(f"Saving videos to {OUTPUT_ROOT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping extractor.")
    finally:
        server.server_close()


class ExtractHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self) -> None:
        self._send(204, b"")

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/health":
            self._send_json({"ok": True})
            return
        if path in {"/extension/content.js", "/extension/content.css"}:
            self._send_file(ROOT / path.lstrip("/"))
            return
        if path.startswith("/jobs/"):
            job_id = path.removeprefix("/jobs/")
            with _lock:
                job = _jobs.get(job_id)
                payload = None if job is None else json.loads(json.dumps(job))
            if payload is None:
                self._send_json({"error": "Unknown extraction."}, status=404)
                return
            self._send_json(payload)
            return
        self._send_json({"error": "Not found."}, status=404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/extract":
            self._send_json({"error": "Not found."}, status=404)
            return
        try:
            payload = json.loads(self._read_body().decode("utf-8") or "{}")
            video_id = parse_video_id(str(payload.get("url") or payload.get("video_id") or ""))
        except (json.JSONDecodeError, ValueError, UnicodeDecodeError) as exc:
            self._send_json({"error": f"Send a YouTube video URL. {exc}"}, status=400)
            return

        job_id = f"{video_id}-{uuid4().hex[:8]}"
        job = {
            "job_id": job_id,
            "video_id": video_id,
            "watch_url": f"https://www.youtube.com/watch?v={video_id}",
            "status": "running",
            "stage": "starting",
            "result": None,
            "error": None,
        }
        with _lock:
            _jobs[job_id] = job
        threading.Thread(target=_run_job, args=(job_id, video_id), daemon=True).start()
        self._send_json(job, status=202)

    def log_message(self, format: str, *args: object) -> None:
        return

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or "0")
        if length > 20_000:
            raise ValueError("Request is too large.")
        return self.rfile.read(length)

    def _send(self, status: int, body: bytes, content_type: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self._send(status, body)

    def _send_file(self, path: Path) -> None:
        if not path.is_file():
            self._send_json({"error": "Not found."}, status=404)
            return
        content_type = "text/css" if path.suffix == ".css" else "text/javascript; charset=utf-8"
        self._send(200, path.read_bytes(), content_type=content_type)


def _run_job(job_id: str, video_id: str) -> None:
    def on_progress(stage: str) -> None:
        _update(job_id, stage=stage)
        print(f"{video_id}: {stage}")

    try:
        result = extract_youtube(video_id, OUTPUT_ROOT, on_progress=on_progress)
    except Exception as exc:
        _update(job_id, status="error", stage="error", error=str(exc))
        print(f"{video_id}: error: {exc}")
        return
    _update(job_id, status="done" if result.ok else "error", stage="done", result=_public_result(result))
    print(f"{video_id}: finished")


def _update(job_id: str, **fields: object) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if job is not None:
            job.update(fields)


def _public_result(result: ExtractionResult) -> dict:
    metadata: dict = {}
    if result.metadata_path is not None and result.metadata_path.exists():
        metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    preview: list[str] = []
    if result.transcript_text_path is not None and result.transcript_text_path.exists():
        preview = [
            line
            for line in result.transcript_text_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ][:8]
    return {
        "video_id": result.video_id,
        "title": metadata.get("title"),
        "channel": metadata.get("channel"),
        "duration_seconds": metadata.get("duration_seconds"),
        "output_dir": str(result.output_dir),
        "files": {
            "metadata": _file_info(result.metadata_path),
            "transcript": _file_info(result.transcript_path),
            "audio": _file_info(result.audio_path),
            "video": _file_info(result.video_path),
            "claims": _file_info(result.claims_path),
        },
        "caption_languages": metadata.get("caption_languages"),
        "transcript_source": metadata.get("transcript_source"),
        "transcript_preview": preview,
        "claim_preview": _claim_preview(result.claims_path),
        "checkable_count": _checkable_count(result.claims_path),
        "warnings": result.warnings,
    }


def _claim_preview(path: Path | None) -> list[str]:
    payload = _read_claims(path)
    lines = []
    for item in payload.get("statements") or []:
        if not isinstance(item, dict) or not item.get("worth_checking"):
            continue
        label = item.get("preliminary_label") or "unverified"
        confidence = item.get("label_confidence")
        score = "" if confidence is None else f" {confidence}"
        lines.append(f"[{_clock(item.get('start_seconds'))}] {label}{score} {item.get('text') or ''}".strip())
    if lines:
        return lines[:8]
    summary = str(payload.get("summary") or "").strip()
    return [summary] if summary else []


def _checkable_count(path: Path | None) -> int:
    payload = _read_claims(path)
    count = payload.get("checkable_count")
    if isinstance(count, int):
        return count
    return 0


def _read_claims(path: Path | None) -> dict:
    if path is None or not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _clock(seconds: object) -> str:
    try:
        total_ms = int(round(float(seconds) * 1000))
    except (TypeError, ValueError):
        total_ms = 0
    minutes, remainder = divmod(max(0, total_ms), 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{minutes:02d}:{secs:02d}.{millis:03d}"


def _file_info(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        return None
    return {"name": path.name, "bytes": path.stat().st_size}


if __name__ == "__main__":
    main()
