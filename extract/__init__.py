"""First pipeline step: pull metadata, transcript, and media from a YouTube URL."""

from extract.youtube import ExtractionResult, extract_youtube

__all__ = ["ExtractionResult", "extract_youtube"]
