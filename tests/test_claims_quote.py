"""The claim text has to be the transcript's own wording."""

from __future__ import annotations

import unittest

from extract.claims import _normalize, _prompt, quote_from_transcript
from extract.verify import cap_words, research_prompt
from extract.youtube import _expand_languages


class QuoteTests(unittest.TestCase):
    def setUp(self):
        self.segments = [
            {"start_seconds": 46.287, "end_seconds": 50.06, "text": "I'm trying to think of the last time I watched an 18-minute TED talk."},
            {"start_seconds": 61.45, "end_seconds": 64.58, "text": "Why are balloons so expensive?"},
            {"start_seconds": 65.705, "end_seconds": 66.941, "text": "Woody Roseland: Inflation."},
        ]

    def test_exact_quote_is_kept(self):
        quote = "I'm trying to think of the last time I watched an 18-minute TED talk."
        text, start, end = quote_from_transcript(self.segments, 46.287, 49.5, quote)
        self.assertEqual(text, quote)
        self.assertEqual(start, 46.287)
        self.assertEqual(end, 50.06)

    def test_paraphrase_is_replaced_by_the_transcript_line(self):
        text, start, end = quote_from_transcript(
            self.segments,
            46.287,
            49.5,
            "The traditional maximum length for a flagship TED talk format is 18 minutes.",
        )
        self.assertEqual(text, "I'm trying to think of the last time I watched an 18-minute TED talk.")
        self.assertEqual((start, end), (46.287, 50.06))

    def test_compact_timestamp_finds_the_spoken_lines(self):
        text, start, end = quote_from_transcript(
            self.segments,
            101.45,
            106.941,
            "Balloons are expensive because of inflation.",
        )
        self.assertEqual(text, "Why are balloons so expensive? Woody Roseland: Inflation.")
        self.assertEqual(start, 61.45)
        self.assertEqual(end, 66.941)

    def test_settled_and_opinion_lines_are_not_sent_to_research(self):
        segments = [
            {"start_seconds": 1, "end_seconds": 2, "text": "The president is on screen."},
            {"start_seconds": 3, "end_seconds": 4, "text": "Inflation was 9 percent."},
            {"start_seconds": 5, "end_seconds": 6, "text": "I last watched a long talk years ago."},
        ]
        payload = _normalize(
            {
                "summary": "A clip.",
                "statements": [
                    {
                        "kind": "claim",
                        "text": "The president is on screen.",
                        "start_seconds": 1,
                        "end_seconds": 2,
                        "worth_checking": True,
                        "preliminary_label": "true",
                        "label_confidence": 0.9,
                        "why_this_label": "Visible.",
                        "research_question": "Is that the president?",
                    },
                    {
                        "kind": "claim",
                        "text": "Inflation was 9 percent.",
                        "start_seconds": 3,
                        "end_seconds": 4,
                        "worth_checking": True,
                        "preliminary_label": "misinfo",
                        "label_confidence": 0.4,
                        "why_this_label": "The video does not show the rate.",
                        "research_question": "Was inflation 9 percent?",
                    },
                    {
                        "kind": "opinion",
                        "text": "I last watched a long talk years ago.",
                        "start_seconds": 5,
                        "end_seconds": 6,
                        "worth_checking": True,
                        "preliminary_label": "opinion",
                        "label_confidence": 0.8,
                        "why_this_label": "Personal memory.",
                        "research_question": "Do talks last 18 minutes?",
                    },
                ],
            },
            "abc",
            segments,
        )
        by_text = {item["text"]: item for item in payload["statements"]}
        settled = by_text["The president is on screen."]
        self.assertFalse(settled["worth_checking"])
        self.assertEqual(settled["research_question"], "")
        self.assertTrue(by_text["Inflation was 9 percent."]["worth_checking"])
        personal = by_text["I last watched a long talk years ago."]
        self.assertFalse(personal["worth_checking"])
        self.assertEqual(personal["research_question"], "")
        self.assertEqual(settled["visual_note"], "")
        self.assertEqual(payload["checkable_count"], 1)

    def test_english_request_includes_regional_caption_tracks(self):
        self.assertEqual(_expand_languages(["en"], ["en-US", "es"]), ["en-US"])
        self.assertEqual(_expand_languages(["en"], ["en", "en-US"]), ["en", "en-US"])

    def test_prompt_keeps_research_on_the_spoken_quote(self):
        prompt = _prompt({"title": "Talk", "channel": "Channel"}, "Hello.")
        self.assertIn("speaker's own experience", prompt)
        self.assertIn("must not ask about a broader claim", prompt)
        self.assertIn("unverified, misinfo, or misleading", prompt)

    def test_context_summary_stops_at_thirty_words(self):
        text = (
            "The 18-minute figure is a real convention in official TED materials. "
            "Search snippets from the TEDx speaker guide and TEDx rules are consistent "
            "with it being a norm rather than a strict limit. Press claims of a strict "
            "cap were only seen as headlines or snippets."
        )
        summary = cap_words(text, 30)
        self.assertLessEqual(len(summary.split()), 30)
        self.assertEqual(summary, "The 18-minute figure is a real convention in official TED materials.")

    def test_research_prompt_passes_the_quote_through_unchanged(self):
        quote = "I'm trying to think of the last time I watched an 18-minute TED talk."
        prompt = research_prompt(
            {"id": "c3", "text": quote, "research_question": "Is 18 minutes a TED limit?"},
            context="[00:46.287] " + quote,
            published="2017-09-13",
        )
        self.assertIn(f"Original transcript quote: {quote}", prompt)
        self.assertIn("Do not rewrite it.", prompt)
        self.assertNotIn("paraphrase", prompt.lower())


if __name__ == "__main__":
    unittest.main()
