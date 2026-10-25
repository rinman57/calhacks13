"""The claim text has to be the transcript's own wording."""

from __future__ import annotations

import unittest

from extract.claims import _chunk_plan, _in_primary, _normalize, _prompt, _researchable_quote, quote_from_transcript
from extract.verify import cap_words, research_prompt
from extract.youtube import _expand_languages, _parse_vtt, _spoken_text


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
                        "research_basis": "skip",
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
                        "research_basis": "public_affairs",
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
        self.assertNotIn("The president is on screen.", by_text)
        spoken_claim = by_text["Inflation was 9 percent."]
        self.assertTrue(spoken_claim["worth_checking"])
        self.assertEqual(spoken_claim["preliminary_label"], "unverified")
        self.assertEqual(spoken_claim["research_question"], "")
        self.assertIsNone(spoken_claim["label_confidence"])
        self.assertTrue(by_text["Inflation was 9 percent."]["worth_checking"])
        self.assertNotIn("I last watched a long talk years ago.", by_text)
        self.assertEqual(spoken_claim["visual_note"], "")
        self.assertEqual(payload["checkable_count"], 1)

    def test_later_chunk_owns_a_quote_that_starts_after_the_boundary(self):
        segments = [
            {"start_seconds": index * 20, "end_seconds": index * 20 + 5, "text": f"Line {index} says a fact."}
            for index in range(10)
        ]
        plans = _chunk_plan(segments)
        self.assertGreater(len(plans), 1)
        self.assertEqual(plans[0]["primary_end"], 75.0)
        self.assertTrue(_in_primary(60.0, plans[0], last=False))
        self.assertFalse(_in_primary(80.0, plans[0], last=False))
        self.assertTrue(_in_primary(80.0, plans[1], last=False))
        self.assertTrue(plans[1]["context"])

    def test_caption_file_header_is_not_speech(self):
        self.assertEqual(_spoken_text("WEBVTT #Elemental Media Engine(TM) 2.11.0.400122"), "")
        self.assertEqual(
            _parse_vtt(
                "\n".join(
                    [
                        "WEBVTT #Elemental Media Engine(TM) 2.11.0.400122",
                        "",
                        "00:00:01.540 --> 00:00:05.340",
                        "And they say the noise causes cancer.",
                    ]
                )
            ),
            [
                {
                    "start_seconds": 1.54,
                    "duration_seconds": 3.8,
                    "end_seconds": 5.34,
                    "text": "And they say the noise causes cancer.",
                }
            ],
        )

    def test_english_request_includes_regional_caption_tracks(self):
        self.assertEqual(_expand_languages(["en"], ["en-US", "es"]), ["en-US"])
        self.assertEqual(_expand_languages(["en"], ["en", "en-US"]), ["en", "en-US"])

    def test_prompt_keeps_research_on_the_spoken_quote(self):
        prompt = _prompt({"title": "Talk", "channel": "Channel"}, "Hello.")
        self.assertIn("speaker's own experience", prompt)
        self.assertIn("Do not paraphrase", prompt)
        self.assertIn("back in 2009,", prompt)
        self.assertIn("44 seconds", prompt)
        self.assertIn("would matter if it were false", prompt)
        self.assertNotIn('kind "opinion"', prompt)
        self.assertNotIn("preliminary_label", prompt)

    def test_prompt_judges_claims_by_the_kind_of_video(self):
        prompt = _prompt(
            {
                "title": "Eat Everything In A Grocery Store, Win $1,000,000",
                "channel": "MrBeast",
                "description": "I trapped 4 strangers in an entire supermarket.\nhttps://example.com/store\n\nLater paragraph",
            },
            "Hello.",
        )
        self.assertIn("Channel: MrBeast", prompt)
        self.assertIn("trapped 4 strangers", prompt)
        self.assertNotIn("https://example.com", prompt)
        self.assertNotIn("Later paragraph", prompt)
        self.assertIn("premise of a stunt", prompt)

    def test_joke_fragments_and_talk_timing_are_not_researched(self):
        self.assertFalse(_researchable_quote("back in 2009,"))
        self.assertFalse(_researchable_quote("I'm at 44 seconds right now;"))
        self.assertTrue(_researchable_quote("Inflation was 9 percent."))
        self.assertTrue(_researchable_quote("they say the noise causes cancer"))
        payload = _normalize(
            {
                "summary": "A short comic talk.",
                "statements": [
                    {"kind": "claim", "research_basis": "public_affairs", "text": "back in 2009,", "start_seconds": 36.141, "end_seconds": 37.855},
                    {
                        "kind": "claim",
                        "text": "I'm at 44 seconds right now;",
                        "research_basis": "other_material_fact",
                        "start_seconds": 57.56,
                        "end_seconds": 59.155,
                    },
                    {
                        "kind": "claim",
                        "text": "they say the noise causes cancer",
                        "research_basis": "health_safety",
                        "start_seconds": 1,
                        "end_seconds": 2,
                    },
                ],
            },
            "abc",
            [
                {"start_seconds": 36.141, "end_seconds": 37.855, "text": "back in 2009,"},
                {"start_seconds": 57.56, "end_seconds": 59.155, "text": "I'm at 44 seconds right now;"},
                {"start_seconds": 1, "end_seconds": 2, "text": "they say the noise causes cancer"},
            ],
        )
        by_text = {item["text"]: item for item in payload["statements"]}
        self.assertNotIn("back in 2009,", by_text)
        self.assertNotIn("I'm at 44 seconds right now;", by_text)
        self.assertTrue(by_text["they say the noise causes cancer"]["worth_checking"])
        self.assertEqual(payload["checkable_count"], 1)

    def test_unqualified_candidates_never_reach_the_research_callback(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from extract.claims import extract_claims

        cases = [
            ("The store has 4.3 million calories of food.", "skip"),
            ("This could take us over a year.", "skip"),
            ("I ate breakfast at home today.", None),
            ("there's too much money involved with sugar the", "unknown"),
            ("they say the noise causes cancer", "health_safety"),
            ("Inflation was 9 percent.", "public_affairs"),
        ]
        segments, candidates = [], []
        for index, (text, basis) in enumerate(cases):
            segment = {"text": text, "start_seconds": index * 3, "end_seconds": index * 3 + 2}
            segments.append(segment)
            candidate = {"kind": "claim", **segment}
            if basis is not None:
                candidate["research_basis"] = basis
            candidates.append(candidate)
        submitted = []
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "metadata.json").write_text(json.dumps({"video_id": "abc"}))
            (path / "transcript.json").write_text(json.dumps({"segments": segments}))
            with patch("extract.claims.load_project_env"), patch.dict(
                "os.environ", {"ANTHROPIC_API_KEY": "test-key"}
            ), patch("extract.claims._generate", return_value={"summary": "Clip", "statements": candidates}):
                result = extract_claims(path, on_claim=submitted.append)
            saved = json.loads(result.read_text())
        self.assertEqual([item["text"] for item in submitted], [text for text, _ in cases[-2:]])
        self.assertEqual(saved["checkable_count"], 2)
        self.assertTrue(all(item["worth_checking"] for item in submitted))
        self.assertEqual(submitted[0]["research_basis"], "health_safety")

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
        self.assertIn("did not judge whether this quote is true", prompt)
        self.assertNotIn("paraphrase", prompt.lower())


if __name__ == "__main__":
    unittest.main()
