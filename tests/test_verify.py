"""Mocked tests for research-then-assessment. No network calls."""

from __future__ import annotations

import json
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from extract.server import _card_sources
from extract.verify import (
    MAX_CONTINUATIONS,
    validate_assessment,
    verify_folder,
)

SECRET = "sk-ant-secretvalue"
DOC = "The Bureau of Labor Statistics reported CPI inflation near 1.4 percent in January 2021."


def response(stop_reason, blocks, searches=1, fetches=0):
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=blocks,
        usage=SimpleNamespace(
            input_tokens=5,
            output_tokens=4,
            server_tool_use=SimpleNamespace(web_search_requests=searches, web_fetch_requests=fetches),
        ),
    )


def search_and_fetch():
    return [
        {"type": "server_tool_use", "id": "search-1", "name": "web_search", "input": {"query": "US CPI January 2021"}},
        {
            "type": "web_search_tool_result",
            "tool_use_id": "search-1",
            "content": [
                {
                    "type": "web_search_result",
                    "url": "https://www.bls.gov/cpi/",
                    "title": "CPI Home",
                    "page_age": "January 2026",
                }
            ],
        },
        {"type": "server_tool_use", "id": "fetch-1", "name": "web_fetch", "input": {"url": "https://www.bls.gov/cpi/"}},
        {
            "type": "web_fetch_tool_result",
            "tool_use_id": "fetch-1",
            "content": {
                "type": "web_fetch_result",
                "url": "https://www.bls.gov/cpi/",
                "retrieved_at": "2026-10-24T19:00:00Z",
                "content": {
                    "type": "document",
                    "title": "CPI Home",
                    "source": {"type": "text", "media_type": "text/plain", "data": DOC},
                },
            },
        },
        {"type": "text", "text": '{"research_gaps": ["The excerpt does not name a monthly index."]}'},
    ]


def assessment(verdict="contradicted", source_id="s1", passage_id="p2"):
    explanation = "The retrieved page says inflation was near 1.4 percent, which contradicts the claim."
    return response(
        "end_turn",
        [
            {
                "type": "text",
                "text": json.dumps(
                    {
                        "verdict": verdict,
                        "headline": "Inflation was not 9 percent in January 2021",
                        "explanation": explanation,
                        "context": "The comparison has to use the January 2021 CPI.",
                        "evidence_refs": [
                            {"source_id": source_id, "passage_ids": [passage_id], "supports": "contradicts_claim"}
                        ],
                        "sentence_refs": [
                            {
                                "sentence": explanation,
                                "source_id": source_id,
                                "passage_ids": [passage_id],
                            }
                        ],
                        "limitations": ["Only the saved page excerpt was assessed."],
                    }
                ),
            }
        ],
    )


def insufficient():
    return response(
        "end_turn",
        [
            {
                "type": "text",
                "text": json.dumps(
                    {
                        "verdict": "insufficient_evidence",
                        "headline": "The saved evidence does not settle it",
                        "explanation": "No fetched page settles the claim.",
                        "context": "",
                        "evidence_refs": [],
                        "sentence_refs": [],
                        "limitations": ["No full page was retrieved."],
                    }
                ),
            }
        ],
    )


class FakeClient:
    def __init__(self, script):
        self.messages = self.Messages(script)

    class Messages:
        def __init__(self, script):
            self.script = list(script)
            self.calls = []
            self._lock = threading.Lock()

        def create(self, **kwargs):
            with self._lock:
                self.calls.append(kwargs)
                if not self.script:
                    raise AssertionError("unexpected model call")
                item = self.script.pop(0)
                if isinstance(item, Exception):
                    raise item
                return item


class VerifyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name) / "abcdefghijk_2026-10-24_12-00-00_PDT"
        self.folder.mkdir()
        self.claims = {
            "video_id": "abcdefghijk",
            "statements": [
                {
                    "id": "c1",
                    "kind": "opinion",
                    "text": "This administration did a better job.",
                    "start_seconds": 0,
                    "end_seconds": 2,
                    "worth_checking": False,
                    "research_question": "",
                    "preliminary_label": "opinion",
                    "why_this_label": "DO_NOT_LEAK_THIS_HINT",
                },
                {
                    "id": "c2",
                    "kind": "claim",
                    "text": "Inflation was at 9 percent.",
                    "start_seconds": 4.5,
                    "end_seconds": 8,
                    "worth_checking": True,
                    "research_question": "What was CPI in January 2021?",
                    "preliminary_label": "misinfo",
                    "label_confidence": 0.4,
                    "why_this_label": "DO_NOT_LEAK_THIS_HINT",
                },
                {
                    "id": "c3",
                    "kind": "claim",
                    "text": "Banks were limited to an 8 dollar fee.",
                    "start_seconds": 12,
                    "end_seconds": 20,
                    "worth_checking": False,
                    "research_question": "Was the fee capped?",
                    "preliminary_label": "true",
                    "why_this_label": "DO_NOT_LEAK_THIS_HINT",
                },
            ],
        }
        self._write("claims.json", self.claims)
        self._write(
            "metadata.json",
            {"video_id": "abcdefghijk", "upload_date": "20210120", "title": "Interview"},
        )
        self._write(
            "transcript.json",
            {
                "segments": [
                    {"start_seconds": 4.0, "end_seconds": 8.0, "text": "Inflation was about nine percent then."}
                ]
            },
        )
        self.original_claims = (self.folder / "claims.json").read_text(encoding="utf-8")

    def _write(self, name, payload):
        (self.folder / name).write_text(json.dumps(payload), encoding="utf-8")

    def _run(self, script, **kwargs):
        client = FakeClient(script)
        result = verify_folder(
            self.folder,
            client=client,
            api_key=SECRET,
            model="claude-sonnet-5-5",
            **kwargs,
        )
        evidence = json.loads((self.folder / "evidence.json").read_text(encoding="utf-8"))
        verdicts = json.loads((self.folder / "verdict.json").read_text(encoding="utf-8"))
        return client, result, evidence, verdicts

    def test_only_checkable_claims_keep_ids_and_skip_gemini_labels(self):
        client, _result, evidence, verdicts = self._run([response("end_turn", search_and_fetch(), fetches=1), assessment()])
        self.assertEqual([item["claim_id"] for item in evidence["claims"]], ["c2"])
        self.assertEqual(verdicts["results"][0]["claim_id"], "c2")
        self.assertEqual((self.folder / "claims.json").read_text(encoding="utf-8"), self.original_claims)
        blob = json.dumps(client.messages.calls)
        self.assertNotIn("DO_NOT_LEAK_THIS_HINT", blob)
        self.assertNotIn("preliminary_label", blob)
        self.assertEqual(client.messages.calls[0]["tools"][0]["name"], "web_search")
        self.assertEqual(client.messages.calls[0]["tools"][1]["name"], "web_fetch")
        self.assertNotIn("tools", client.messages.calls[1])
        self.assertNotIn("format", client.messages.calls[0]["output_config"])
        self.assertEqual(client.messages.calls[1]["output_config"]["effort"], "medium")
        self.assertEqual(client.messages.calls[1]["output_config"]["format"]["type"], "json_schema")
        claim = evidence["claims"][0]
        self.assertEqual(claim["searched_queries"], ["US CPI January 2021"])
        self.assertEqual(claim["sources"][0]["retrieval_status"], "fetched")
        self.assertIsNone(claim["sources"][0]["publisher"])
        self.assertIsNone(claim["sources"][0]["published_at"])
        self.assertEqual(claim["sources"][0]["page_age"], "January 2026")
        self.assertEqual(verdicts["results"][0]["verdict"], "contradicted")
        self.assertNotIn(SECRET, json.dumps(evidence) + json.dumps(verdicts))

    def test_invented_reference_is_rejected(self):
        _client, _result, evidence, verdicts = self._run(
            [response("end_turn", search_and_fetch(), fetches=1), assessment(source_id="s99"), assessment(source_id="s99")]
        )
        self.assertEqual(evidence["claims"][0]["research_status"], "complete")
        self.assertEqual(verdicts["results"][0]["assessment_status"], "failed")
        self.assertIsNone(verdicts["results"][0]["verdict"])
        self.assertIn("Unknown source id", verdicts["results"][0]["error"])

    def test_failed_research_zero_evidence_and_partial_stay_distinct(self):
        _client, result, evidence, verdicts = self._run(
            [
                response(
                    "end_turn",
                    [
                        {
                            "type": "web_search_tool_result",
                            "content": {"type": "web_search_tool_result_error", "error_code": "unavailable"},
                        }
                    ],
                )
            ]
        )
        self.assertEqual(evidence["claims"][0]["research_status"], "failed")
        self.assertEqual(evidence["claims"][0]["sources"], [])
        self.assertEqual(verdicts["results"][0]["assessment_status"], "skipped")
        self.assertIsNone(verdicts["results"][0]["verdict"])
        self.assertTrue(any(item.startswith("Research failed") for item in result.warnings))

        _client, result, evidence, verdicts = self._run(
            [response("refusal", [{"type": "text", "text": "no"}])]
        )
        self.assertEqual(evidence["claims"][0]["research_status"], "failed")
        self.assertEqual(verdicts["results"][0]["assessment_status"], "skipped")

        client, result, evidence, verdicts = self._run(
            [
                response(
                    "end_turn",
                    [
                        {"type": "server_tool_use", "name": "web_search", "input": {"query": "CPI"}},
                        {
                            "type": "web_search_tool_result",
                            "content": [
                                {"type": "web_search_result", "url": "https://example.com/cpi", "title": "CPI"}
                            ],
                        },
                        {
                            "type": "web_fetch_tool_result",
                            "content": {"type": "web_fetch_tool_result_error", "error_code": "url_not_accessible"},
                        },
                        {"type": "text", "text": '{"research_gaps": ["The page could not be fetched."]}'},
                    ],
                ),
                insufficient(),
            ]
        )
        self.assertEqual(evidence["claims"][0]["research_status"], "partial")
        self.assertEqual(len(evidence["claims"][0]["sources"]), 1)
        self.assertEqual(verdicts["results"][0]["verdict"], "insufficient_evidence")
        self.assertEqual(len(client.messages.calls), 2)
        self.assertTrue(any(item.startswith("Research was partial") for item in result.warnings))

    def test_pause_truncation_and_continuation_limit(self):
        original = MAX_CONTINUATIONS
        try:
            import extract.verify as verify_module

            verify_module.MAX_CONTINUATIONS = 0
            client, _result, evidence, _verdicts = self._run(
                [
                    response(
                        "pause_turn",
                        [
                            {"type": "server_tool_use", "name": "web_search", "input": {"query": "CPI history"}},
                            {
                                "type": "web_search_tool_result",
                                "content": [
                                    {"type": "web_search_result", "url": "https://example.com/cpi", "title": "CPI"}
                                ],
                            },
                        ],
                    ),
                    insufficient(),
                ]
            )
        finally:
            import extract.verify as verify_module

            verify_module.MAX_CONTINUATIONS = original
        self.assertEqual(evidence["claims"][0]["research_status"], "partial")
        self.assertEqual(len(client.messages.calls), 2)
        self.assertNotIn("tools", client.messages.calls[1])

        client, _result, evidence, verdicts = self._run(
            [response("max_tokens", [{"type": "text", "text": "cut off"}]), insufficient()]
        )
        self.assertEqual(evidence["claims"][0]["research_status"], "failed")
        self.assertEqual(verdicts["results"][0]["assessment_status"], "skipped")
        self.assertEqual(len(client.messages.calls), 1)

    def test_pause_turn_sends_the_assistant_message_back(self):
        import extract.verify as verify_module

        original = verify_module.MAX_CONTINUATIONS
        verify_module.MAX_CONTINUATIONS = 2
        try:
            client, _result, evidence, _verdicts = self._run(
                [
                    response("pause_turn", [{"type": "text", "text": "still searching"}]),
                    response("end_turn", search_and_fetch(), fetches=1),
                    assessment(),
                ]
            )
        finally:
            verify_module.MAX_CONTINUATIONS = original
        second = client.messages.calls[1]["messages"]
        self.assertEqual([item["role"] for item in second], ["user", "assistant"])
        self.assertEqual(second[1]["content"][0]["text"], "still searching")
        self.assertEqual(evidence["claims"][0]["research_status"], "complete")

    def test_exception_is_not_retried_and_secret_is_removed(self):
        self.claims["statements"][2]["worth_checking"] = True
        self._write("claims.json", self.claims)

        class Messages(FakeClient.Messages):
            def create(self, **kwargs):
                if "8 dollar" in json.dumps(kwargs.get("messages", "")):
                    with self._lock:
                        self.calls.append(kwargs)
                    raise RuntimeError(f"request failed {SECRET}")
                return super().create(**kwargs)

        client = FakeClient([response("end_turn", search_and_fetch(), fetches=1), assessment()])
        client.messages = Messages(client.messages.script)
        result = verify_folder(self.folder, client=client, api_key=SECRET, model="claude-sonnet-5-5")
        evidence = json.loads((self.folder / "evidence.json").read_text(encoding="utf-8"))
        self.assertEqual(evidence["claims"][0]["research_status"], "complete")
        self.assertEqual(evidence["claims"][1]["claim_id"], "c3")
        self.assertEqual(evidence["claims"][1]["research_status"], "failed")
        saved = (self.folder / "evidence.json").read_text(encoding="utf-8")
        self.assertNotIn(SECRET, saved)
        self.assertNotIn(SECRET, " ".join(result.warnings))
        self.assertIn("[redacted]", saved)
        self.assertEqual(len(client.messages.calls), 3)

        client, _result, evidence, verdicts = self._run(
            [response("end_turn", search_and_fetch(), fetches=1), assessment()]
        )
        self.assertEqual(evidence["claims"][0]["searched_queries"], ["US CPI January 2021"])
        self.assertEqual(evidence["claims"][1]["research_status"], "complete")
        self.assertEqual(len(client.messages.calls), 2)
        self.assertEqual(verdicts["results"][0]["verdict"], "contradicted")

    def test_cache_reuses_completed_work_until_the_claim_changes(self):
        self._run([response("end_turn", search_and_fetch(), fetches=1), assessment()])
        client, _result, _evidence, _verdicts = self._run([])
        self.assertEqual(client.messages.calls, [])

        self.claims["statements"][1]["text"] = "Inflation was at 9 percent when the term began."
        self._write("claims.json", self.claims)
        client, _result, evidence, _verdicts = self._run(
            [response("end_turn", search_and_fetch(), fetches=1), assessment()]
        )
        self.assertEqual(len(client.messages.calls), 2)
        prompt = client.messages.calls[0]["messages"][0]["content"]
        self.assertIn("term began", prompt)
        self.assertNotEqual(evidence["claims"][0]["input_hash"], "")

    def test_deadline_skips_the_model(self):
        client, result, evidence, verdicts = self._run([], deadline_seconds=0)
        self.assertEqual(client.messages.calls, [])
        self.assertEqual(evidence["claims"][0]["research_status"], "failed")
        self.assertEqual(verdicts["results"][0]["assessment_status"], "skipped")
        self.assertTrue(any("deadline" in item.lower() for item in result.warnings))

    def test_validator_rejects_unknown_passage_and_snippet_only_support(self):
        sources = [
            {
                "source_id": "s1",
                "passages": [{"passage_id": "p1", "text": "CPI", "content_type": "search_snippet"}],
            }
        ]
        with self.assertRaises(ValueError):
            validate_assessment(
                {
                    "verdict": "supported",
                    "headline": "Supported",
                    "explanation": "The page supports it.",
                    "context": "",
                    "evidence_refs": [{"source_id": "s9", "passage_ids": ["p1"], "supports": "supports_claim"}],
                    "sentence_refs": [],
                    "limitations": [],
                },
                sources,
            )
        with self.assertRaises(ValueError):
            validate_assessment(
                {
                    "verdict": "supported",
                    "headline": "Supported",
                    "explanation": "The page supports it.",
                    "context": "",
                    "evidence_refs": [{"source_id": "s1", "passage_ids": ["p1"], "supports": "supports_claim"}],
                    "sentence_refs": [
                        {"sentence": "The page supports it.", "source_id": "s1", "passage_ids": ["p1"]}
                    ],
                    "limitations": [],
                },
                sources,
            )

    def test_public_cards_keep_only_http_links(self):
        cards = _card_sources(
            {
                "sources": [
                    {"source_id": "s1", "url": "javascript:alert(1)", "title": "bad"},
                    {"source_id": "s2", "url": "https://example.com/cpi", "title": "CPI"},
                ]
            },
            {"evidence_refs": [{"source_id": "s1"}, {"source_id": "s2"}]},
        )
        self.assertEqual(cards, [{"title": "CPI", "url": "https://example.com/cpi"}])


if __name__ == "__main__":
    unittest.main()
