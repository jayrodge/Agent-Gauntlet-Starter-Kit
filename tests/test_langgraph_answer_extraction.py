from __future__ import annotations

from pathlib import Path
import sys
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
LANGGRAPH_DIR = ROOT / "examples" / "langgraph"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(LANGGRAPH_DIR) not in sys.path:
    sys.path.insert(0, str(LANGGRAPH_DIR))

from arena_clients.proxy_headers import build_proxy_headers  # noqa: E402
from agent import (  # noqa: E402
    _build_mcp_sse_connection,
    _derive_answer_from_challenge,
    _derive_answer_from_evidence,
    _extract_ordered_answer_from_rules,
    _is_invalid_answer_candidate,
    extract_answer,
)


BREAKFAST_RULES = (
    "Find the unique chronological order (1st to 5th) that satisfies all "
    "constraints. Format: 'Name1, Name2, Name3, Name4, Name5'"
)
FAKE_LLM_HOST = "https://llm-proxy.example.test"
FAKE_LLM_API_KEY = "test-api-key"


class LangGraphAnswerExtractionTests(unittest.TestCase):
    def test_rejects_placeholder_format_from_rules(self) -> None:
        fallback = _extract_ordered_answer_from_rules(BREAKFAST_RULES)

        self.assertTrue(_is_invalid_answer_candidate(fallback))

    def test_extracts_nested_answer_line_from_reasoning(self) -> None:
        raw = "\n".join(
            [
                "Thus order: Eve, Bob, Alice, Charlie, David.",
                "Thus final answer: ANSWER: Eve, Bob, Alice, Charlie, David",
            ]
        )

        self.assertEqual(
            extract_answer(raw, BREAKFAST_RULES),
            "Eve, Bob, Alice, Charlie, David",
        )

    def test_placeholder_format_is_not_valid_final_answer(self) -> None:
        self.assertTrue(
            _is_invalid_answer_candidate("Name1, Name2, Name3, Name4, Name5")
        )

    def test_extracts_order_line_from_thought_stream(self) -> None:
        raw = "\n".join(
            [
                "Exactly one person between Eve and Alice: yes.",
                "Thus unique order: Eve, Bob, Alice, Charlie, David.",
                "Output format: 'Name1, Name2, Name3, Name4, Name5' exactly.",
            ]
        )

        self.assertEqual(
            extract_answer(raw, BREAKFAST_RULES),
            "Eve, Bob, Alice, Charlie, David",
        )


class LangGraphProxyHeaderTests(unittest.TestCase):
    def test_mcp_credential_is_carried_in_header_not_url(self) -> None:
        connection = _build_mcp_sse_connection(
            "https://arena.example.test/mcp",
            "arena-secret",
        )

        self.assertEqual(connection["url"], "https://arena.example.test/mcp/sse")
        self.assertEqual(
            connection["headers"],
            {"X-Arena-API-Key": "arena-secret"},
        )
        self.assertNotIn("arena-secret", connection["url"])

    def test_build_proxy_headers_uses_agent_and_round_scope(self) -> None:
        self.assertEqual(
            build_proxy_headers("team-beacon", "round-123"),
            {"X-Agent-ID": "team-beacon", "X-Round-ID": "round-123"},
        )

    def test_evidence_fallback_passes_proxy_headers_to_openai_client(self) -> None:
        captured_headers: dict[str, str] = {}

        class FakeOpenAI:
            def __init__(self, **kwargs):
                captured_headers.update(kwargs.get("default_headers") or {})
                self.chat = types.SimpleNamespace(
                    completions=types.SimpleNamespace(create=self._create)
                )

            def _create(self, **_kwargs):
                message = types.SimpleNamespace(content="Eve, Bob, Alice, Charlie, David")
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(choices=[choice])

        fake_openai = types.ModuleType("openai")
        fake_openai.OpenAI = FakeOpenAI

        with mock.patch.dict(sys.modules, {"openai": fake_openai}):
            answer = _derive_answer_from_evidence(
                llm_host=FAKE_LLM_HOST,
                llm_api_key=FAKE_LLM_API_KEY,
                model_name="gpt-oss-20b",
                challenge_description="Find the order.",
                challenge_rules=BREAKFAST_RULES,
                evidence_lines=["Order: Eve, Bob, Alice, Charlie, David"],
                agent_id="team-beacon",
                usage_scope="round-123",
            )

        self.assertEqual(answer, "Eve, Bob, Alice, Charlie, David")
        self.assertEqual(
            captured_headers,
            {"X-Agent-ID": "team-beacon", "X-Round-ID": "round-123"},
        )

    def test_direct_fallback_passes_proxy_headers_to_openai_client(self) -> None:
        captured_headers: dict[str, str] = {}

        class FakeOpenAI:
            def __init__(self, **kwargs):
                captured_headers.update(kwargs.get("default_headers") or {})
                self.chat = types.SimpleNamespace(
                    completions=types.SimpleNamespace(create=self._create)
                )

            def _create(self, **_kwargs):
                message = types.SimpleNamespace(content="Eve, Bob, Alice, Charlie, David")
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(choices=[choice])

        fake_openai = types.ModuleType("openai")
        fake_openai.OpenAI = FakeOpenAI

        with mock.patch.dict(sys.modules, {"openai": fake_openai}):
            answer = _derive_answer_from_challenge(
                llm_host=FAKE_LLM_HOST,
                llm_api_key=FAKE_LLM_API_KEY,
                model_name="gpt-oss-20b",
                challenge_description="Find the order.",
                challenge_rules=BREAKFAST_RULES,
                clues=["Eve is first."],
                evidence_lines=["Order: Eve, Bob, Alice, Charlie, David"],
                agent_id="team-beacon",
                usage_scope="round-123",
            )

        self.assertEqual(answer, "Eve, Bob, Alice, Charlie, David")
        self.assertEqual(
            captured_headers,
            {"X-Agent-ID": "team-beacon", "X-Round-ID": "round-123"},
        )


if __name__ == "__main__":
    unittest.main()
