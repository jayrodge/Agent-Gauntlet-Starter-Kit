from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
CREWAI_EXAMPLE = ROOT / "examples" / "crewai"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(CREWAI_EXAMPLE) not in sys.path:
    sys.path.insert(0, str(CREWAI_EXAMPLE))

from arena_clients.proxy_headers import build_proxy_headers  # noqa: E402

FAKE_LLM_HOST = "https://llm-proxy.example.test"
FAKE_LLM_API_KEY = "test-api-key"


class _BaseTool:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def _load_crewai_agent_module():
    crewai = types.ModuleType("crewai")
    tools = types.ModuleType("crewai.tools")
    tools.BaseTool = _BaseTool
    crewai.tools = tools

    with mock.patch.dict(sys.modules, {"crewai": crewai, "crewai.tools": tools}):
        spec = importlib.util.spec_from_file_location(
            "crewai_agent_under_test",
            CREWAI_EXAMPLE / "agent.py",
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


class CrewAIProxyHeaderTests(unittest.TestCase):
    def _install_fake_openai(self, content: str, captured_headers: dict[str, str]):
        class FakeOpenAI:
            def __init__(self, **kwargs):
                captured_headers.update(kwargs.get("default_headers") or {})
                self.chat = types.SimpleNamespace(
                    completions=types.SimpleNamespace(create=self._create)
                )

            def _create(self, **_kwargs):
                message = types.SimpleNamespace(content=content)
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(choices=[choice])

        fake_openai = types.ModuleType("openai")
        fake_openai.OpenAI = FakeOpenAI
        return mock.patch.dict(sys.modules, {"openai": fake_openai})

    def test_build_proxy_headers_uses_agent_and_round_scope(self) -> None:
        self.assertEqual(
            build_proxy_headers("team-cipher", "round-123"),
            {"X-Agent-ID": "team-cipher", "X-Round-ID": "round-123"},
        )

    def test_build_proxy_headers_falls_back_to_environment(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"AGENT_ID": "team-cipher", "ARENA_USAGE_SCOPE": "round-123"},
            clear=False,
        ):
            headers = build_proxy_headers()

        self.assertEqual(
            headers,
            {"X-Agent-ID": "team-cipher", "X-Round-ID": "round-123"},
        )

    def test_crewai_llm_receives_proxy_headers_when_supported(self) -> None:
        module = _load_crewai_agent_module()

        class DummyLLM:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        llm = module._make_crewai_llm(
            DummyLLM,
            model="gpt-oss-20b",
            api_key=FAKE_LLM_API_KEY,
            base_url=FAKE_LLM_HOST,
            temperature=0.0,
            max_tokens=512,
            proxy_headers={"X-Agent-ID": "team-cipher"},
            timeout_s=45,
        )

        self.assertEqual(llm.kwargs["extra_headers"], {"X-Agent-ID": "team-cipher"})
        self.assertEqual(llm.kwargs["timeout"], 45)
        self.assertEqual(llm.kwargs["model"], "openai/gpt-oss-20b")

    def test_crewai_llm_rejects_versions_without_extra_headers(self) -> None:
        module = _load_crewai_agent_module()

        class StrictLLM:
            def __init__(self, **kwargs):
                if "extra_headers" in kwargs:
                    raise TypeError("unexpected extra_headers")
                self.kwargs = kwargs

        with self.assertRaisesRegex(
            RuntimeError,
            "does not support proxy attribution headers",
        ):
            module._make_crewai_llm(
                StrictLLM,
                model="gpt-oss-20b",
                api_key=FAKE_LLM_API_KEY,
                base_url=FAKE_LLM_HOST,
                temperature=0.0,
                max_tokens=512,
                proxy_headers={"X-Agent-ID": "team-cipher"},
            )

    def test_repair_text_answer_passes_proxy_headers_to_openai_client(self) -> None:
        module = _load_crewai_agent_module()
        captured_headers: dict[str, str] = {}

        with self._install_fake_openai(
            "ANSWER: Eve, Bob, Alice, Charlie, David",
            captured_headers,
        ):
            answer = module._repair_text_answer(
                raw_content="The answer is Eve, Bob, Alice, Charlie, David.",
                rules="Return five names.",
                llm_host=FAKE_LLM_HOST,
                llm_api_key=FAKE_LLM_API_KEY,
                repair_model="gpt-oss-20b",
                agent_id="team-beacon",
                usage_scope="round-123",
            )

        self.assertEqual(answer, "Eve, Bob, Alice, Charlie, David")
        self.assertEqual(
            captured_headers,
            {"X-Agent-ID": "team-beacon", "X-Round-ID": "round-123"},
        )

    def test_evidence_fallback_passes_proxy_headers_to_openai_client(self) -> None:
        module = _load_crewai_agent_module()
        captured_headers: dict[str, str] = {}

        with self._install_fake_openai(
            "ANSWER: Eve, Bob, Alice, Charlie, David",
            captured_headers,
        ):
            answer = module._derive_answer_from_evidence(
                llm_host=FAKE_LLM_HOST,
                llm_api_key=FAKE_LLM_API_KEY,
                model_name="gpt-oss-20b",
                challenge_description="Find the order.",
                challenge_rules="Return five names.",
                evidence_lines=["Eve is first."],
                agent_id="team-beacon",
                usage_scope="round-123",
            )

        self.assertEqual(answer, "Eve, Bob, Alice, Charlie, David")
        self.assertEqual(
            captured_headers,
            {"X-Agent-ID": "team-beacon", "X-Round-ID": "round-123"},
        )

    def test_direct_text_fallback_passes_proxy_headers_to_openai_client(self) -> None:
        module = _load_crewai_agent_module()
        captured_headers: dict[str, str] = {}

        with self._install_fake_openai(
            "ANSWER: Eve, Bob, Alice, Charlie, David",
            captured_headers,
        ):
            answer = module._direct_text_fallback_answer(
                llm_host=FAKE_LLM_HOST,
                llm_api_key=FAKE_LLM_API_KEY,
                model_name="gpt-oss-20b",
                challenge_description="Find the order.",
                challenge_rules="Return five names.",
                clues=["Eve is first."],
                agent_id="team-beacon",
                usage_scope="round-123",
            )

        self.assertEqual(answer, "Eve, Bob, Alice, Charlie, David")
        self.assertEqual(
            captured_headers,
            {"X-Agent-ID": "team-beacon", "X-Round-ID": "round-123"},
        )

    def test_image_planner_candidates_exclude_image_output_models(self) -> None:
        module = _load_crewai_agent_module()

        candidates = module._build_planner_candidate_models(
            modality="image",
            strategy_model="gemini-3.1-flash-lite-image",
            selector_model="",
            ranked_models=[
                "gemini-3.1-flash-lite-image",
                "qwen3.7-flash",
            ],
            available_models=[
                "gemini-3.1-flash-lite-image",
                "gpt-oss-20b",
                "qwen3.7-flash",
            ],
        )

        self.assertEqual(candidates, ["qwen3.7-flash", "gpt-oss-20b"])

    def test_image_planner_uses_task_context_and_reserves_tool_time(self) -> None:
        module = _load_crewai_agent_module()

        context = module._build_context(
            challenge_type="image",
            description="Edit the supplied portrait.",
            rules="Preserve the background.",
            max_time_s=120,
            available_models=["gpt-oss-20b"],
            image_url="data:image/png;base64,c291cmNl",
        )

        planner_context = module._planner_selection_context(context, "image")
        self.assertEqual(planner_context.challenge_type, "tool-orchestration")
        self.assertIsNone(planner_context.image_url)
        self.assertEqual(
            module._crewai_execution_limits("image", 120),
            (2, 45),
        )
        self.assertEqual(
            module._crewai_execution_limits("text", 120),
            (5, 120),
        )
        self.assertFalse(module._native_crewai_tools_enabled("image"))
        self.assertTrue(module._native_crewai_tools_enabled("text"))
        self.assertEqual(module._planner_failure_fallback("image"), "")
        self.assertIsNone(module._planner_failure_fallback("text"))


if __name__ == "__main__":
    unittest.main()
