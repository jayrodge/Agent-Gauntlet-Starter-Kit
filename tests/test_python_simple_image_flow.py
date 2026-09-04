from __future__ import annotations

import asyncio
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arena_clients.mcp_client import ImageChallengeInfo
from base_strategy import BaseStrategy, ChallengeContext

SIMPLE_AGENT = ROOT / "examples" / "python_simple" / "agent.py"


def _load_simple_agent_module():
    strategy_module = types.ModuleType("my_strategy")
    strategy_module.MyStrategy = type("MyStrategy", (BaseStrategy,), {})
    with mock.patch.dict(sys.modules, {"my_strategy": strategy_module}):
        spec = importlib.util.spec_from_file_location(
            "python_simple_agent_under_test",
            SIMPLE_AGENT,
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


class PythonSimpleImageFlowTests(unittest.TestCase):
    def test_analyze_first_edit_plan_continues_to_output_tool(self) -> None:
        module = _load_simple_agent_module()

        class AnalyzeFirstStrategy:
            def plan_image_tool(self, _ctx, _available_tools):
                return "image_analyze"

        context = ChallengeContext(
            challenge_type="image_edit",
            image_url="input://challenge",
        )

        self.assertEqual(
            module._plan_image_tool_sequence(
                AnalyzeFirstStrategy(),
                context,
                ["image_edit", "image_generate", "image_analyze"],
            ),
            ["image_analyze", "image_edit"],
        )


class _FakeSessionMonitor:
    def __init__(self) -> None:
        self.stop_calls = 0

    def start(self) -> "_FakeSessionMonitor":
        return self

    async def stop(self) -> None:
        self.stop_calls += 1


class _FakeHttpClient:
    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs
        self.submit_calls: list[dict] = []

    def register(self, agent_id: str, agent_name: str):
        del agent_name
        return types.SimpleNamespace(session_id="sess-1", status="registered", agent_id=agent_id)

    def fetch_usage_scope(self) -> str:
        return "battle-scope"

    def broadcast_thought(self, *args, **kwargs) -> None:
        del args, kwargs

    def update_status(self, *args, **kwargs) -> None:
        del args, kwargs

    def get_session(self, agent_id: str) -> dict:
        return {"agent_id": agent_id, "status": "running"}

    def submit(self, agent_id, answer=None, client_metrics=None, challenge_type=None, **kwargs):
        del kwargs
        self.submit_calls.append(
            {
                "agent_id": agent_id,
                "answer": answer,
                "challenge_type": challenge_type,
                "client_metrics": client_metrics,
            }
        )
        return {"accepted": True}


class _FakeMcpClient:
    def __init__(
        self,
        *,
        input_image_uri: str,
        tools: list[str] | None = None,
    ) -> None:
        self.input_image_uri = input_image_uri
        self.tools = tools or ["image_edit", "arena.image.get_challenge"]
        self.submit_image_calls: list[dict] = []

    async def __aenter__(self) -> "_FakeMcpClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        del exc_type, exc, tb
        return False

    async def list_tools(self) -> list[str]:
        return list(self.tools)

    async def list_tool_defs(self) -> list:
        return []

    async def get_image_challenge(self, agent_id: str) -> ImageChallengeInfo:
        del agent_id
        return ImageChallengeInfo(
            challenge_type="image",
            challenge_id="challenge-1",
            puzzle_id="puzzle-1",
            difficulty="easy",
            description="edit the image",
            prompt="make it blue",
            reference_notes="",
            max_time_s=60,
            input_image_uri=self.input_image_uri,
            time_remaining_s=60.0,
        )

    async def call_tool(self, name: str, arguments: dict) -> dict:
        del name, arguments
        raise ConnectionError("502 Bad Gateway")

    async def submit_image(self, **kwargs) -> dict:
        self.submit_image_calls.append(dict(kwargs))
        raise AssertionError("MCP submit must not run after a dead session")


class PythonSimpleImageMcpFallbackTests(unittest.TestCase):
    def _run_image_main(
        self,
        module,
        *,
        input_image_uri: str,
        tools: list[str] | None = None,
    ) -> tuple[_FakeHttpClient, _FakeSessionMonitor]:
        http_client = _FakeHttpClient()
        monitor = _FakeSessionMonitor()
        mcp_client = _FakeMcpClient(input_image_uri=input_image_uri, tools=tools)

        class _McpFactory:
            def __init__(self, *args, **kwargs) -> None:
                del args, kwargs

            async def __aenter__(self):
                return mcp_client

            async def __aexit__(self, exc_type, exc, tb) -> bool:
                return await mcp_client.__aexit__(exc_type, exc, tb)

            @staticmethod
            def detect_modality(discovered_tools):
                del discovered_tools
                return "image"

        with (
            mock.patch.object(module, "ensure_connected"),
            mock.patch.object(module, "get_api_base", return_value="http://127.0.0.1:8000"),
            mock.patch.object(module, "get_mcp_url", return_value="http://127.0.0.1:5001"),
            mock.patch.object(module, "get_proxy_host", return_value="http://127.0.0.1:4001"),
            mock.patch.object(module, "get_arena_api_key", return_value="test-key"),
            mock.patch.object(module, "HttpArenaClient", return_value=http_client),
            mock.patch.object(module, "McpArenaClient", _McpFactory),
            mock.patch.object(module, "monitor_session", return_value=monitor),
            mock.patch.object(
                module,
                "fetch_available_models",
                return_value=["gemini-3.1-flash-image"],
            ),
            mock.patch.object(
                module,
                "require_explicit_model",
                return_value="gemini-3.1-flash-image",
            ),
            mock.patch.object(
                module,
                "_fetch_proxy_usage",
                return_value={
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            ),
            mock.patch.object(module, "HAS_OPENAI", True),
            mock.patch.dict(
                os.environ,
                {"AGENT_ID": "alpha", "AGENT_NAME": "Alpha"},
                clear=False,
            ),
        ):
            asyncio.run(module.main())
        return http_client, monitor

    def test_image_edit_failure_submits_input_uri_over_rest(self) -> None:
        module = _load_simple_agent_module()
        input_uri = "data:image/png;base64,aW5wdXQ="
        http_client, monitor = self._run_image_main(module, input_image_uri=input_uri)

        self.assertEqual(len(http_client.submit_calls), 1)
        submit = http_client.submit_calls[0]
        self.assertEqual(submit["answer"], input_uri)
        self.assertEqual(submit["challenge_type"], "image")
        self.assertNotEqual(submit["answer"], module.BLANK_PNG_DATA_URI)
        self.assertGreaterEqual(monitor.stop_calls, 1)

    def test_image_tool_failure_without_input_uri_raises(self) -> None:
        module = _load_simple_agent_module()
        with self.assertRaises(RuntimeError) as raised:
            self._run_image_main(
                module,
                input_image_uri="",
                tools=["image_generate", "arena.image.get_challenge"],
            )
        self.assertIn("refusing to invent image bytes", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
