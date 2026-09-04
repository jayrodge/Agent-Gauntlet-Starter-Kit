from __future__ import annotations

import asyncio
import importlib.util
import io
from pathlib import Path
import sys
import time
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_AGENT = ROOT / "examples" / "python_reference" / "agent.py"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_reference_agent_module():
    strategy_module = types.ModuleType("my_strategy")

    class MyStrategy:
        pass

    strategy_module.MyStrategy = MyStrategy
    with mock.patch.dict(sys.modules, {"my_strategy": strategy_module}):
        spec = importlib.util.spec_from_file_location(
            "python_reference_agent_under_test",
            REFERENCE_AGENT,
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


class PythonReferenceTimeoutTests(unittest.TestCase):
    def test_solve_timeout_reserves_submission_time(self) -> None:
        module = _load_reference_agent_module()

        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(module._derive_llm_solve_timeout_s(120), 90.0)

        with mock.patch.dict("os.environ", {"LLM_TIMEOUT_S": "45"}, clear=True):
            self.assertEqual(module._derive_llm_solve_timeout_s(120), 45.0)

        self.assertFalse(module._has_final_repair_budget(30.0))
        self.assertTrue(module._has_final_repair_budget(60.0))

    def test_sse_iterator_does_not_read_after_deadline(self) -> None:
        module = _load_reference_agent_module()

        class FailIfRead:
            def readline(self):
                raise AssertionError("expired stream must not be read")

        events = list(
            module._iter_sse_events(
                FailIfRead(),
                deadline=module.time.monotonic() - 1.0,
            )
        )
        self.assertEqual(events, [])

    def test_image_tool_call_returns_fallback_error_at_work_deadline(self) -> None:
        module = _load_reference_agent_module()

        class HangingMcpClient:
            async def call_tool(self, name, arguments):
                await asyncio.Event().wait()

        result = asyncio.run(
            module._call_image_tool_before_deadline(
                HangingMcpClient(),
                "image_edit",
                {"prompt": "test"},
                deadline=time.monotonic() + 0.01,
            )
        )

        self.assertEqual(result["error"], "image_edit timed out before submission reserve")

    def test_stream_consumer_stops_at_wall_clock_deadline(self) -> None:
        module = _load_reference_agent_module()
        class DelayedBytesIO(io.BytesIO):
            def __init__(self, payload: bytes):
                super().__init__(payload)
                self.readline_calls = 0

            def readline(self, *args, **kwargs):
                self.readline_calls += 1
                if self.readline_calls == 3:
                    time.sleep(0.03)
                return super().readline(*args, **kwargs)

        stream = DelayedBytesIO(
            b'data: {"choices":[{"delta":{"content":"ANSWER: alpha"}}]}\n\n'
            b'data: {"choices":[{"delta":{"content":" ignored"}}]}\n\n'
            b'data: [DONE]\n\n'
        )
        messages: list[str] = []

        async def broadcast(message: str) -> None:
            messages.append(message)

        start_time = module.time.monotonic()
        answer, _reasoning, _usage, _ttft_ms, _streamed = asyncio.run(
            module._consume_sse_stream(
                stream,
                broadcast,
                start_time=start_time,
                deadline=start_time + 0.01,
            )
        )

        self.assertEqual(answer, "alpha")
        self.assertNotIn("ignored", " ".join(messages))


if __name__ == "__main__":
    unittest.main()
