from __future__ import annotations

import asyncio
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arena_clients import ArenaAPIError
from arena_clients.session_monitor import (
    TERMINAL_SESSION_STATUSES,
    get_session_stop_reason,
    monitor_session,
)


class FakeHttpClient:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.calls = 0

    def get_session(self, agent_id: str) -> dict:
        self.calls += 1
        if self.responses:
            response = self.responses.pop(0)
        else:
            response = {"agent_id": agent_id, "status": "running"}
        if isinstance(response, BaseException):
            raise response
        return response


class SessionMonitorReasonTests(unittest.TestCase):
    def test_active_session_keeps_running(self) -> None:
        client = FakeHttpClient([{"agent_id": "agent-1", "status": "running"}])
        self.assertIsNone(get_session_stop_reason(client, "agent-1"))

    def test_404_triggers_stop(self) -> None:
        client = FakeHttpClient([ArenaAPIError(404, "Session not found")])
        reason = get_session_stop_reason(client, "agent-1")
        self.assertIsNotNone(reason)
        self.assertEqual(reason.status, "missing")
        self.assertEqual(reason.exit_code, 0)

    def test_missing_session_triggers_stop(self) -> None:
        client = FakeHttpClient([{}])
        reason = get_session_stop_reason(client, "agent-1")
        self.assertIsNotNone(reason)
        self.assertEqual(reason.status, "missing")

    def test_terminal_statuses_trigger_stop(self) -> None:
        for status in TERMINAL_SESSION_STATUSES:
            with self.subTest(status=status):
                client = FakeHttpClient([{"agent_id": "agent-1", "status": status}])
                reason = get_session_stop_reason(client, "agent-1")
                self.assertIsNotNone(reason)
                self.assertEqual(reason.status, status)
                self.assertEqual(reason.exit_code, 0)

    def test_transient_connection_errors_do_not_stop(self) -> None:
        client = FakeHttpClient([TimeoutError("temporary timeout")])
        self.assertIsNone(get_session_stop_reason(client, "agent-1"))


class SessionMonitorAsyncTests(unittest.TestCase):
    def test_monitor_cancels_work_on_terminal_status(self) -> None:
        async def scenario() -> bool:
            client = FakeHttpClient(
                [
                    {"agent_id": "agent-1", "status": "running"},
                    {"agent_id": "agent-1", "status": "disconnected"},
                ]
            )
            async with monitor_session(client, "agent-1", poll_interval_s=0.01) as monitor:
                try:
                    await asyncio.sleep(1.0)
                except asyncio.CancelledError:
                    self.assertIsNotNone(monitor.stop_reason)
                    raise
            return monitor.stop_reason is not None

        self.assertTrue(asyncio.run(scenario()))

    def test_monitor_ignores_transient_errors(self) -> None:
        async def scenario() -> int:
            client = FakeHttpClient(
                [
                    TimeoutError("temporary timeout"),
                    {"agent_id": "agent-1", "status": "running"},
                ]
            )
            async with monitor_session(client, "agent-1", poll_interval_s=0.01):
                await asyncio.sleep(0.03)
            return client.calls

        self.assertGreaterEqual(asyncio.run(scenario()), 2)


if __name__ == "__main__":
    unittest.main()
