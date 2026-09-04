"""Participant session stop monitoring for Agent Gauntlet agents."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from types import TracebackType
from typing import Any

from .http_client import ArenaAPIError, HttpArenaClient


DEFAULT_SESSION_POLL_INTERVAL_S = 2.0
TERMINAL_SESSION_STATUSES = {"stopped", "failed", "error", "disconnected"}


@dataclass(frozen=True)
class SessionStopReason:
    """Server-side reason that should stop a participant process."""

    agent_id: str
    status: str
    message: str
    exit_code: int = 0


def resolve_session_poll_interval(default: float = DEFAULT_SESSION_POLL_INTERVAL_S) -> float:
    """Resolve the session polling interval from the environment."""
    raw_value = os.getenv("ARENA_SESSION_POLL_INTERVAL_S")
    if raw_value is None:
        return default
    try:
        interval = float(raw_value)
    except (TypeError, ValueError):
        return default
    if interval <= 0:
        return default
    return interval


def get_session_stop_reason(
    http_client: HttpArenaClient,
    agent_id: str,
) -> SessionStopReason | None:
    """Return a stop reason when the server session is gone or terminal.

    Transient API/network errors are intentionally non-fatal. A definitive
    ``404`` means the organizer disconnected the session or the server no
    longer recognizes it.
    """
    try:
        session = http_client.get_session(agent_id)
    except ArenaAPIError as exc:
        if exc.status_code == 404:
            return SessionStopReason(
                agent_id=agent_id,
                status="missing",
                message=(
                    f"Session for agent '{agent_id}' is missing on the server; "
                    "exiting without broadcasting."
                ),
            )
        return None
    except Exception:
        return None

    if not isinstance(session, dict) or not session:
        return SessionStopReason(
            agent_id=agent_id,
            status="missing",
            message=(
                f"Session for agent '{agent_id}' is missing on the server; "
                "exiting without broadcasting."
            ),
        )

    status = str(session.get("status") or "").strip().lower()
    if status in TERMINAL_SESSION_STATUSES:
        return SessionStopReason(
            agent_id=agent_id,
            status=status,
            message=(
                f"Session for agent '{agent_id}' is {status}; "
                "exiting without broadcasting."
            ),
        )
    return None


class SessionMonitor:
    """Async context manager that exits agent work on server-side stop."""

    def __init__(
        self,
        http_client: HttpArenaClient,
        agent_id: str,
        *,
        poll_interval_s: float | None = None,
    ) -> None:
        self.http_client = http_client
        self.agent_id = agent_id
        self.poll_interval_s = (
            poll_interval_s
            if poll_interval_s is not None
            else resolve_session_poll_interval()
        )
        self.stop_reason: SessionStopReason | None = None
        self._owner_task: asyncio.Task[Any] | None = None
        self._monitor_task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()

    def start(self) -> "SessionMonitor":
        """Start monitoring the current asyncio task."""
        if self._monitor_task is not None:
            return self
        self._owner_task = asyncio.current_task()
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        return self

    async def stop(self) -> None:
        """Stop monitoring after normal agent completion."""
        self._stop_event.set()
        if self._monitor_task is not None:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

    async def __aenter__(self) -> "SessionMonitor":
        self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        await self.stop()

        if exc_type is asyncio.CancelledError and self.stop_reason is not None:
            return True
        return False

    async def _monitor_loop(self) -> None:
        while not self._stop_event.is_set():
            reason = await asyncio.to_thread(
                get_session_stop_reason,
                self.http_client,
                self.agent_id,
            )
            if reason is not None:
                self.stop_reason = reason
                print(f"[agent] {reason.message}", flush=True)
                if self._owner_task is not None:
                    self._owner_task.cancel()
                return
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self.poll_interval_s,
                )
            except asyncio.TimeoutError:
                continue


def monitor_session(
    http_client: HttpArenaClient,
    agent_id: str,
    *,
    poll_interval_s: float | None = None,
) -> SessionMonitor:
    """Create a session monitor context manager."""
    return SessionMonitor(
        http_client=http_client,
        agent_id=agent_id,
        poll_interval_s=poll_interval_s,
    )
