"""HTTP client for Agent Gauntlet Platform REST API coordination.

This client handles session management, thought broadcasting, and submission
via the REST API. It's framework-agnostic and can be used from any agent.

The API base URL is derived from `ARENA_SERVER` by default, and the competitor
key is sent as the `X-Arena-API-Key` header.

Example:
    client = HttpArenaClient()  # reads ARENA_SERVER and ARENA_API_KEY
    # or, explicitly:
    client = HttpArenaClient(
        api_base="https://arena.example.com",
        api_key="<battle-key>",
    )

    # Register session
    session = client.register("my-agent", "My Agent Name")

    # Broadcast thoughts
    client.broadcast_thought("my-agent", "Analyzing the problem...")

    # Submit answer
    result = client.submit("my-agent", "E, A, B, C||D", {"model": "my-model"})
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import time
from typing import Any

import requests

from .config import get_api_base, get_arena_api_key


@dataclass
class SessionInfo:
    """Session information returned from registration."""

    session_id: str
    agent_id: str
    agent_name: str
    status: str
    started_at: float


@dataclass
class SubmitResult:
    """Result of submitting an answer."""

    accepted: bool
    agent_id: str
    answer: str
    score: dict[str, Any] | None
    status: str


class HttpArenaClient:
    """HTTP client for Agent Gauntlet Platform REST API coordination.

    This client uses requests for predictable POST behavior across local and
    remote rehearsal machines.
    """

    def __init__(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        """Initialize the client.

        Args:
            api_base: Base URL for the Agent Gauntlet Platform API (default: ARENA_API_BASE or
                derived from ARENA_SERVER)
            api_key: API key for authentication (default: ARENA_API_KEY env var)
            timeout: Request timeout in seconds
        """
        self.api_base = get_api_base(api_base)
        self.api_key = get_arena_api_key(api_key)
        self.timeout = timeout
        self._usage_scope_cache: str | None = None

    def _request(
        self,
        method: str,
        path: str,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make an HTTP request to the API."""
        url = f"{self.api_base}{path}"

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-Arena-API-Key"] = self.api_key

        try:
            response = requests.request(
                method,
                url,
                headers=headers,
                json=data if data is not None else None,
                timeout=self.timeout,
            )
            if response.status_code >= 400:
                raise ArenaAPIError(response.status_code, response.text)
            return response.json()
        except requests.RequestException as e:
            raise ArenaConnectionError(str(e)) from e

    def health(self) -> dict[str, Any]:
        """Check API health."""
        return self._request("GET", "/api/health")

    def register(
        self,
        agent_id: str,
        agent_name: str | None = None,
    ) -> SessionInfo:
        """Register an agent session.

        Args:
            agent_id: Unique identifier for the agent
            agent_name: Display name for the agent (optional)

        Returns:
            SessionInfo with session details
        """
        data = {"agent_id": agent_id}
        if agent_name:
            data["agent_name"] = agent_name

        raw_timeout_s = os.getenv("ARENA_REGISTRATION_TIMEOUT_S", "600")
        try:
            timeout_s = float(raw_timeout_s)
        except ValueError:
            timeout_s = 600.0
        if timeout_s <= 0:
            timeout_s = 600.0
        deadline = time.monotonic() + timeout_s
        delay_s = 1.0
        while True:
            try:
                result = self._request("POST", "/api/session/register", data)
                break
            except ArenaAPIError as e:
                if e.status_code == 409:
                    if e.code == "registration_too_late":
                        raise ArenaConnectionError(
                            "Registration is closed for the current round. "
                            "Register during the next lobby."
                        ) from e
                    remaining_s = deadline - time.monotonic()
                    if remaining_s <= 0:
                        timeout_label = f"{timeout_s:g}"
                        raise ArenaConnectionError(
                            "Registration did not open within "
                            f"{timeout_label} seconds."
                        ) from e
                    print("   Lobby is not open yet. Waiting for organizer...", flush=True)
                    time.sleep(min(delay_s, remaining_s))
                    delay_s = min(delay_s * 2.0, 10.0)
                else:
                    raise

        return SessionInfo(
            session_id=result["session_id"],
            agent_id=result["agent_id"],
            agent_name=result["agent_name"],
            status=result["status"],
            started_at=result["started_at"],
        )

    def update_status(
        self,
        agent_id: str,
        status: str,
        client_metrics: dict[str, Any] | None = None,
    ) -> bool:
        """Update agent status shown in Agent Gauntlet.

        Args:
            agent_id: The agent's ID
            status: Session status (ready/running/submitted/failed/etc.)
            client_metrics: Optional live metrics (tokens, elapsed time, model)
        """
        payload: dict[str, Any] = {"agent_id": agent_id, "status": status}
        if client_metrics:
            payload["client_metrics"] = client_metrics
        result = self._request(
            "POST",
            "/api/status",
            payload,
        )
        return result.get("updated", False)

    def broadcast_thought(self, agent_id: str, thought: str) -> bool:
        """Broadcast a thought to Agent Gauntlet.

        Args:
            agent_id: The agent's ID
            thought: The thought to broadcast

        Returns:
            True if accepted
        """
        result = self._request(
            "POST",
            "/api/thought",
            {
                "agent_id": agent_id,
                "thought": thought,
            },
        )
        return result.get("accepted", False)

    def save_draft(
        self,
        agent_id: str,
        draft: str,
        rationale: str | None = None,
    ) -> bool:
        """Save a draft answer.

        Args:
            agent_id: The agent's ID
            draft: The draft answer
            rationale: Optional rationale for the draft

        Returns:
            True if saved
        """
        data = {"agent_id": agent_id, "draft": draft}
        if rationale:
            data["rationale"] = rationale

        result = self._request("POST", "/api/draft", data)
        return result.get("saved", False)

    def submit(
        self,
        agent_id: str,
        answer: str,
        client_metrics: dict[str, Any] | None = None,
        challenge_type: str | None = "text",
    ) -> SubmitResult:
        """Submit a final answer.

        Args:
            agent_id: The agent's ID
            answer: The final answer
            client_metrics: Optional metrics (model_name, tokens, etc.)
            challenge_type: "text" or "image". ``None`` omits the field so
                the server uses its authoritative assignment.

        Returns:
            SubmitResult with score and status
        """
        payload: dict[str, Any] = {
            "agent_id": agent_id,
            "answer": answer,
            "client_metrics": client_metrics or {},
        }
        if challenge_type is not None:
            payload["challenge_type"] = challenge_type
        result = self._request(
            "POST",
            "/api/submit",
            payload,
        )

        return SubmitResult(
            accepted=result["accepted"],
            agent_id=result["agent_id"],
            answer=result["answer"],
            score=result.get("score"),
            status=result["status"],
        )

    def get_session(self, agent_id: str) -> dict[str, Any]:
        """Get current session state.

        Args:
            agent_id: The agent's ID

        Returns:
            Session state dictionary
        """
        return self._request("GET", f"/api/session/{agent_id}")

    def get_leaderboard(self) -> list[dict[str, Any]]:
        """Get the current leaderboard.

        Returns:
            List of leaderboard entries
        """
        return self._request("GET", "/api/leaderboard")

    def get_competition(self) -> dict[str, Any]:
        """Get current competition phase/state."""
        return self._request("GET", "/api/competition")

    def fetch_usage_scope(self) -> str | None:
        """Fetch and cache the active battle usage scope when available."""
        if self._usage_scope_cache:
            return self._usage_scope_cache
        try:
            competition = self.get_competition()
        except Exception:
            return None
        scope = str(competition.get("usage_scope") or "").strip()
        if scope:
            self._usage_scope_cache = scope
            return scope
        return None


class ArenaAPIError(Exception):
    """Error from the Agent Gauntlet Platform API."""

    def __init__(self, status_code: int, message: str):
        code: str | None = None
        try:
            payload = json.loads(message)
        except (TypeError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, dict):
            detail = payload.get("detail")
            if isinstance(detail, dict):
                raw_code = detail.get("code")
                if isinstance(raw_code, str) and raw_code.strip():
                    code = raw_code.strip()

        self.status_code = status_code
        self.message = message
        self.code = code
        super().__init__(f"API error {status_code}: {message}")


class ArenaConnectionError(Exception):
    """Connection error to the Agent Gauntlet Platform API."""

    pass
