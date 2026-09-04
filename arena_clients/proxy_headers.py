"""Helpers for attributing LLM proxy usage to an agent and battle."""

from __future__ import annotations

import os


def resolve_usage_scope(explicit_scope: str | None = None) -> str | None:
    """Return a non-empty usage scope from an explicit value or ARENA_USAGE_SCOPE."""
    scope_value = str(explicit_scope or os.getenv("ARENA_USAGE_SCOPE") or "").strip()
    return scope_value or None


def build_proxy_headers(
    agent_id: str | None = None,
    usage_scope: str | None = None,
) -> dict[str, str]:
    """Build headers used by the LLM proxy to attribute model usage."""
    headers: dict[str, str] = {}
    resolved_agent_id = str(agent_id or os.getenv("AGENT_ID") or "").strip()
    resolved_scope = resolve_usage_scope(usage_scope)
    if resolved_agent_id:
        headers["X-Agent-ID"] = resolved_agent_id
    if resolved_scope:
        headers["X-Round-ID"] = resolved_scope
    return headers
