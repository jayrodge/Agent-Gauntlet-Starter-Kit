"""Shared environment resolution helpers for starter-kit clients."""

from __future__ import annotations

import os
from functools import lru_cache
import ipaddress
from urllib.parse import urlparse

import requests

_DEFAULT_API_BASE = "http://localhost:8000"
_DEFAULT_MCP_URL = "http://localhost:5001"
_DEFAULT_PROXY_HOST = "http://localhost:4001"


def _is_loopback_host(host: str) -> bool:
    normalized = str(host or "").strip().strip("[]").lower()
    if normalized == "localhost" or normalized.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _require_secure_remote_url(value: str) -> str:
    parsed = urlparse(value)
    scheme = parsed.scheme.lower()
    host = str(parsed.hostname or "").strip()
    if scheme == "http" and not _is_loopback_host(host):
        raise ValueError(
            f"Remote Agent Gauntlet URLs must use HTTPS; received {value!r}"
        )
    return value


def _read_env(name: str) -> str:
    return (os.getenv(name) or "").strip()


def _normalize_server(server: str) -> tuple[str, str, str]:
    raw = server.strip()
    if not raw:
        return "http", "", ""

    if "://" in raw:
        parsed = urlparse(raw)
        scheme = parsed.scheme or "https"
        host = parsed.hostname or parsed.netloc or parsed.path
        authority = parsed.netloc or host
    else:
        parsed = urlparse(f"//{raw}")
        host = parsed.hostname or raw
        authority = parsed.netloc or host
        scheme = "http" if _is_loopback_host(str(host or "")) else "https"

    if scheme == "http" and not _is_loopback_host(str(host or "")):
        raise ValueError(
            f"Remote Agent Gauntlet URLs must use HTTPS; received {server!r}"
        )

    return (
        scheme,
        str(host or "").strip().strip("/"),
        str(authority or "").strip().strip("/"),
    )


def _server_origin() -> str:
    server = _read_env("ARENA_SERVER")
    if not server:
        return ""
    scheme, host, authority = _normalize_server(server)
    target = authority if scheme == "https" else host
    return f"{scheme}://{target}" if target else ""


def _expand_server_reference(value: str) -> str:
    resolved = value.strip()
    if not resolved:
        return ""

    origin = _server_origin()
    if not origin:
        return resolved

    for token in ("${ARENA_SERVER}", "$ARENA_SERVER", "ARENA_SERVER"):
        if resolved == token:
            return origin
        if resolved.startswith(f"{token}/"):
            return f"{origin}{resolved[len(token):]}"

    return resolved


def _resolve_service_url(
    explicit: str | None,
    *,
    env_name: str,
    port: int,
    fallback: str,
    reverse_proxy_path: str = "",
) -> str:
    resolved = _expand_server_reference(explicit or "")
    if resolved:
        return _require_secure_remote_url(resolved.rstrip("/"))

    resolved = _expand_server_reference(_read_env(env_name))
    if resolved:
        return _require_secure_remote_url(resolved.rstrip("/"))

    server = _read_env("ARENA_SERVER")
    if server:
        scheme, host, authority = _normalize_server(server)
        if scheme == "https" and authority:
            origin = f"{scheme}://{authority}"
            if reverse_proxy_path:
                return f"{origin}/{reverse_proxy_path.strip('/')}"
            return origin
        if host:
            return f"{scheme}://{host}:{port}"

    return fallback


def get_api_base(explicit: str | None = None) -> str:
    """Resolve the Agent Gauntlet Platform REST API base URL."""
    return _resolve_service_url(
        explicit,
        env_name="ARENA_API_BASE",
        port=8000,
        fallback=_DEFAULT_API_BASE,
    )


def get_mcp_url(explicit: str | None = None) -> str:
    """Resolve the Agent Gauntlet Platform MCP URL."""
    return _resolve_service_url(
        explicit,
        env_name="ARENA_MCP_URL",
        port=5001,
        fallback=_DEFAULT_MCP_URL,
    )


def get_proxy_host(explicit: str | None = None) -> str:
    """Resolve the Agent Gauntlet Platform LLM proxy base URL."""
    return _resolve_service_url(
        explicit,
        env_name="LLM_PROXY_HOST",
        port=4001,
        fallback=_DEFAULT_PROXY_HOST,
        reverse_proxy_path="proxy",
    )


def get_arena_api_key(explicit: str | None = None) -> str:
    """Resolve the competitor Agent Gauntlet API key."""
    resolved = (explicit or "").strip()
    if resolved:
        return resolved
    return _read_env("ARENA_API_KEY")


@lru_cache(maxsize=1)
def ensure_connected(timeout_s: float = 3.0) -> None:
    """Fail fast when the starter kit is missing Agent Gauntlet connectivity config."""
    if not (_read_env("ARENA_SERVER") or _read_env("ARENA_API_BASE")):
        raise SystemExit(
            "Agent Gauntlet server is not configured. Set ARENA_SERVER to the organizer-provided "
            "HTTPS origin in your .env file."
        )

    api_key = get_arena_api_key()
    if not api_key:
        raise SystemExit(
            "ARENA_API_KEY is missing. Set it in your .env file before running "
            "the agent."
        )

    api_base = get_api_base()
    url = f"{api_base.rstrip('/')}/api/keys/validate"

    try:
        response = requests.get(
            url,
            headers={
                "Accept": "application/json",
                "X-Arena-API-Key": api_key,
            },
            timeout=timeout_s,
        )
        if response.status_code >= 400:
            raise SystemExit(
                f"Could not verify ARENA_API_KEY with the Agent Gauntlet Platform API at {api_base} "
                f"(HTTP {response.status_code}). Check ARENA_SERVER and try again."
            )
        payload = response.json()
    except requests.RequestException as exc:
        raise SystemExit(
            f"Could not reach the Agent Gauntlet Platform API at {api_base}. Check ARENA_SERVER "
            "and your network connection."
        ) from exc
    except ValueError as exc:
        raise SystemExit(
            f"The Agent Gauntlet Platform API at {api_base} returned an invalid validation "
            "response. Check that the server is running the expected build."
        ) from exc

    if not isinstance(payload, dict) or not bool(payload.get("valid")):
        raise SystemExit(
            "ARENA_API_KEY is missing or invalid. Set it in your .env file "
            "before running the agent."
        )
