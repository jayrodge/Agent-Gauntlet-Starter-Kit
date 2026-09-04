"""Readiness doctor for Agent Gauntlet connectivity.

Extends `ensure_connected()` with health, MCP tools, proxy models,
attributed inference, and scoped usage checks.

The default checks are read-only and safe to run against live hosts in any
competition phase. `--full` additionally registers, pulls the challenge, and
submits, so it refuses to run anywhere a submission could count.

`--certify` is the published play-rehearsal contract (advisory when run inside
a competitor image). It implies the full play path and still fails closed off
Practice via `check_full_gate()`.

Frozen certify interface
------------------------
    python -m arena_clients.doctor --certify --json

- `--certify` rehearses a real play: wait for lobby, register as `AGENT_ID`,
  wait for GO, get the challenge, submit, then probe exact-retry and conflict.
- The sandbox must inject `ARENA_USAGE_SCOPE`. Certify mode never invents a
  `doctor-<uuid>` scope. If the env var is unset, the doctor reads
  `usage_scope` from `/api/competition` instead.
- `--json` writes the checklist object to stdout as pretty JSON. Optional
  `--output PATH` also writes that same JSON to a file for the receipt.
- Exit 0 only if every checklist item is PASS; otherwise exit 1.
- Checklist keys, each PASS or FAIL:
  `registered_in_lobby`, `waited_for_go`, `in_frozen_roster`,
  `answer_accepted`, `scored`, `retry_is_canonical`,
  `conflicting_answer_rejected`, `attribution_scope_exact`.

Usage:
    python -m arena_clients.doctor
    python -m arena_clients.doctor --full
    python -m arena_clients.doctor --certify --json
    python -m arena_clients.doctor --certify --json --output receipt.json
    python -m arena_clients.doctor --certify --modality image --json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import requests
from dotenv import load_dotenv

from .config import (
    ensure_connected,
    get_api_base,
    get_arena_api_key,
    get_mcp_url,
    get_proxy_host,
)
from .http_client import (
    ArenaAPIError,
    ArenaConnectionError,
    HttpArenaClient,
    SubmitResult,
)
from .mcp_client import McpArenaClient
from .proxy_headers import build_proxy_headers
from model_selector import fetch_available_models


CheckFn = Callable[[], None]

# The practice deployment identifies itself here; the operator platform reports
# service "agent-arena-api" and has no practice_gate_mode field.
PRACTICE_SERVICE_NAMES = ("practice-arena-api",)

FULL_REFUSAL_FIX = (
    "Point ARENA_SERVER at the Practice deployment, or ask the organizer to "
    "open a warmup battle. Run plain `python -m arena_clients.doctor` for the "
    "read-only checks."
)

CERTIFY_CHECKLIST_KEYS = (
    "registered_in_lobby",
    "waited_for_go",
    "in_frozen_roster",
    "answer_accepted",
    "scored",
    "retry_is_canonical",
    "conflicting_answer_rejected",
    "attribution_scope_exact",
)
CERTIFY_PASS = "PASS"
CERTIFY_FAIL = "FAIL"
CERTIFY_ANSWER = "doctor --certify play rehearsal"
CERTIFY_CONFLICT_ANSWER = "doctor --certify conflicting answer"
CERTIFY_IMAGE_THOUGHT = "Doctor image certification smoke test"
CERTIFY_MODALITIES = ("text", "image")

# When --json is set, human-readable progress goes here so stdout stays JSON.
_HUMAN_STREAM = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_env() -> None:
    load_dotenv(_repo_root() / ".env")


def _human_stream():
    return _HUMAN_STREAM if _HUMAN_STREAM is not None else sys.stdout


def _say(message: str) -> None:
    print(message, file=_human_stream())


def _print_ok(label: str, detail: str = "") -> None:
    suffix = f" — {detail}" if detail else ""
    _say(f"OK  {label}{suffix}")


def _fail(label: str, reason: str, fix: str) -> None:
    _say(f"FAIL  {label}")
    _say(f"      {reason}")
    _say(f"      Fix: {fix}")
    raise SystemExit(1)


def empty_certify_checklist() -> dict[str, str]:
    """Return the frozen checklist with every item set to FAIL."""
    return {key: CERTIFY_FAIL for key in CERTIFY_CHECKLIST_KEYS}


def checklist_all_passed(checklist: dict[str, str]) -> bool:
    return all(checklist.get(key) == CERTIFY_PASS for key in CERTIFY_CHECKLIST_KEYS)


def format_certify_checklist(checklist: dict[str, str]) -> str:
    """Pretty-print the frozen checklist object (stable key order)."""
    payload = {key: checklist.get(key, CERTIFY_FAIL) for key in CERTIFY_CHECKLIST_KEYS}
    return json.dumps(payload, indent=2) + "\n"


def emit_certify_checklist(
    checklist: dict[str, str],
    *,
    json_stdout: bool = False,
    output_path: str | os.PathLike[str] | None = None,
) -> str:
    """Write the checklist JSON to stdout and/or a receipt file."""
    text = format_certify_checklist(checklist)
    if json_stdout:
        sys.stdout.write(text)
        sys.stdout.flush()
    if output_path:
        Path(output_path).write_text(text, encoding="utf-8")
    return text


def resolve_certify_agent_id() -> str:
    """Return the team AGENT_ID. Certify never invents a doctor-* identity."""
    return (os.getenv("AGENT_ID") or "").strip()


def infer_assignment_modality(*payloads: Any) -> str:
    """Return ``text`` or ``image`` from competition/session challenge_type."""
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        raw = str(
            payload.get("challenge_type") or payload.get("modality") or ""
        ).strip().lower()
        if "image" in raw:
            return "image"
        if raw in {"text", "text_challenge"}:
            return "text"
    return "text"


def normalize_requested_modality(value: str | None) -> str | None:
    if value is None:
        return None
    raw = str(value).strip().lower()
    if raw not in CERTIFY_MODALITIES:
        raise ValueError(
            f"modality must be one of: {', '.join(CERTIFY_MODALITIES)}"
        )
    return raw


def resolve_certify_usage_scope() -> str:
    """Return the battle usage scope without inventing a doctor-<uuid> value.

    Prefer the sandbox-injected ``ARENA_USAGE_SCOPE``. If that is unset, read
    ``usage_scope`` from ``/api/competition``.
    """
    env_scope = (os.getenv("ARENA_USAGE_SCOPE") or "").strip()
    if env_scope:
        return env_scope
    competition = _fetch_json_or_none("/api/competition")
    if isinstance(competition, dict):
        return str(competition.get("usage_scope") or "").strip()
    return ""


def _read_phase(health: Any, competition: Any) -> str:
    for payload in (competition, health):
        if isinstance(payload, dict):
            phase = str(payload.get("phase") or "").strip().lower()
            if phase:
                return phase
    return ""


def _phase_timeout_s(default: float = 600.0) -> float:
    raw = os.getenv("ARENA_CERTIFY_PHASE_TIMEOUT_S") or os.getenv(
        "ARENA_REGISTRATION_TIMEOUT_S",
        "",
    )
    if not str(raw).strip():
        return default
    try:
        timeout_s = float(raw)
    except ValueError:
        return default
    return timeout_s if timeout_s > 0 else default


def wait_for_phase(
    phases: set[str],
    *,
    timeout_s: float | None = None,
    poll_s: float = 1.0,
    fetch_health: Callable[[], Any] | None = None,
    fetch_competition: Callable[[], Any] | None = None,
    sleep: Callable[[float], None] | None = None,
    monotonic: Callable[[], float] | None = None,
) -> str:
    """Poll health/competition until ``phase`` is one of ``phases``.

    Raises ``TimeoutError`` if the deadline is reached first.
    """
    wanted = {str(phase).strip().lower() for phase in phases if str(phase).strip()}
    if not wanted:
        raise ValueError("wait_for_phase requires at least one phase")
    limit_s = _phase_timeout_s() if timeout_s is None else timeout_s
    get_health = fetch_health or (lambda: _fetch_json_or_none("/api/health"))
    get_competition = fetch_competition or (lambda: _fetch_json_or_none("/api/competition"))
    sleeper = time.sleep if sleep is None else sleep
    clock = time.monotonic if monotonic is None else monotonic
    deadline = clock() + max(limit_s, 0.0)
    while True:
        phase = _read_phase(get_health(), get_competition())
        if phase in wanted:
            return phase
        remaining = deadline - clock()
        if remaining <= 0:
            wanted_label = ", ".join(sorted(wanted))
            raise TimeoutError(
                f"Timed out after {limit_s:g}s waiting for phase {wanted_label} "
                f"(last seen {phase or 'unknown'})."
            )
        sleeper(min(poll_s, remaining) if poll_s > 0 else remaining)



def _http_json(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    payload: dict[str, Any] | None = None,
    timeout_s: float = 10.0,
) -> tuple[int, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urlopen(request, timeout=timeout_s) as response:
            body = response.read().decode("utf-8")
            status = int(getattr(response, "status", 200) or 200)
            if not body.strip():
                return status, None
            return status, json.loads(body)
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            parsed: Any = json.loads(raw) if raw.strip() else None
        except json.JSONDecodeError:
            parsed = raw
        return int(exc.code), parsed
    except (URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(str(exc)) from exc


def check_resolved_urls() -> None:
    """Print and validate resolved API / MCP / proxy URLs."""
    try:
        api_base = get_api_base()
        mcp_url = get_mcp_url()
        proxy_host = get_proxy_host()
    except ValueError as exc:
        _fail(
            "resolved URLs",
            str(exc),
            "Set ARENA_SERVER to an HTTPS origin (or loopback host for local dev).",
        )

    if not (os.getenv("ARENA_SERVER") or "").strip() and not (
        os.getenv("ARENA_API_BASE") or ""
    ).strip():
        _fail(
            "resolved URLs",
            "ARENA_SERVER (or ARENA_API_BASE) is not set.",
            "Copy .env.example to .env and set ARENA_SERVER to the organizer origin.",
        )

    _print_ok(
        "resolved URLs",
        f"api={api_base} mcp={mcp_url} proxy={proxy_host}",
    )


def check_api_health(timeout_s: float = 5.0) -> None:
    """GET /api/health on the resolved API base."""
    api_base = get_api_base()
    url = f"{api_base.rstrip('/')}/api/health"
    try:
        response = requests.get(url, headers={"Accept": "application/json"}, timeout=timeout_s)
    except requests.RequestException as exc:
        _fail(
            "API health",
            f"Could not reach {url}: {exc}",
            "Confirm ARENA_SERVER is reachable and the Agent Gauntlet API is running.",
        )

    if response.status_code >= 400:
        _fail(
            "API health",
            f"{url} returned HTTP {response.status_code}.",
            "Confirm the API service is up on the organizer host.",
        )
    _print_ok("API health", f"{url} -> HTTP {response.status_code}")


def check_api_key() -> None:
    """Reuse ensure_connected() for key validation."""
    try:
        ensure_connected.cache_clear()
        ensure_connected()
    except SystemExit as exc:
        message = str(exc) if str(exc) else "ARENA_API_KEY validation failed."
        _fail(
            "API key",
            message,
            "Set ARENA_API_KEY in .env to the organizer-provided battle/practice key.",
        )
    _print_ok("API key", "validated via /api/keys/validate")


async def _list_mcp_tools() -> list[str]:
    async with McpArenaClient() as client:
        return await client.list_tools()


def check_mcp_tools() -> None:
    """Connect over MCP SSE and list discovered tools."""
    mcp_url = get_mcp_url()
    try:
        tools = asyncio.run(_list_mcp_tools())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - surface any transport failure
        _fail(
            "MCP tools",
            f"Could not list tools from {mcp_url}/sse: {exc}",
            "Confirm ARENA_MCP_URL/ARENA_SERVER, ARENA_API_KEY, and that the MCP "
            "service is listening (SSE at /sse).",
        )

    if not tools:
        _fail(
            "MCP tools",
            "MCP connected but returned an empty tool list.",
            "Ask the organizer whether challenge plugins are loaded; retry after lobby opens.",
        )

    preview = ", ".join(tools[:12])
    if len(tools) > 12:
        preview += ", ..."
    _print_ok("MCP tools", f"{len(tools)} tools: {preview}")


def check_proxy_models() -> list[str]:
    """Fetch the proxy /models roster."""
    proxy_host = get_proxy_host()
    models = fetch_available_models()
    if not models:
        _fail(
            "proxy models",
            f"No models returned from {proxy_host.rstrip('/')}/models.",
            "Confirm LLM_PROXY_HOST/ARENA_SERVER, ARENA_API_KEY bearer auth, and "
            "that the LLM proxy is running.",
        )
    preview = ", ".join(models[:8])
    if len(models) > 8:
        preview += ", ..."
    _print_ok("proxy models", f"{len(models)} models: {preview}")
    return models


def _usage_total_tokens(payload: Any) -> int:
    if not isinstance(payload, dict):
        return 0
    usage = payload.get("usage")
    if isinstance(usage, dict):
        for key in ("total_tokens", "prompt_tokens", "completion_tokens"):
            try:
                value = int(usage.get(key) or 0)
            except (TypeError, ValueError):
                value = 0
            if value > 0:
                return value
    for key in ("total_tokens", "prompt_tokens", "completion_tokens"):
        try:
            value = int(payload.get(key) or 0)
        except (TypeError, ValueError):
            value = 0
        if value > 0:
            return value
    return 0


def check_attributed_inference_and_usage(
    models: list[str],
    *,
    certify: bool = False,
) -> None:
    """Run one minimal chat completion with telemetry headers, then assert usage."""
    proxy_host = get_proxy_host()
    api_key = get_arena_api_key()
    if certify:
        agent_id = resolve_certify_agent_id()
        if not agent_id:
            _fail(
                "attributed inference",
                "AGENT_ID is required in --certify mode so usage is attributed "
                "to the team identity, not a throwaway doctor-* id.",
                "Set AGENT_ID to the organizer-approved team identity.",
            )
        usage_scope = resolve_certify_usage_scope()
        if not usage_scope:
            _fail(
                "scoped usage",
                "No usage_scope is available. Certify mode never invents a "
                "doctor-<uuid> scope.",
                "The sandbox must inject ARENA_USAGE_SCOPE, or /api/competition "
                "must report usage_scope.",
            )
    else:
        agent_id = (os.getenv("AGENT_ID") or "").strip() or "doctor-agent"
        usage_scope = (os.getenv("ARENA_USAGE_SCOPE") or "").strip() or (
            f"doctor-{uuid.uuid4().hex[:10]}"
        )
    model = models[0]
    url = f"{proxy_host.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    headers.update(build_proxy_headers(agent_id=agent_id, usage_scope=usage_scope))
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly: pong"}],
        "temperature": 0.0,
        "max_tokens": 8,
    }

    try:
        status, body = _http_json("POST", url, headers=headers, payload=payload, timeout_s=60.0)
    except RuntimeError as exc:
        _fail(
            "attributed inference",
            f"Inference request to {url} failed: {exc}",
            "Confirm proxy connectivity and that the first roster model can serve chat.",
        )

    if status >= 400:
        _fail(
            "attributed inference",
            f"Inference returned HTTP {status}: {body!r}",
            "Confirm ARENA_API_KEY is authorized for the proxy and the model alias exists.",
        )

    _print_ok(
        "attributed inference",
        f"model={model} X-Agent-ID={agent_id} X-Round-ID={usage_scope}",
    )

    encoded_scope = quote(usage_scope, safe="")
    encoded_agent = quote(agent_id, safe="")
    usage_url = f"{proxy_host.rstrip('/')}/usage/{encoded_scope}/{encoded_agent}"
    try:
        usage_status, usage_body = _http_json(
            "GET",
            usage_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
            },
            timeout_s=10.0,
        )
    except RuntimeError as exc:
        _fail(
            "scoped usage",
            f"Could not read {usage_url}: {exc}",
            "Confirm the proxy exposes /usage/{scope}/{agent_id} and auth matches ARENA_API_KEY.",
        )

    if usage_status == 404 or _usage_total_tokens(usage_body) <= 0:
        if certify:
            _fail(
                "scoped usage",
                f"Expected nonzero scope-exact tokens at {usage_url}; "
                f"got HTTP {usage_status} payload={usage_body!r}. Unscoped "
                "fallback is disabled in --certify mode.",
                "Confirm X-Agent-ID and X-Round-ID were accepted and "
                "/usage/{scope}/{agent_id} is keyed by the injected "
                "ARENA_USAGE_SCOPE.",
            )
        # Fall back to unscoped agent usage if the proxy has not keyed by scope yet.
        fallback_url = f"{proxy_host.rstrip('/')}/usage/{encoded_agent}"
        try:
            usage_status, usage_body = _http_json(
                "GET",
                fallback_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Accept": "application/json",
                },
                timeout_s=10.0,
            )
        except RuntimeError as exc:
            _fail(
                "scoped usage",
                f"Scoped usage missing and fallback {fallback_url} failed: {exc}",
                "Confirm X-Agent-ID / X-Round-ID were accepted and /usage is enabled.",
            )

    total = _usage_total_tokens(usage_body)
    if usage_status >= 400 or total <= 0:
        _fail(
            "scoped usage",
            f"Expected nonzero token usage for agent={agent_id} scope={usage_scope}; "
            f"got HTTP {usage_status} payload={usage_body!r}.",
            "Ensure the inference call sent X-Agent-ID and X-Round-ID, then retry. "
            "On practice, confirm the LiteLLM usage tracker is enabled.",
        )

    _print_ok("scoped usage", f"total_tokens={total} scope={usage_scope} agent={agent_id}")


def _fetch_json_or_none(path: str, timeout_s: float = 5.0) -> Any:
    """GET a JSON endpoint, returning None on any failure."""
    try:
        url = f"{get_api_base().rstrip('/')}{path}"
        status, body = _http_json(
            "GET",
            url,
            headers={"Accept": "application/json"},
            timeout_s=timeout_s,
        )
    except (RuntimeError, ValueError):
        return None
    if status >= 400:
        return None
    return body


def classify_full_target(health: Any, competition: Any) -> tuple[str | None, str]:
    """Decide whether a destructive `--full` run is allowed.

    Returns ``(target, detail)``. A ``target`` of None means refuse. Anything
    that is not positively identified as Practice or a warmup battle is
    refused, so unreachable or unfamiliar servers fail closed.
    """
    if isinstance(health, dict):
        service = str(health.get("service") or "").strip().lower()
        if service in PRACTICE_SERVICE_NAMES:
            return "practice", f"/api/health reports service={service}"
        if "practice_gate_mode" in health:
            return "practice", "/api/health reports practice_gate_mode"
    else:
        service = ""

    if isinstance(competition, dict):
        if competition.get("warmup") is True:
            return "warmup", "/api/competition reports warmup=true (non-scoring battle)"
        if str(competition.get("phase") or "").strip().lower() == "warmup":
            return "warmup", "/api/competition reports phase=warmup"

    if health is None:
        return None, "/api/health could not be read"
    if service:
        return None, f"/api/health reports service={service} and no warmup battle is open"
    return None, "/api/health did not identify a Practice server and no warmup battle is open"


def check_full_gate() -> str:
    """Refuse `--full` unless the target is Practice or a warmup battle."""
    health = _fetch_json_or_none("/api/health")
    competition = _fetch_json_or_none("/api/competition")
    target, detail = classify_full_target(health, competition)
    if target is None:
        _fail(
            "full-run gate",
            f"Refusing --full: {detail}. On the operator platform the first "
            "submission per agent is official and a completed battle is written "
            "to the global leaderboard, so --full would burn a real submission.",
            FULL_REFUSAL_FIX,
        )
    _print_ok("full-run gate", f"target={target} — {detail}")
    return target


def _score_value(score: Any) -> float | None:
    if not isinstance(score, dict):
        return None
    for key in ("final_score", "total_score", "score"):
        if key not in score:
            continue
        try:
            return float(score[key])
        except (TypeError, ValueError):
            continue
    return None


async def _fetch_text_challenge(agent_id: str) -> tuple[list[str], Any]:
    async with McpArenaClient() as client:
        tools = await client.list_tools()
        if "arena.get_challenge" not in tools:
            return tools, None
        return tools, await client.get_challenge(agent_id)


async def _play_image_certify(agent_id: str) -> tuple[list[str], Any, Any]:
    """Fetch the image assignment, broadcast, and echo input_image_uri."""
    async with McpArenaClient() as client:
        tools = await client.list_tools()
        if "arena.image.get_challenge" not in tools:
            return tools, None, None
        challenge = await client.get_image_challenge(agent_id)
        uri = str(getattr(challenge, "input_image_uri", "") or "").strip()
        if not uri:
            return tools, challenge, None
        await client.broadcast_image_thought(CERTIFY_IMAGE_THOUGHT, agent_id)
        submitted = await client.submit_image(
            agent_id,
            uri,
            {"model_name": "doctor-certify", "total_tokens": "0"},
            rationale=CERTIFY_IMAGE_THOUGHT,
        )
        return tools, challenge, submitted


def _submit_result_from_payload(
    payload: Any,
    *,
    agent_id: str,
    answer: str,
) -> SubmitResult:
    if isinstance(payload, SubmitResult):
        return payload
    body = payload if isinstance(payload, dict) else {}
    status = str(body.get("status") or "").strip() or "submitted"
    return SubmitResult(
        accepted=bool(body.get("accepted", status == "submitted")),
        agent_id=str(body.get("agent_id") or agent_id),
        answer=str(body.get("answer") or body.get("edited_image") or answer),
        score=body.get("score") if isinstance(body.get("score"), dict) else body.get("score"),
        status=status,
    )


def check_full_round_trip(target: str) -> None:
    """Register, read the challenge, submit, and assert a score comes back."""
    agent_id = f"doctor-full-{uuid.uuid4().hex[:8]}"
    client = HttpArenaClient()

    # Keep the lobby wait short; the doctor is a preflight, not an agent run.
    restore_timeout = "ARENA_REGISTRATION_TIMEOUT_S" not in os.environ
    if restore_timeout:
        os.environ["ARENA_REGISTRATION_TIMEOUT_S"] = "60"
    try:
        client.register(agent_id, "Agent Gauntlet doctor --full")
    except (ArenaAPIError, ArenaConnectionError) as exc:
        _fail(
            "full round trip: register",
            f"Registration failed for {agent_id}: {exc}",
            "Confirm the lobby is open on the target server and ARENA_API_KEY is valid.",
        )
    finally:
        if restore_timeout:
            os.environ.pop("ARENA_REGISTRATION_TIMEOUT_S", None)
    _print_ok("full round trip: register", f"agent_id={agent_id}")

    try:
        tools, challenge = asyncio.run(_fetch_text_challenge(agent_id))
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - surface any transport failure
        _fail(
            "full round trip: get challenge",
            f"arena.get_challenge failed: {exc}",
            "Confirm the MCP service is reachable and the agent is eligible to play.",
        )

    if challenge is None:
        _fail(
            "full round trip: get challenge",
            f"No text challenge tool is active (tools: {', '.join(tools) or 'none'}).",
            "Run --full against a text challenge; image challenges are not covered "
            "by the doctor round trip.",
        )
    _print_ok(
        "full round trip: get challenge",
        f"puzzle_id={challenge.puzzle_id or 'unknown'} type={challenge.challenge_type or 'text'}",
    )

    # A thought keeps the broadcast-thought scoring penalty out of the probe.
    try:
        client.broadcast_thought(agent_id, "doctor --full readiness probe")
    except (ArenaAPIError, ArenaConnectionError):
        pass

    try:
        result = client.submit(
            agent_id,
            "doctor --full readiness probe",
            {"model_name": "doctor-full", "total_tokens": "0"},
            challenge_type="text",
        )
    except (ArenaAPIError, ArenaConnectionError) as exc:
        _fail(
            "full round trip: submit",
            f"Submission failed for {agent_id}: {exc}",
            "Confirm the battle is running and this agent is eligible to submit.",
        )

    if result.status != "submitted":
        _fail(
            "full round trip: submit",
            f"Submission was not recorded (status={result.status}).",
            "Check the server logs for the rejection reason and retry.",
        )
    _print_ok(
        "full round trip: submit",
        f"status={result.status} answer_accepted={result.accepted}",
    )

    score = _score_value(result.score)
    if score is None:
        _fail(
            "full round trip: score",
            f"Submission accepted but no numeric score was returned: {result.score!r}.",
            "On Practice, confirm the judge is configured (judge_configured in "
            "/api/health). On a warmup battle, scores may only appear after stop.",
        )
    _print_ok("full round trip: score", f"final_score={score} target={target}")


def receipts_are_canonical(first: Any, retry: Any) -> bool:
    """True when an exact retry returned the same official receipt."""
    if first is None or retry is None:
        return False
    if getattr(first, "answer", None) != getattr(retry, "answer", None):
        return False
    if getattr(first, "status", None) != getattr(retry, "status", None):
        return False
    if getattr(first, "accepted", None) != getattr(retry, "accepted", None):
        return False
    if getattr(first, "agent_id", None) != getattr(retry, "agent_id", None):
        return False
    first_score = _score_value(getattr(first, "score", None))
    retry_score = _score_value(getattr(retry, "score", None))
    if first_score is not None or retry_score is not None:
        return first_score == retry_score
    return getattr(first, "score", None) == getattr(retry, "score", None)


def _eligible_agent_ids(competition: Any) -> list[str] | None:
    if not isinstance(competition, dict):
        return None
    eligible = competition.get("eligible_agent_ids")
    if eligible is None:
        return None
    if not isinstance(eligible, list):
        return None
    return [str(item).strip() for item in eligible if str(item).strip()]


def _probe_idempotency(
    client: HttpArenaClient,
    *,
    agent_id: str,
    official_answer: str,
    result: SubmitResult,
    checklist: dict[str, str],
) -> None:
    """Exact retry and conflict probes omit challenge_type."""
    metrics = {"model_name": "doctor-certify", "total_tokens": "0"}
    try:
        retry = client.submit(
            agent_id,
            official_answer,
            metrics,
            challenge_type=None,
        )
    except (ArenaAPIError, ArenaConnectionError) as exc:
        _fail(
            "certify: exact retry",
            f"Exact retry did not return the canonical receipt: {exc}",
            "Retry the identical first answer; the server should return the "
            "original receipt instead of opening a new submission.",
        )
    if not receipts_are_canonical(result, retry):
        _fail(
            "certify: exact retry",
            "Exact retry returned a different receipt than the official submit.",
            "The first submission is official; an identical retry must echo it.",
        )
    checklist["retry_is_canonical"] = CERTIFY_PASS
    _print_ok("certify: exact retry", "canonical receipt returned")

    try:
        client.submit(
            agent_id,
            CERTIFY_CONFLICT_ANSWER,
            metrics,
            challenge_type=None,
        )
    except ArenaAPIError as exc:
        if exc.status_code == 409:
            checklist["conflicting_answer_rejected"] = CERTIFY_PASS
            _print_ok("certify: conflicting answer", "HTTP 409")
            return
        _fail(
            "certify: conflicting answer",
            f"Different answer returned HTTP {exc.status_code}, expected 409.",
            "A later different answer must be rejected with 409.",
        )
    except ArenaConnectionError as exc:
        _fail(
            "certify: conflicting answer",
            f"Different answer failed without a 409: {exc}",
            "A later different answer must be rejected with 409.",
        )
    _fail(
        "certify: conflicting answer",
        "A different second answer was accepted; the first submit is official.",
        "Practice must reject a conflicting answer with HTTP 409.",
    )


def check_certify_round_trip(
    target: str,
    checklist: dict[str, str],
    requested_modality: str | None = None,
) -> None:
    """Play one certification round against Practice using the real AGENT_ID."""
    agent_id = resolve_certify_agent_id()
    if not agent_id:
        _fail(
            "certify: identity",
            "AGENT_ID is required in --certify mode.",
            "Set AGENT_ID to the organizer-approved team identity. Certify "
            "never registers as doctor-full-<uuid>.",
        )
    agent_name = (os.getenv("AGENT_NAME") or "").strip() or (
        "Agent Gauntlet doctor --certify"
    )
    client = HttpArenaClient()

    try:
        phase = wait_for_phase({"lobby"})
    except TimeoutError as exc:
        _fail(
            "certify: wait for lobby",
            str(exc),
            "Run certification against Practice in PRACTICE_GATE_MODE=cycle "
            "and wait for the next lobby.",
        )
    try:
        client.register(agent_id, agent_name)
    except (ArenaAPIError, ArenaConnectionError) as exc:
        _fail(
            "certify: register",
            f"Registration failed for {agent_id} during {phase}: {exc}",
            "Confirm the lobby is still open and AGENT_ID / ARENA_API_KEY match "
            "the attempt identity.",
        )
    checklist["registered_in_lobby"] = CERTIFY_PASS
    _print_ok("certify: register", f"agent_id={agent_id} phase={phase}")

    try:
        go_phase = wait_for_phase({"running"})
    except TimeoutError as exc:
        _fail(
            "certify: wait for GO",
            str(exc),
            "Wait for phase=running before calling arena.get_challenge.",
        )
    checklist["waited_for_go"] = CERTIFY_PASS
    _print_ok("certify: wait for GO", f"phase={go_phase}")

    try:
        competition = client.get_competition()
    except (ArenaAPIError, ArenaConnectionError):
        competition = _fetch_json_or_none("/api/competition")
    eligible = _eligible_agent_ids(competition)
    if eligible is None or agent_id not in eligible:
        _fail(
            "certify: frozen roster",
            f"{agent_id} is not in eligible_agent_ids={eligible!r}.",
            "Register during lobby so countdown freezes this AGENT_ID into "
            "the eligible roster.",
        )
    checklist["in_frozen_roster"] = CERTIFY_PASS
    _print_ok("certify: frozen roster", f"agent_id={agent_id}")

    assignment = infer_assignment_modality(competition)
    wanted = normalize_requested_modality(requested_modality)
    if wanted is not None and wanted != assignment:
        _fail(
            "certify: modality",
            f"Requested modality {wanted} does not match assignment {assignment}.",
            "Pin Practice to the requested modality or omit --modality.",
        )

    if assignment == "image":
        try:
            tools, challenge, submitted = asyncio.run(_play_image_certify(agent_id))
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001 - surface any transport failure
            _fail(
                "certify: get challenge",
                f"arena.image.get_challenge failed: {exc}",
                "Confirm the MCP service is reachable and the image assignment "
                "is active.",
            )
        if challenge is None:
            _fail(
                "certify: get challenge",
                f"No image challenge tool is active (tools: {', '.join(tools) or 'none'}).",
                "Run --certify --modality image against an image assignment.",
            )
        image_uri = str(getattr(challenge, "input_image_uri", "") or "").strip()
        if not image_uri:
            _fail(
                "certify: get challenge",
                "Image assignment did not expose input_image_uri; refusing to "
                "invent image bytes.",
                "Confirm the Practice image puzzle publishes input_image_uri.",
            )
        if submitted is None:
            _fail(
                "certify: submit",
                "Image assignment did not expose input_image_uri; refusing to "
                "invent image bytes.",
                "Confirm the Practice image puzzle publishes input_image_uri.",
            )
        _print_ok(
            "certify: get challenge",
            f"puzzle_id={challenge.puzzle_id or 'unknown'} type="
            f"{challenge.challenge_type or 'image'}",
        )
        result = _submit_result_from_payload(
            submitted,
            agent_id=agent_id,
            answer=image_uri,
        )
        official_answer = image_uri
    else:
        try:
            tools, challenge = asyncio.run(_fetch_text_challenge(agent_id))
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001 - surface any transport failure
            _fail(
                "certify: get challenge",
                f"arena.get_challenge failed: {exc}",
                "Confirm the MCP service is reachable and the battle is running.",
            )
        if challenge is None:
            _fail(
                "certify: get challenge",
                f"No text challenge tool is active (tools: {', '.join(tools) or 'none'}).",
                "Run --certify against a text challenge, or pass --modality image "
                "when Practice is pinned to an image assignment.",
            )
        _print_ok(
            "certify: get challenge",
            f"puzzle_id={challenge.puzzle_id or 'unknown'} type="
            f"{challenge.challenge_type or 'text'}",
        )

        try:
            client.broadcast_thought(agent_id, "doctor --certify play rehearsal")
        except (ArenaAPIError, ArenaConnectionError) as exc:
            _fail(
                "certify: broadcast thought",
                f"broadcast_thought failed for {agent_id}: {exc}",
                "Thought broadcast is required in --certify mode; a silent miss "
                "would hide the scoring penalty path.",
            )
        _print_ok("certify: broadcast thought", f"agent_id={agent_id}")

        try:
            result = client.submit(
                agent_id,
                CERTIFY_ANSWER,
                {"model_name": "doctor-certify", "total_tokens": "0"},
                challenge_type="text",
            )
        except (ArenaAPIError, ArenaConnectionError) as exc:
            _fail(
                "certify: submit",
                f"Submission failed for {agent_id}: {exc}",
                "Confirm the battle is running and this agent is in the frozen roster.",
            )
        official_answer = CERTIFY_ANSWER

    if result.status != "submitted":
        _fail(
            "certify: submit",
            f"Submission was not recorded (status={result.status}).",
            "Check the server logs for the rejection reason and retry.",
        )

    try:
        session = client.get_session(agent_id)
    except (ArenaAPIError, ArenaConnectionError) as exc:
        _fail(
            "certify: session",
            f"Could not read /api/session/{agent_id}: {exc}",
            "The session must expose final_answer and submitted_at after submit.",
        )
    if not isinstance(session, dict):
        session = {}
    final_answer = str(session.get("final_answer") or "").strip()
    if not final_answer or session.get("submitted_at") is None:
        _fail(
            "certify: answer accepted",
            f"Session is missing final_answer/submitted_at: {session!r}.",
            "Confirm the first submit landed and the session still exists "
            "before the next lobby reset.",
        )
    checklist["answer_accepted"] = CERTIFY_PASS
    _print_ok(
        "certify: submit",
        f"status={result.status} answer_accepted={result.accepted} modality={assignment}",
    )

    score = _score_value(result.score)
    if score is None:
        _fail(
            "certify: score",
            f"Submission accepted but no numeric score was returned: {result.score!r}.",
            "On Practice, confirm the judge is configured (judge_configured in "
            "/api/health).",
        )
    checklist["scored"] = CERTIFY_PASS
    _print_ok("certify: score", f"final_score={score} target={target}")

    _probe_idempotency(
        client,
        agent_id=agent_id,
        official_answer=official_answer,
        result=result,
        checklist=checklist,
    )


def run_doctor(
    full: bool = False,
    certify: bool = False,
    json_output: bool = False,
    output: str | os.PathLike[str] | None = None,
    modality: str | None = None,
) -> int:
    """Run readiness checks. Returns 0 on success; exits 1 on failure."""
    global _HUMAN_STREAM
    _load_env()
    checklist: dict[str, str] | None = empty_certify_checklist() if certify else None
    previous_stream = _HUMAN_STREAM
    if json_output:
        _HUMAN_STREAM = sys.stderr
    try:
        return _run_doctor_body(
            full=full,
            certify=certify,
            json_output=json_output,
            output=output,
            checklist=checklist,
            modality=modality,
        )
    except SystemExit:
        if checklist is not None:
            emit_certify_checklist(
                checklist,
                json_stdout=json_output,
                output_path=output,
            )
        raise
    finally:
        _HUMAN_STREAM = previous_stream


def _run_doctor_body(
    *,
    full: bool,
    certify: bool,
    json_output: bool,
    output: str | os.PathLike[str] | None,
    checklist: dict[str, str] | None,
    modality: str | None = None,
) -> int:
    _say("Agent Gauntlet doctor")
    _say("---------------------")
    check_resolved_urls()
    check_api_health()
    check_api_key()
    check_mcp_tools()
    models = check_proxy_models()
    check_attributed_inference_and_usage(models, certify=certify)
    if certify:
        assert checklist is not None
        checklist["attribution_scope_exact"] = CERTIFY_PASS
        target = check_full_gate()
        check_certify_round_trip(target, checklist, requested_modality=modality)
        emit_certify_checklist(
            checklist,
            json_stdout=json_output,
            output_path=output,
        )
        if not checklist_all_passed(checklist):
            _say("---------------------")
            _say("Certification checklist failed.")
            return 1
        _say("---------------------")
        _say("All certification checks passed.")
        return 0
    if full:
        target = check_full_gate()
        check_full_round_trip(target)
    _say("---------------------")
    _say("All checks passed.")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m arena_clients.doctor",
        description=(
            "Agent Gauntlet readiness checks. "
            "Frozen certify contract: python -m arena_clients.doctor "
            "--certify --json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "certify contract:\n"
            "  Flags: python -m arena_clients.doctor --certify --json\n"
            "  Sandbox injects ARENA_USAGE_SCOPE (never invent doctor-<uuid>).\n"
            "  --json writes the checklist object to stdout (pretty JSON).\n"
            "  Optional --output PATH also writes that JSON for the receipt.\n"
            "  Exit 0 only if every checklist item is PASS; otherwise exit 1.\n"
            "  Checklist keys (PASS or FAIL): registered_in_lobby, waited_for_go,\n"
            "  in_frozen_roster, answer_accepted, scored, retry_is_canonical,\n"
            "  conflicting_answer_rejected, attribution_scope_exact.\n"
            "  --certify implies the full play path and reuses check_full_gate().\n"
            "  Optional --modality {text,image} fails closed on a mismatch.\n"
            "  Image play uses arena.image.get_challenge and echoes input_image_uri."
        ),
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "Also run a real register / get-challenge / submit / score round "
            "trip. Refuses to run unless the target is the Practice server or "
            "a warmup battle."
        ),
    )
    parser.add_argument(
        "--certify",
        action="store_true",
        help=(
            "Play-rehearsal certification: wait for lobby, register as AGENT_ID, "
            "wait for GO, submit, then assert exact-retry and 409-on-conflict. "
            "Implies the full play path and still fails closed off Practice. "
            "Never invents a doctor-<uuid> usage scope."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help=(
            "With --certify, write the PASS/FAIL checklist object to stdout "
            "as pretty JSON. Human-readable progress goes to stderr."
        ),
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        default=None,
        help=(
            "With --certify, also write the same checklist JSON to PATH "
            "for the receipt."
        ),
    )
    parser.add_argument(
        "--modality",
        choices=CERTIFY_MODALITIES,
        default=None,
        help=(
            "With --certify, fail closed unless the assignment matches. "
            "Omit to auto-detect from /api/competition."
        ),
    )
    args = parser.parse_args(argv)
    if (args.json or args.output or args.modality) and not args.certify:
        parser.error("--json, --output, and --modality require --certify")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_doctor(
        full=args.full,
        certify=args.certify,
        json_output=args.json,
        output=args.output,
        modality=args.modality,
    )


if __name__ == "__main__":
    sys.exit(main())
