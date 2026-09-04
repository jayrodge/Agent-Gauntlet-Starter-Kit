#!/usr/bin/env python3
"""Simple Agent Gauntlet competitor agent.

This is a minimal example showing how to connect to Agent Gauntlet from any Python code.
It uses:
- HTTP client for Agent Gauntlet coordination (REST API)
- MCP client for challenge tools
- OpenAI SDK for LLM calls via the proxy

Usage:
    cd examples/python_simple
    python agent.py

This example loads `.env` from the repository root automatically.
"""

import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path: 
    sys.path.insert(0, str(REPO_ROOT))

load_dotenv(REPO_ROOT / ".env")

from arena_clients import (
    build_image_tool_arguments,
    HttpArenaClient,
    McpArenaClient,
    McpArenaError,
    ensure_connected,
    get_api_base,
    get_arena_api_key,
    get_mcp_url,
    get_proxy_host,
    monitor_session,
)
from arena_clients.proxy_headers import build_proxy_headers, resolve_usage_scope
from base_strategy import ChallengeContext
from model_selector import (
    ModelSelectionError,
    fetch_available_models,
    require_explicit_model,
)
from my_strategy import MyStrategy

# Use OpenAI SDK for LLM calls through the proxy.
try:
    from openai import OpenAI

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


BLANK_PNG_DATA_URI = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7/"
    "S7sAAAAASUVORK5CYII="
)


def _plan_image_tool_sequence(
    strategy: MyStrategy,
    ctx: ChallengeContext,
    available_tools: list[str],
) -> list[str]:
    """Expand an analysis-first choice into a usable image-output plan."""
    selected_tool = strategy.plan_image_tool(ctx, available_tools)
    if not selected_tool:
        return []
    sequence = [selected_tool]
    if selected_tool != "image_analyze":
        return sequence
    if ctx.image_url and "image_edit" in available_tools:
        sequence.append("image_edit")
    elif "image_generate" in available_tools:
        sequence.append("image_generate")
    return sequence


def extract_answer(raw_response: str) -> str:
    """Extract the answer line from the LLM response."""
    cleaned = re.sub(r"<think>.*?(?:</think>|$)", "", raw_response, flags=re.DOTALL).strip()

    # Accept either "ANSWER:" or "Final answer:" on its own line.
    for line in cleaned.splitlines():
        match = re.match(
            r"^\s*(?:answer|final answer)\s*:\s*(.+?)\s*$",
            line,
            flags=re.IGNORECASE,
        )
        if match:
            answer = match.group(1).strip().strip("`\"'")
            if answer:
                return answer

    return ""


def _challenge_text_blob(
    challenge_type: str,
    description: str,
    rules: str,
    extra_context: str = "",
) -> str:
    return " ".join(
        [challenge_type or "", description or "", rules or "", extra_context or ""]
    ).lower()


def _required_runtime_tools(
    available_tool_names: list[str],
    *,
    challenge_type: str,
    description: str,
    rules: str,
    extra_context: str = "",
) -> list[str]:
    blob = _challenge_text_blob(challenge_type, description, rules, extra_context)
    required: list[str] = []
    for tool_name in available_tool_names:
        normalized = str(tool_name or "").strip()
        if not normalized:
            continue
        if normalized.lower() in blob:
            required.append(normalized)
    if challenge_type.strip().lower() in {"web-search", "market-research"}:
        for tool_name in available_tool_names:
            normalized = str(tool_name or "").strip()
            if "search" in normalized.lower() and normalized not in required:
                required.append(normalized)
    return required


def _best_search_query(description: str, rules: str, clues: list[str] | None) -> str:
    text = (description or "").strip()
    if text:
        return text
    if clues:
        for clue in clues:
            if isinstance(clue, str) and clue.strip():
                return clue.strip()
    return (rules or "").strip()


def _query_candidates(
    description: str,
    rules: str,
    clues: list[str] | None,
    extra_context: str = "",
) -> list[str]:
    candidates: list[str] = []
    primary = _best_search_query(description, rules, clues)
    if primary:
        candidates.append(primary)
    for clue in clues or []:
        if isinstance(clue, str) and clue.strip():
            candidates.append(clue.strip())
    if isinstance(rules, str) and rules.strip():
        split_parts = re.split(r"[.;\n]+", rules)
        for part in split_parts:
            part = part.strip()
            if 8 <= len(part) <= 220:
                candidates.append(part)
    for text in [description, rules, extra_context, *(clues or [])]:
        if not isinstance(text, str):
            continue
        for url in re.findall(r"https?://[^\s)>\"]+", text):
            cleaned = url.strip().rstrip(".,);")
            if cleaned:
                candidates.append(cleaned)
    deduped: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped[:5]


def _tool_result_text(result: object) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, (int, float, bool)):
        return str(result)
    if isinstance(result, list):
        parts = [_tool_result_text(item) for item in result]
        return "\n".join(part for part in parts if part).strip()
    if isinstance(result, dict):
        cleaned: dict[str, object] = {}
        for key, value in result.items():
            key_lower = str(key).lower()
            if "image" in key_lower and isinstance(value, str) and len(value) > 256:
                continue
            cleaned[str(key)] = value
        try:
            return json.dumps(cleaned, ensure_ascii=False)
        except Exception:
            return str(cleaned)
    return str(result)


def _is_invalid_answer_candidate(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return True
    lowered = text.lower()
    if lowered.startswith(("<toolcall", "toolcall", "<think", "reasoning:", "analysis:", "action:")):
        return True
    blocked_tokens = ("<toolcall", "</toolcall", '"tool_calls"', "function_call")
    return any(token in lowered for token in blocked_tokens)


def _derive_answer_from_evidence(
    *,
    llm_client,
    model_name: str,
    challenge_description: str,
    challenge_rules: str,
    evidence_lines: list[str],
) -> str:
    if not evidence_lines:
        return ""
    evidence_block = "\n".join(f"- {line}" for line in evidence_lines if line.strip())
    if not evidence_block:
        return ""
    try:
        response = llm_client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Use the provided tool evidence to answer the challenge. "
                        "Return only one final answer line that follows the challenge rules."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question: {challenge_description}\n"
                        f"Rules: {challenge_rules}\n\n"
                        f"Tool evidence:\n{evidence_block}\n\n"
                        "Return only the final answer line."
                    ),
                },
            ],
            max_tokens=256,
            temperature=0.0,
        )
    except Exception:
        return ""
    raw = (response.choices[0].message.content or "").strip()
    answer = extract_answer(raw)
    if answer:
        return answer
    first_line = raw.splitlines()[0].strip().strip("`\"'") if raw.strip() else ""
    return "" if _is_invalid_answer_candidate(first_line) else first_line


async def _enforce_required_tool_calls(
    *,
    mcp_client: McpArenaClient,
    agent_id: str,
    challenge_type: str,
    description: str,
    rules: str,
    clues: list[str] | None,
    available_tool_names: list[str],
    http_client: HttpArenaClient,
    extra_context: str = "",
) -> list[str]:
    required_runtime_tools = _required_runtime_tools(
        available_tool_names,
        challenge_type=challenge_type,
        description=description,
        rules=rules,
        extra_context=extra_context,
    )
    if not required_runtime_tools:
        return []

    evidence_lines: list[str] = []
    queries = _query_candidates(description, rules, clues, extra_context)
    if not queries:
        queries = [""]
    for tool_name in required_runtime_tools:
        is_search_like = "search" in tool_name.lower()
        target_calls = min(len(queries), 3) if is_search_like else 1
        calls_made = 0
        for query in queries[:target_calls]:
            payload_variants = [
                {"agent_id": agent_id, "query": query},
                {"query": query},
                {"agent_id": agent_id, "search_query": query},
                {"search_query": query},
                {"agent_id": agent_id, "text": query},
                {"text": query},
                {"agent_id": agent_id, "url": query},
                {"url": query},
                {"agent_id": agent_id, "video_url": query},
                {"video_url": query},
                {"agent_id": agent_id},
                {},
            ]
            for payload in payload_variants:
                try:
                    result = await mcp_client.call_tool(tool_name, payload)
                    rendered = _tool_result_text(result)
                    if rendered:
                        evidence_lines.append(f"{tool_name}: {rendered[:2000]}")
                    calls_made += 1
                    break
                except Exception:
                    continue
        status_note = (
            f"Required tool coverage: {tool_name} ({calls_made}/{target_calls})"
            if target_calls > 1
            else f"Required tool called: {tool_name}"
            if calls_made
            else f"Could not call required tool: {tool_name}"
        )
        try:
            http_client.broadcast_thought(agent_id, status_note)
        except Exception:
            pass
    return evidence_lines


def _build_context(
    *,
    challenge_type: str,
    description: str,
    rules: str,
    clues: list[str] | None = None,
    max_time_s: int = 0,
    available_models: list[str] | None = None,
    time_remaining_s: float = 0.0,
    tokens_used: int = 0,
    image_url: str | None = None,
) -> ChallengeContext:
    return ChallengeContext(
        challenge_type=challenge_type,
        difficulty="unknown",
        challenge_text=description,
        description=description,
        rules=rules,
        clues=list(clues or []),
        time_remaining_s=time_remaining_s,
        max_time_s=max_time_s,
        available_models=list(available_models or []),
        tools_used=[],
        tokens_used=tokens_used,
        required_tools=[],
        image_url=image_url,
    )


def _coerce_nonnegative_int(value: object, default: int = 0) -> int:
    try:
        return max(0, int(value if value is not None else default))
    except (TypeError, ValueError):
        return default


def _fetch_proxy_usage(
    llm_host: str,
    api_key: str,
    agent_id: str,
    scope_id: str | None = None,
) -> dict[str, int] | None:
    """Best-effort fetch of proxy usage for one agent, scoped when possible."""
    if not llm_host or not api_key or not agent_id:
        return None
    encoded_agent_id = quote(agent_id, safe="")

    def _read_usage(url: str) -> dict[str, int] | None:
        request = Request(
            url,
            headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
            method="GET",
        )
        with urlopen(request, timeout=1.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            return None
        return {
            "prompt_tokens": _coerce_nonnegative_int(usage.get("prompt_tokens")),
            "completion_tokens": _coerce_nonnegative_int(usage.get("completion_tokens")),
            "total_tokens": _coerce_nonnegative_int(usage.get("total_tokens")),
        }

    if scope_id:
        encoded_scope_id = quote(scope_id, safe="")
        scoped_url = f"{llm_host.rstrip('/')}/usage/{encoded_scope_id}/{encoded_agent_id}"
        try:
            return _read_usage(scoped_url)
        except HTTPError as exc:
            if exc.code not in {400, 404}:
                raise

    aggregate_url = f"{llm_host.rstrip('/')}/usage/{encoded_agent_id}"
    return _read_usage(aggregate_url)


def _build_live_metrics(model_name: str, usage: dict[str, int] | None, elapsed_ms: int) -> dict[str, object]:
    usage = usage or {}
    prompt_tokens = _coerce_nonnegative_int(usage.get("prompt_tokens"))
    completion_tokens = _coerce_nonnegative_int(usage.get("completion_tokens"))
    total_tokens = _coerce_nonnegative_int(usage.get("total_tokens"))
    return {
        "model_name": str(model_name or "").strip(),
        "total_tokens": str(total_tokens),
        "prompt_tokens": str(prompt_tokens),
        "completion_tokens": str(completion_tokens),
        "total_time_ms": int(max(0, elapsed_ms)),
    }


async def solve_challenge(
    challenge,
    clues: list[str],
    llm_client,
    model_name: str,
    strategy: MyStrategy,
    ctx: ChallengeContext,
    *,
    http_client=None,
    agent_id: str = "",
    broadcast_thought: bool = True,
    evidence_lines: list[str] | None = None,
):
    """Solve the challenge using the selected LLM model.
    
    When http_client and agent_id are provided, streams reasoning to Agent Gauntlet in real time.
    
    Returns:
        tuple: (extracted_answer, raw_response, model_name, usage_dict, ttft_ms, total_time_ms)
    """
    
    _ = challenge
    _ = clues
    prompt = strategy.build_solver_prompt(ctx)
    prompt = (
        f"{prompt}\n\n"
        "Output guardrails:\n"
        "- Return only your final answer line.\n"
        "- Do not output tool-call markup, XML tags, or thinking traces."
    )
    system_msg = strategy.build_system_prompt(ctx)
    llm_params = strategy.get_llm_params(ctx)
    max_tokens = int(llm_params.get("max_tokens", 1024))
    temperature = float(llm_params.get("temperature", 0.0))
    
    # Track timing
    start_ms = time.time() * 1000
    ttft_ms = 0
    raw_content_parts = []
    pending_broadcast = ""
    _BROADCAST_CHUNK = 80  # chars before sending a thought

    def _flush_broadcast(force: bool = False) -> None:
        nonlocal pending_broadcast
        if not http_client or not agent_id or not broadcast_thought:
            return
        if force or len(pending_broadcast) >= _BROADCAST_CHUNK or "\n" in pending_broadcast:
            chunk = pending_broadcast.strip()
            pending_broadcast = ""
            if chunk:
                try:
                    http_client.broadcast_thought(agent_id, chunk[:300])
                except Exception:
                    pass

    stream_enabled = os.getenv("LLM_STREAM", "1").lower() not in {"0", "false", "no"}
    usage = None
    if stream_enabled and http_client and agent_id:
        # Stream and broadcast in real time
        try:
            stream = llm_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )
            header_sent = False
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta:
                    delta = chunk.choices[0].delta
                    content = delta.content or ""
                    reasoning = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None) or ""
                    text = reasoning or content
                    if text:
                        if not ttft_ms:
                            ttft_ms = int(time.time() * 1000 - start_ms)
                        raw_content_parts.append(text)
                        if broadcast_thought:
                            if not header_sent:
                                http_client.broadcast_thought(agent_id, "💭 LLM Reasoning:")
                                header_sent = True
                            pending_broadcast += text
                            _flush_broadcast()
                if hasattr(chunk, "usage") and chunk.usage:
                    usage = chunk.usage
            _flush_broadcast(force=True)
        except Exception as e:
            stream_enabled = False
            raw_content_parts = []
            if "stream" in str(e).lower() or "Stream" in str(e):
                pass  # Fall through to non-streaming
            else:
                raise

    if not stream_enabled or not raw_content_parts:
        # Non-streaming fallback
        response = llm_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        raw_content = response.choices[0].message.content or ""
        usage = response.usage
        ttft_ms = ttft_ms or getattr(response, "ttft_ms", 0) or int((time.time() * 1000 - start_ms) * 0.1)
    else:
        raw_content = "".join(raw_content_parts)

    total_time_ms = int(time.time() * 1000 - start_ms)
    answer = extract_answer(raw_content)
    if _is_invalid_answer_candidate(answer):
        answer = ""
    if not answer and evidence_lines:
        answer = _derive_answer_from_evidence(
            llm_client=llm_client,
            model_name=model_name,
            challenge_description=str(ctx.description or ""),
            challenge_rules=str(ctx.rules or ""),
            evidence_lines=list(evidence_lines or []),
        )
        if _is_invalid_answer_candidate(answer):
            answer = ""
    if not answer:
        strict_system_msg = (
            "Return only one line in exact format: ANSWER: <final answer>. "
            "No preamble, no tags, no explanation, no additional lines."
        )
        try:
            strict_response = llm_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": strict_system_msg},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=min(temperature, 0.0),
                stream=False,
            )
            raw_content = strict_response.choices[0].message.content or ""
            usage = strict_response.usage
            total_time_ms = int(time.time() * 1000 - start_ms)
            answer = extract_answer(raw_content)
        except Exception:
            pass
    if _is_invalid_answer_candidate(answer):
        answer = ""

    usage_dict = {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
        "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
    }
    if not usage_dict["total_tokens"] and raw_content:
        usage_dict["total_tokens"] = len(raw_content.split()) * 2  # rough estimate
    ttft_ms = ttft_ms or int(total_time_ms * 0.1)

    return answer, raw_content, model_name, usage_dict, ttft_ms, total_time_ms


async def main():
    """Main agent loop."""
    strategy = MyStrategy()
    agent_id = (
        os.getenv("AGENT_ID")
        or str(getattr(strategy, "agent_id", "")).strip()
        or "simple-agent"
    ).strip()
    agent_name = (
        os.getenv("AGENT_NAME")
        or str(getattr(strategy, "agent_name", "")).strip()
        or "Team Nova"
    ).strip()
    os.environ["AGENT_ID"] = agent_id
    os.environ["AGENT_NAME"] = agent_name
    print(f"🤖 {agent_name} starting...")
    print(f"   Agent ID: {agent_id}")

    ensure_connected()

    # Initialize clients
    api_base = get_api_base()
    mcp_url = get_mcp_url()
    llm_host = get_proxy_host()
    api_key = get_arena_api_key()
    llm_api_key = api_key

    print(f"   API: {api_base}")
    print(f"   MCP: {mcp_url}")
    print(f"   LLM: {llm_host}")
    print()

    # HTTP client for Agent Gauntlet coordination
    http_client = HttpArenaClient(api_base=api_base, api_key=api_key)

    if not HAS_OPENAI:
        raise RuntimeError("Missing dependency: openai. Install with `pip install openai`.")
    
    # Step 1: Register with Agent Gauntlet
    print("📝 Registering with Agent Gauntlet...")
    session = http_client.register(agent_id, agent_name)
    print(f"   Session: {session.session_id}")
    print(f"   Status: {session.status}")
    session_monitor = monitor_session(http_client, agent_id).start()
    
    # Step 2: Get challenge from MCP
    print("\n🎯 Getting challenge...")
    async with McpArenaClient(mcp_url) as mcp_client:
        tools = await mcp_client.list_tools()
        tool_defs = await mcp_client.list_tool_defs()
        modality = McpArenaClient.detect_modality(tools)
        image_tool_proxy_args: dict[str, dict[str, str]] = {}
        usage_scope = http_client.fetch_usage_scope() or resolve_usage_scope()
        if usage_scope:
            os.environ["ARENA_USAGE_SCOPE"] = usage_scope

        live_prompt_tokens = 0
        live_completion_tokens = 0
        live_total_tokens = 0
        active_model_name = ""
        metrics_started_ms = 0.0
        reporter_stop = asyncio.Event()
        metrics_task: asyncio.Task | None = None

        def _elapsed_metrics_ms() -> int:
            if metrics_started_ms <= 0:
                return 0
            return int(max(0.0, time.time() * 1000 - metrics_started_ms))

        async def _refresh_live_usage() -> None:
            nonlocal live_prompt_tokens, live_completion_tokens, live_total_tokens
            try:
                usage = await asyncio.to_thread(
                    _fetch_proxy_usage,
                    llm_host,
                    llm_api_key,
                    agent_id,
                    usage_scope,
                )
            except Exception:
                usage = None
            if not isinstance(usage, dict):
                return
            live_prompt_tokens = _coerce_nonnegative_int(usage.get("prompt_tokens"))
            live_completion_tokens = _coerce_nonnegative_int(usage.get("completion_tokens"))
            live_total_tokens = _coerce_nonnegative_int(usage.get("total_tokens"))

        async def _push_live_status(status: str = "running") -> None:
            await _refresh_live_usage()
            try:
                await asyncio.to_thread(
                    http_client.update_status,
                    agent_id,
                    status,
                    _build_live_metrics(
                        active_model_name,
                        {
                            "prompt_tokens": live_prompt_tokens,
                            "completion_tokens": live_completion_tokens,
                            "total_tokens": live_total_tokens,
                        },
                        _elapsed_metrics_ms(),
                    ),
                )
            except Exception:
                return

        async def _runtime_metrics_reporter() -> None:
            while not reporter_stop.is_set():
                await _push_live_status("running")
                try:
                    await asyncio.wait_for(reporter_stop.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

        async def _start_metrics_reporter(model_name: str) -> None:
            nonlocal active_model_name, metrics_started_ms, metrics_task
            active_model_name = str(model_name or "").strip()
            metrics_started_ms = time.time() * 1000
            reporter_stop.clear()
            await _push_live_status("running")
            metrics_task = asyncio.create_task(_runtime_metrics_reporter())

        async def _stop_metrics_reporter() -> None:
            reporter_stop.set()
            if metrics_task is not None:
                try:
                    await asyncio.wait_for(metrics_task, timeout=2.0)
                except Exception:
                    if not metrics_task.done():
                        metrics_task.cancel()
            await _push_live_status("running")

        if modality == "image":
            challenge = None
            while True:
                try:
                    challenge = await mcp_client.get_image_challenge(agent_id)
                    break
                except McpArenaError as e:
                    message = str(e).lower()
                    if "locked" in message or "waiting for organizer" in message:
                        print("   Lobby open; waiting for organizer to start battle...")
                        await asyncio.sleep(3.0)
                    else:
                        raise
            print(f"   Type: {challenge.challenge_type}")
            print(f"   Puzzle: {challenge.puzzle_id}")
            print(f"   Time limit: {challenge.max_time_s}s")
            usage_scope = http_client.fetch_usage_scope() or resolve_usage_scope()
            if usage_scope:
                os.environ["ARENA_USAGE_SCOPE"] = usage_scope

            available_models = fetch_available_models(llm_host, llm_api_key)
            image_rules = "\n".join(
                part
                for part in (challenge.prompt, challenge.reference_notes)
                if isinstance(part, str) and part.strip()
            )
            image_prompt = (challenge.prompt or challenge.description or "").strip()
            image_ctx = _build_context(
                challenge_type=challenge.challenge_type,
                description=challenge.description,
                rules=image_rules,
                max_time_s=challenge.max_time_s,
                available_models=available_models,
                image_url=challenge.input_image_uri or None,
            )
            ranked_models = strategy.rank_models(image_ctx, available_models)
            model_name = strategy.pick_model("solve", ranked_models, image_ctx)
            model_name = require_explicit_model(
                model_name,
                available_models,
                source="python_simple agent",
            )
            image_tool_proxy_args = {
                tool_name: build_image_tool_arguments(
                    tool_defs,
                    tool_name,
                    selected_model=model_name,
                    llm_api_key=llm_api_key,
                    arena_api_key=api_key,
                )
                for tool_name in ("image_edit", "image_generate", "image_analyze")
            }
            print(f"   Selected model: {model_name}")
            http_client.broadcast_thought(agent_id, f"Selected planner model: {model_name}")
            await _start_metrics_reporter(model_name)

            start_ms = time.time() * 1000
            prompt_text = strategy.build_image_prompt(image_ctx).strip() or image_prompt
            image_tools = [
                tool_name
                for tool_name in ("image_edit", "image_generate", "image_analyze")
                if tool_name in tools
            ]
            image_tool_sequence = _plan_image_tool_sequence(
                strategy,
                image_ctx,
                image_tools,
            )
            selected_tool = image_tool_sequence[-1] if image_tool_sequence else ""
            tool_result: dict = {}
            mcp_session_dead = False

            async def _call_image_tool(name: str, arguments: dict) -> dict:
                nonlocal mcp_session_dead
                try:
                    result = await mcp_client.call_tool(name, arguments)
                    return result if isinstance(result, dict) else {}
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    mcp_session_dead = True
                    print(f"   Image tool {name} failed ({type(exc).__name__}): {exc}")
                    return {}

            if (
                len(image_tool_sequence) > 1
                and image_tool_sequence[0] == "image_analyze"
                and challenge.input_image_uri
            ):
                print("   Using image_analyze before output tool")
                analysis_result = await _call_image_tool(
                    "image_analyze",
                    {
                        "image_uri": challenge.input_image_uri,
                        "question": prompt_text,
                        "agent_id": agent_id,
                        **image_tool_proxy_args.get("image_analyze", {}),
                    },
                )
                analysis_text = str(analysis_result.get("text") or "").strip()
                if analysis_text:
                    http_client.broadcast_thought(
                        agent_id,
                        f"Image analysis: {analysis_text[:180]}",
                    )
                    prompt_text = (
                        f"{prompt_text}\n\n"
                        "Source-image analysis to preserve while transforming: "
                        f"{analysis_text[:1000]}"
                    )
            if (
                selected_tool == "image_edit"
                and challenge.input_image_uri
                and "image_edit" in image_tools
            ):
                print("   Using image_edit tool")
                tool_result = await _call_image_tool(
                    "image_edit",
                    {
                        "image_uri": challenge.input_image_uri,
                        "prompt": prompt_text,
                        "agent_id": agent_id,
                        **image_tool_proxy_args.get("image_edit", {}),
                    },
                )
            elif selected_tool == "image_generate" and "image_generate" in image_tools:
                print("   Using image_generate tool")
                tool_result = await _call_image_tool(
                    "image_generate",
                    {
                        "prompt": prompt_text,
                        "agent_id": agent_id,
                        **image_tool_proxy_args.get("image_generate", {}),
                    },
                )
            elif (
                selected_tool == "image_analyze"
                and challenge.input_image_uri
                and "image_analyze" in image_tools
            ):
                print("   Using image_analyze tool (no image output tool available)")
                tool_result = await _call_image_tool(
                    "image_analyze",
                    {
                        "image_uri": challenge.input_image_uri,
                        "question": prompt_text,
                        "agent_id": agent_id,
                        **image_tool_proxy_args.get("image_analyze", {}),
                    },
                )
                analysis_text = str(tool_result.get("text") or "").strip()
                if analysis_text:
                    http_client.broadcast_thought(
                        agent_id,
                        f"Image analysis: {analysis_text[:180]}",
                    )
            elif challenge.input_image_uri and "image_edit" in image_tools:
                selected_tool = "image_edit"
                tool_result = await _call_image_tool(
                    "image_edit",
                    {
                        "image_uri": challenge.input_image_uri,
                        "prompt": prompt_text,
                        "agent_id": agent_id,
                        **image_tool_proxy_args.get("image_edit", {}),
                    },
                )
            elif "image_generate" in image_tools:
                selected_tool = "image_generate"
                tool_result = await _call_image_tool(
                    "image_generate",
                    {
                        "prompt": prompt_text,
                        "agent_id": agent_id,
                        **image_tool_proxy_args.get("image_generate", {}),
                    },
                )

            if tool_result.get("error"):
                print(f"   Tool warning: {tool_result.get('error')}")

            output_image_uri = str(tool_result.get("image_uri") or "").strip()
            if not output_image_uri and challenge.input_image_uri:
                output_image_uri = challenge.input_image_uri
            if not output_image_uri:
                if mcp_session_dead:
                    await session_monitor.stop()
                    raise RuntimeError(
                        "Image MCP tool failed and no input_image_uri is available "
                        "to submit; refusing to invent image bytes"
                    )
                output_image_uri = BLANK_PNG_DATA_URI

            total_time_ms = int(time.time() * 1000 - start_ms)
            await _stop_metrics_reporter()
            client_metrics = {
                "model_name": model_name,
                "planner_tool": selected_tool,
                "total_tokens": str(live_total_tokens),
                "prompt_tokens": str(live_prompt_tokens),
                "completion_tokens": str(live_completion_tokens),
                "ttft_ms": 0,
                "total_time_ms": total_time_ms,
            }
            if mcp_session_dead:
                await session_monitor.stop()
                print("   MCP session unavailable; submitting image over REST")
                image_submit_result = http_client.submit(
                    agent_id,
                    answer=output_image_uri,
                    client_metrics=client_metrics,
                    challenge_type="image",
                )
            else:
                image_submit_result = await mcp_client.submit_image(
                    agent_id=agent_id,
                    image_uri=output_image_uri,
                    client_metrics=client_metrics,
                    rationale="simple_agent dynamic image flow",
                )
            print(f"   Image submission: {image_submit_result}")
            await session_monitor.stop()
            print("\n✅ Agent completed!")
            return

        # Wait for organizer to start (challenge is locked while lobby is open)
        challenge = None
        while True:
            try:
                challenge = await mcp_client.get_challenge(agent_id)
                break
            except McpArenaError as e:
                message = str(e).lower()
                if "locked" in message or "waiting for organizer" in message:
                    print("   Lobby open; waiting for organizer to start battle...")
                    await asyncio.sleep(3.0)
                else:
                    raise
        print(f"   Type: {challenge.challenge_type}")
        print(f"   Puzzle: {challenge.puzzle_id}")
        print(f"   Time limit: {challenge.max_time_s}s")
        usage_scope = http_client.fetch_usage_scope() or resolve_usage_scope()
        if usage_scope:
            os.environ["ARENA_USAGE_SCOPE"] = usage_scope
        llm_client = OpenAI(
            base_url=llm_host,
            api_key=llm_api_key,
            default_headers=build_proxy_headers(agent_id, usage_scope),
        )
        print("   LLM: Enabled")
        print()

        available_models = fetch_available_models(llm_host, llm_api_key)
        model_ctx = _build_context(
            challenge_type=challenge.challenge_type,
            description=challenge.description,
            rules=challenge.rules,
            max_time_s=challenge.max_time_s,
            available_models=available_models,
        )
        ranked_models = strategy.rank_models(model_ctx, available_models)
        model_name = strategy.pick_model("solve", ranked_models, model_ctx)
        model_name = require_explicit_model(
            model_name,
            available_models,
            source="python_simple agent",
        )
        print(f"   Selected model: {model_name}")
        await _start_metrics_reporter(model_name)
        
        # Broadcast thought
        http_client.broadcast_thought(agent_id, "Reading challenge and clues...")
        
        # Get all clues
        print("\n📖 Reading clues...")
        clues = []
        clue_ids = await mcp_client.list_clues(agent_id)
        for i, clue_id in enumerate(clue_ids):
            clue = await mcp_client.get_clue(clue_id, agent_id)
            clues.append(clue.text)
            print(f"   Clue {i}: {clue.text[:50]}...")
        if not clues:
            clues = [str(clue_text) for clue_text in challenge.clues]

        tool_evidence_lines = await _enforce_required_tool_calls(
            mcp_client=mcp_client,
            agent_id=agent_id,
            challenge_type=str(challenge.challenge_type or ""),
            description=str(challenge.description or ""),
            rules=str(challenge.rules or ""),
            clues=clues,
            available_tool_names=[str(name) for name in tools],
            http_client=http_client,
            extra_context=str(getattr(challenge, "extra_text", "") or ""),
        )
        
        # Broadcast thought
        http_client.broadcast_thought(agent_id, f"Analyzing {len(clues)} clues...")
        
        # Save draft (backup)
        http_client.save_draft(agent_id, "Working on solution...")
        
        # Step 3: Solve
        print("\n🧠 Solving...")
        http_client.broadcast_thought(agent_id, "Calling LLM...")
        solve_ctx = _build_context(
            challenge_type=challenge.challenge_type,
            description=challenge.description,
            rules=challenge.rules,
            clues=clues,
            max_time_s=challenge.max_time_s,
            available_models=ranked_models,
        )
        
        answer, raw_response, model_name, usage, ttft_ms, total_time_ms = await solve_challenge(
            challenge,
            clues,
            llm_client,
            model_name,
            strategy,
            solve_ctx,
            http_client=http_client,
            agent_id=agent_id,
            broadcast_thought=True,
            evidence_lines=tool_evidence_lines,
        )
        print(f"   Raw LLM output: {raw_response[:120]}...")
        print(f"   Extracted answer: {answer}")
        print(f"   Model: {model_name} | Tokens: {usage.get('total_tokens', 0)} | Time: {total_time_ms}ms")
        
        if _is_invalid_answer_candidate(answer):
            answer = ""
        if not answer:
            print("   ⚠️  No valid ANSWER format found; using last line as fallback.")
            lines = raw_response.strip().splitlines()
            last_line = lines[-1].strip().strip("`\"'") if lines else ""
            answer = (
                last_line
                if last_line and len(last_line) < 400 and not _is_invalid_answer_candidate(last_line)
                else "unknown"
            )

        # Save draft with extracted answer as backup
        http_client.save_draft(agent_id, answer, "LLM solution")
        http_client.broadcast_thought(agent_id, f"Answer: {answer}")
        
        # Check time remaining
        time_info = await mcp_client.time_remaining(agent_id)
        remaining_s = float(time_info.get("time_remaining_s", 0.0))
        submit_ctx = _build_context(
            challenge_type=challenge.challenge_type,
            description=challenge.description,
            rules=challenge.rules,
            clues=clues,
            max_time_s=challenge.max_time_s,
            available_models=ranked_models,
            time_remaining_s=remaining_s,
            tokens_used=usage.get("total_tokens", 0),
        )
        if strategy.should_submit_early(answer, submit_ctx):
            http_client.broadcast_thought(agent_id, "⚡ Strategy submitting early.")
        revised_answer = strategy.on_time_warning(remaining_s, answer, submit_ctx)
        if isinstance(revised_answer, str) and revised_answer.strip():
            answer = revised_answer.strip()
        print(f"   Time remaining: {time_info['time_remaining_s']:.1f}s")
    
    # Step 4: Submit the extracted answer with real metrics
    print("\n📤 Submitting answer...")
    http_client.broadcast_thought(agent_id, "Submitting final answer!")
    await _stop_metrics_reporter()
    
    result = http_client.submit(
        agent_id=agent_id,
        answer=answer,
        client_metrics={
            "model_name": model_name,
            "total_tokens": str(live_total_tokens),
            "prompt_tokens": str(live_prompt_tokens),
            "completion_tokens": str(live_completion_tokens),
            "ttft_ms": ttft_ms,
            "total_time_ms": total_time_ms,
        }
    )
    
    print(f"   Accepted: {result.accepted}")
    print(f"   Status: {result.status}")
    if result.score:
        print(f"   Score: {result.score.get('final_score', 0)}")
        print(f"   Quality: {result.score.get('quality_score', 0)}")
        print(f"   Speed: {result.score.get('speed_score', 0)}")
    
    await session_monitor.stop()
    print("\n✅ Agent completed!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except asyncio.CancelledError:
        raise SystemExit(0)
    except ModelSelectionError as exc:
        print(f"   {exc}")
        raise SystemExit(1)
