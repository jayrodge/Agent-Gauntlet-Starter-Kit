#!/usr/bin/env python3
"""CrewAI example agent for Agent Gauntlet.

This example uses CrewAI native tools backed by the starter-kit MCP client
instead of CrewAI's direct MCP transport layer. That keeps the example aligned
with current CrewAI function-calling rules while preserving runtime tool
discovery and both text and image challenge flows.

Install dependencies first:

    pip install 'crewai[tools]' mcp

Usage:
    cd examples/crewai
    python agent.py
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

load_dotenv(REPO_ROOT / ".env")

from arena_clients import (
    HttpArenaClient,
    McpArenaClient,
    McpArenaError,
    build_image_tool_arguments,
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
    require_available_models,
)
from my_strategy import MyStrategy
from arena_tools import (
    ArenaToolState,
    ToolSpec,
    build_crewai_tools,
    classify_image_tool,
    discover_tool_specs,
    unsupported_required_fields,
)

BLANK_PNG_DATA_URI = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7/"
    "S7sAAAAASUVORK5CYII="
)
DEFAULT_AGENT_ID = "crewai-agent"
DEFAULT_AGENT_NAME = "CrewAI Agent"
STRATEGY = MyStrategy()


def _pin_crewai_model_env(model_name: str) -> None:
    """Pin explicit model env vars for CrewAI/LiteLLM internals."""
    model = str(model_name or "").strip()
    if not model:
        return
    os.environ["OPENAI_MODEL_NAME"] = model
    os.environ["MODEL"] = model
    os.environ["LITELLM_MODEL"] = model
    os.environ["LLM_MODEL"] = model
    os.environ["CREWAI_MODEL"] = model


def _make_crewai_llm(
    llm_cls,
    *,
    model: str,
    api_key: str,
    base_url: str,
    temperature: float,
    max_tokens: int,
    proxy_headers: dict[str, str],
    timeout_s: float | None = None,
):
    proxy_model = model if model.startswith("openai/") else f"openai/{model}"
    kwargs = {
        "model": proxy_model,
        "api_key": api_key,
        "base_url": base_url,
        "api_base": base_url,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if timeout_s is not None and timeout_s > 0:
        kwargs["timeout"] = timeout_s
    if proxy_headers:
        kwargs["extra_headers"] = proxy_headers
    try:
        return llm_cls(**kwargs)
    except TypeError as exc:
        if proxy_headers:
            raise RuntimeError(
                "Installed CrewAI/LiteLLM does not support proxy attribution headers. "
                "Upgrade CrewAI/LiteLLM before running with the Agent Gauntlet LLM proxy."
            ) from exc
        kwargs.pop("extra_headers", None)
        return llm_cls(**kwargs)


def _check_dependencies() -> bool:
    try:
        from crewai import Agent, Crew, LLM, Task
        from crewai.tools import BaseTool

        _ = (Agent, Crew, LLM, Task, BaseTool)
        return True
    except ImportError as exc:
        print("Missing CrewAI dependencies. Install with:")
        print("  pip install 'crewai[tools]' mcp")
        print(f"Error: {exc}")
        return False


def _extract_inline_answer_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    patterns = (
        r"(?:final answer(?: line)?(?: should be)?|answer(?: line)?(?: should be)?|output would be)\s*[:=]\s*[\"'`]?(.+?)[\"'`]?\s*$",
        r"(?:therefore|thus)\s*,?\s*the answer is\s*[\"'`]?(.+?)[\"'`]?\s*$",
    )
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        for pattern in patterns:
            match = re.search(pattern, line, flags=re.IGNORECASE)
            if not match:
                continue
            candidate = match.group(1).strip().strip("`\"' .")
            if candidate:
                candidates.append(candidate)
    for quoted in re.findall(r"[\"']([^\"'\n]{4,220})[\"']", text):
        candidate = quoted.strip().strip("`\"' .")
        if candidate:
            candidates.append(candidate)
    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _best_effort_single_line(text: str) -> str:
    best = ""
    best_score = -10
    for raw_line in text.splitlines():
        line = raw_line.strip().strip("`")
        if not line:
            continue
        lowered = line.lower()
        if lowered.startswith(("<think", "reasoning", "analysis", "action", "observation")):
            continue
        score = 0
        if "|" in line:
            score += 5
        if ";" in line:
            score += 4
        if "," in line:
            score += 2
        if len(line.split()) <= 24:
            score += 1
        if line.startswith(("-", "*", "#")):
            score -= 2
        if score > best_score:
            best = line
            best_score = score
    return best if best_score >= 1 else ""


def extract_answer(raw_response: str, rules: str = "") -> str:
    raw_text = str(raw_response or "")
    cleaned = re.sub(r"<think>.*?(?:</think>|$)", "", raw_text, flags=re.DOTALL).strip()
    for source in (cleaned, raw_text):
        if not source:
            continue
        for line in source.splitlines():
            match = re.match(
                r"^\s*(?:answer|final answer)\s*:\s*(.+?)\s*$",
                line,
                flags=re.IGNORECASE,
            )
            if match:
                answer = match.group(1).strip().strip("`\"'")
                if answer:
                    return answer
        for candidate in _extract_inline_answer_candidates(source):
            if candidate:
                return candidate
    fallback = _best_effort_single_line(cleaned) or _best_effort_single_line(raw_text)
    if fallback:
        return fallback.strip().strip("`\"'")
    return cleaned.splitlines()[0].strip().strip("`\"'") if cleaned else ""


def _is_invalid_answer_candidate(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return True
    lowered = text.lower()
    if lowered.startswith(("<toolcall", "toolcall", "<think", "reasoning:", "analysis:", "action:")):
        return True
    blocked_tokens = ("<toolcall", "</toolcall", '"tool_calls"', "function_call")
    return any(token in lowered for token in blocked_tokens)


def extract_image_uri(raw_response: str) -> str:
    patterns = (
        r"IMAGE_URI:\s*(\S+)",
        r'"image_uri"\s*:\s*"([^"]+)"',
        r"'image_uri'\s*:\s*'([^']+)'",
        r"(data:image/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, raw_response, re.IGNORECASE)
        if not match:
            continue
        candidate = match.group(1).strip().rstrip(".,)")
        if candidate.lower() in {"stored_by_runtime", "stored_by_tool", "runtime"}:
            return ""
        return candidate
    return ""


def extract_image_plan(raw_response: str) -> tuple[str, str, str]:
    tool_name = ""
    instruction_text = ""
    summary = ""
    for line in raw_response.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.upper().startswith("TOOL:"):
            tool_name = stripped.split(":", 1)[1].strip().lower()
        elif stripped.upper().startswith("INSTRUCTION:") or stripped.upper().startswith("PROMPT:"):
            instruction_text = stripped.split(":", 1)[1].strip()
        elif stripped.upper().startswith("SUMMARY:"):
            summary = stripped.split(":", 1)[1].strip()
    return tool_name, instruction_text, summary


def _repair_text_answer(
    *,
    raw_content: str,
    rules: str,
    llm_host: str,
    llm_api_key: str,
    repair_model: str,
    agent_id: str | None = None,
    usage_scope: str | None = None,
) -> str:
    """Best-effort formatter repair for empty/malformed CrewAI text output."""
    if not raw_content.strip():
        return ""

    try:
        from openai import OpenAI
    except Exception:
        return ""

    try:
        client = OpenAI(
            base_url=llm_host,
            api_key=llm_api_key,
            default_headers=build_proxy_headers(agent_id, usage_scope),
        )
        response = client.chat.completions.create(
            model=repair_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict final-answer formatter. "
                        "Return exactly one final answer line that follows the challenge rules. "
                        "Do not include reasoning, XML tags, JSON, markdown, or bullets."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Challenge rules:\n{rules}\n\n"
                        f"Model output:\n{raw_content}\n\n"
                        "Return the final answer now as exactly one line."
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=512,
        )
    except Exception:
        return ""

    repaired = (response.choices[0].message.content or "").strip()
    return extract_answer(repaired)


def _derive_answer_from_evidence(
    *,
    llm_host: str,
    llm_api_key: str,
    model_name: str,
    challenge_description: str,
    challenge_rules: str,
    evidence_lines: list[str],
    agent_id: str | None = None,
    usage_scope: str | None = None,
) -> str:
    if not evidence_lines:
        return ""
    try:
        from openai import OpenAI
    except Exception:
        return ""

    evidence_block = "\n".join(f"- {line}" for line in evidence_lines if line.strip())
    if not evidence_block:
        return ""

    try:
        client = OpenAI(
            base_url=llm_host,
            api_key=llm_api_key,
            default_headers=build_proxy_headers(agent_id, usage_scope),
        )
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Use the provided tool evidence to answer the question. "
                        "Return exactly one final answer line that follows the challenge rules."
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
            temperature=0.0,
            max_tokens=768,
        )
    except Exception:
        return ""

    raw = (response.choices[0].message.content or "").strip()
    answer = extract_answer(raw)
    if answer and not _is_invalid_answer_candidate(answer):
        return answer
    first_line = raw.splitlines()[0].strip().strip("`\"'") if raw.strip() else ""
    return "" if _is_invalid_answer_candidate(first_line) else first_line


def _direct_text_fallback_answer(
    *,
    llm_host: str,
    llm_api_key: str,
    model_name: str,
    challenge_description: str,
    challenge_rules: str,
    clues: list[str],
    agent_id: str | None = None,
    usage_scope: str | None = None,
) -> str:
    try:
        from openai import OpenAI
    except Exception:
        return ""

    clues_block = "\n".join(f"- {clue}" for clue in clues if clue.strip()) or "- (No clues provided.)"
    try:
        client = OpenAI(
            base_url=llm_host,
            api_key=llm_api_key,
            default_headers=build_proxy_headers(agent_id, usage_scope),
        )
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise challenge solver. "
                        "Return exactly one final answer line that follows rules. "
                        "Do not include reasoning, tags, or bullets."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Description: {challenge_description}\n"
                        f"Rules: {challenge_rules}\n\n"
                        f"Clues:\n{clues_block}\n\n"
                        "Return only the final answer line."
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=768,
        )
    except Exception:
        return ""

    raw = (response.choices[0].message.content or "").strip()
    answer = extract_answer(raw)
    if answer and not _is_invalid_answer_candidate(answer):
        return answer
    first_line = raw.splitlines()[0].strip().strip("`\"'") if raw else ""
    return "" if _is_invalid_answer_candidate(first_line) else first_line


def _looks_empty_raw_output(raw_output: str) -> bool:
    text = str(raw_output or "").strip()
    if not text:
        return True
    lowered = text.lower()
    if lowered in {"answer:", "final answer:", "none"}:
        return True
    if all(ch in ". \n\r\t" for ch in text):
        return True
    return False


def _extract_image_uri_from_tool_result(result: dict) -> str:
    if not isinstance(result, dict):
        return ""
    for field_name in ("image_uri", "output_image_uri", "edited_image", "data_uri"):
        image_uri = result.get(field_name)
        if isinstance(image_uri, str) and image_uri.strip():
            return image_uri.strip()
    return ""


def _build_context(
    *,
    challenge_type: str,
    description: str,
    rules: str,
    clues: list[str] | None = None,
    max_time_s: int,
    available_models: list[str],
    time_remaining_s: float = 0.0,
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
        available_models=available_models,
        tools_used=[],
        tokens_used=0,
        required_tools=[],
        image_url=image_url,
    )


def _planner_selection_context(
    selection_ctx: ChallengeContext,
    modality: str,
) -> ChallengeContext:
    if modality.strip().lower() != "image":
        return selection_ctx
    return replace(selection_ctx, challenge_type="tool-orchestration", image_url=None)


def _challenge_rules_text(challenge: object, modality: str) -> str:
    if modality == "text":
        rules = str(getattr(challenge, "rules", "") or "").strip()
        return rules or str(getattr(challenge, "description", "") or "").strip()
    return "\n".join(
        part
        for part in (
            getattr(challenge, "prompt", ""),
            getattr(challenge, "reference_notes", ""),
            getattr(challenge, "description", ""),
        )
        if isinstance(part, str) and part.strip()
    )


def _normalize_tool_key(value: str) -> str:
    return str(value or "").strip().lower()


def _is_runtime_control_tool(tool_name: str) -> bool:
    lower = _normalize_tool_key(tool_name)
    if not lower.startswith("arena."):
        return False
    return (
        "get_challenge" in lower
        or "broadcast_thought" in lower
        or ".submit" in lower
    )


def _order_image_tool_specs(
    image_tool_specs: list[ToolSpec],
    *,
    has_input_image: bool,
) -> list[ToolSpec]:
    kind_priority = (
        {"edit": 0, "generate": 1, "analyze": 2, "other": 3, "none": 4}
        if has_input_image
        else {"generate": 0, "edit": 1, "analyze": 2, "other": 3, "none": 4}
    )
    return sorted(
        image_tool_specs,
        key=lambda spec: (
            kind_priority.get(classify_image_tool(spec), 9),
            spec.original_name.lower(),
        ),
    )


def _build_image_tool_selection_map(
    image_tool_specs: list[ToolSpec],
) -> tuple[list[str], dict[str, ToolSpec]]:
    selection_choices: list[str] = []
    selection_map: dict[str, ToolSpec] = {}

    def register(key: str, spec: ToolSpec) -> None:
        normalized = _normalize_tool_key(key)
        if not normalized or normalized in selection_map:
            return
        selection_map[normalized] = spec
        selection_choices.append(key)

    first_by_kind: dict[str, ToolSpec] = {}
    for spec in image_tool_specs:
        kind = classify_image_tool(spec)
        if kind in {"edit", "generate", "analyze"} and kind not in first_by_kind:
            first_by_kind[kind] = spec

    for kind, alias in (
        ("edit", "image_edit"),
        ("generate", "image_generate"),
        ("analyze", "image_analyze"),
    ):
        spec = first_by_kind.get(kind)
        if spec:
            register(alias, spec)

    for spec in image_tool_specs:
        register(spec.original_name, spec)
        register(spec.sanitized_name, spec)

    return selection_choices, selection_map


def _choose_image_tool_spec(
    image_ctx: ChallengeContext,
    image_tool_specs: list[ToolSpec],
    planned_tool: str,
) -> ToolSpec | None:
    if not image_tool_specs:
        return None

    strategy_choices, selection_map = _build_image_tool_selection_map(image_tool_specs)
    planned_spec = selection_map.get(_normalize_tool_key(planned_tool))
    if planned_spec:
        return planned_spec

    strategy_choice = STRATEGY.plan_image_tool(image_ctx, strategy_choices)
    strategy_spec = selection_map.get(_normalize_tool_key(strategy_choice))
    if strategy_spec:
        return strategy_spec

    return image_tool_specs[0]


def _describe_image_tool(spec: ToolSpec) -> str:
    kind = classify_image_tool(spec)
    kind_label = {
        "edit": "edits an existing image",
        "generate": "generates an image from text",
        "analyze": "analyzes an input image",
        "other": "handles image-related data",
    }.get(kind, "image-related")
    parts = [kind_label]
    if spec.runtime_hints.image_input_field:
        parts.append(f"image input via `{spec.runtime_hints.image_input_field}`")
    if spec.instruction_field:
        parts.append(f"instruction via `{spec.instruction_field}`")
    return "; ".join(parts)


def _dedupe_models(
    candidates: list[str],
    *,
    available_models: list[str],
) -> list[str]:
    allowed = set(available_models)
    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = str(candidate or "").strip()
        if not normalized:
            continue
        if normalized.lower() == "default":
            continue
        if available_models and normalized not in allowed:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _build_candidate_models(
    *,
    strategy_model: str,
    selector_model: str,
    ranked_models: list[str],
    available_models: list[str],
) -> list[str]:
    ordered: list[str] = []
    ordered.extend([strategy_model, selector_model])
    ordered.extend(ranked_models)
    ordered.extend(available_models)
    return _dedupe_models(ordered, available_models=available_models)


def _build_planner_candidate_models(
    *,
    modality: str,
    strategy_model: str,
    selector_model: str,
    ranked_models: list[str],
    available_models: list[str],
) -> list[str]:
    candidates = _build_candidate_models(
        strategy_model=strategy_model,
        selector_model=selector_model,
        ranked_models=ranked_models,
        available_models=available_models,
    )
    if modality.strip().lower() != "image":
        return candidates
    return [
        model_name
        for model_name in candidates
        if "image" not in model_name.casefold()
    ]


def _crewai_execution_limits(modality: str, max_time_s: int) -> tuple[int, int]:
    normalized_time_s = max(0, int(max_time_s or 0))
    if modality.strip().lower() == "image":
        half_budget_s = normalized_time_s // 2 if normalized_time_s else 45
        return 2, max(15, min(45, half_budget_s))
    return 5, max(30, normalized_time_s)


def _native_crewai_tools_enabled(modality: str) -> bool:
    return modality.strip().lower() != "image"


def _planner_failure_fallback(modality: str) -> str | None:
    return "" if modality.strip().lower() == "image" else None


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
    deduped: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    for text in [description, rules, extra_context, *(clues or [])]:
        if not isinstance(text, str):
            continue
        for url in re.findall(r"https?://[^\s)>\"]+", text):
            cleaned = url.strip().rstrip(".,);")
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(cleaned)
    return deduped[:7]


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


async def _enforce_required_tool_calls(
    *,
    arena_mcp: McpArenaClient,
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

    query_candidates = _query_candidates(description, rules, clues, extra_context)
    if not query_candidates:
        query_candidates = [_best_search_query(description, rules, clues)]
    if not query_candidates:
        query_candidates = [""]
    evidence_lines: list[str] = []
    for tool_name in required_runtime_tools:
        is_search_like = "search" in tool_name.lower()
        target_calls = min(len(query_candidates), 3) if is_search_like else 1
        calls_made = 0
        for query_text in query_candidates[:target_calls]:
            payload_variants = [
                {"agent_id": agent_id, "query": query_text},
                {"query": query_text},
                {"agent_id": agent_id, "search_query": query_text},
                {"search_query": query_text},
                {"agent_id": agent_id, "text": query_text},
                {"text": query_text},
                {"agent_id": agent_id, "url": query_text},
                {"url": query_text},
                {"agent_id": agent_id, "video_url": query_text},
                {"video_url": query_text},
                {"agent_id": agent_id},
                {},
            ]
            for payload in payload_variants:
                try:
                    result = await arena_mcp.call_tool(tool_name, payload)
                    rendered = _tool_result_text(result)
                    if rendered:
                        evidence_lines.append(f"{tool_name}: {rendered[:2000]}")
                    calls_made += 1
                    break
                except Exception:
                    continue
        note = (
            f"Required tool coverage: {tool_name} ({calls_made}/{target_calls})"
            if target_calls > 1
            else f"Required tool called before solve: {tool_name}"
            if calls_made
            else f"Could not call required tool: {tool_name}"
        )
        try:
            await asyncio.to_thread(http_client.broadcast_thought, agent_id, note)
        except Exception:
            pass
    return evidence_lines


async def _wait_for_start_gate(http_client: HttpArenaClient, agent_id: str) -> None:
    await asyncio.to_thread(http_client.update_status, agent_id, "ready")
    await asyncio.to_thread(
        http_client.broadcast_thought,
        agent_id,
        "Connected to Agent Gauntlet",
    )
    print("   Connected to Agent Gauntlet")

    last_phase: str | None = None
    last_countdown: object | None = None
    waiting_for_next_round = False

    while True:
        try:
            competition = await asyncio.to_thread(http_client.get_competition)
        except Exception:
            print("   Competition endpoint unavailable; proceeding without start gate.")
            return

        phase = str(competition.get("phase") or "").lower()
        countdown_value = competition.get("countdown_value")
        eligible_agent_ids = competition.get("eligible_agent_ids")

        eligible_for_current_round = True
        if isinstance(eligible_agent_ids, list):
            eligible_set = {
                str(value)
                for value in eligible_agent_ids
                if isinstance(value, str) and value.strip()
            }
            if eligible_set:
                eligible_for_current_round = agent_id in eligible_set

        if phase == "running":
            if not eligible_for_current_round:
                if not waiting_for_next_round:
                    print("   Battle already running. Waiting for next organizer start...")
                    await asyncio.to_thread(
                        http_client.broadcast_thought,
                        agent_id,
                        "Battle already running. Waiting for next battle.",
                    )
                    waiting_for_next_round = True
                await asyncio.sleep(1.0)
                continue
            print("   GO - challenge unlocked")
            await asyncio.to_thread(
                http_client.broadcast_thought,
                agent_id,
                "GO - challenge unlocked",
            )
            return

        waiting_for_next_round = False
        if phase == "countdown":
            if countdown_value != last_countdown:
                print(f"   Countdown: {countdown_value}")
                await asyncio.to_thread(
                    http_client.broadcast_thought,
                    agent_id,
                    f"Countdown: {countdown_value}",
                )
                last_countdown = countdown_value
        elif phase != last_phase:
            print("   Waiting for organizer start...")
            await asyncio.to_thread(
                http_client.broadcast_thought,
                agent_id,
                "Waiting for organizer start",
            )
            last_phase = phase

        await asyncio.sleep(1.0)


async def _fetch_challenge(arena_mcp: McpArenaClient, modality: str, agent_id: str):
    while True:
        try:
            if modality == "image":
                return await arena_mcp.get_image_challenge(agent_id)
            return await arena_mcp.get_challenge(agent_id)
        except McpArenaError as exc:
            message = str(exc).lower()
            if "locked" in message or "waiting for organizer" in message:
                print("   Waiting for organizer start...")
                await asyncio.sleep(1.0)
                continue
            raise


def _build_text_task_description(
    challenge,
    text_ctx: ChallengeContext,
    available_capabilities: list[str],
) -> str:
    clue_preview = "\n".join(
        f"- {clue}"
        for clue in (challenge.clues or [])
        if isinstance(clue, str) and clue.strip()
    )
    if not clue_preview:
        clue_preview = "- (No clues provided.)"

    challenge_type = str(challenge.challenge_type or "").lower()
    rules_lower = str(challenge.rules or "").lower()
    challenge_text_lower = " ".join(
        [
            str(challenge.challenge_type or ""),
            str(challenge.description or ""),
            str(challenge.rules or ""),
        ]
    ).lower()
    required_tool_hint = ""
    if challenge_type in {"web-search", "market-research"} or "search" in rules_lower:
        required_tool_hint += "- Call the search tool at least once before your final answer.\n"
    if challenge_type == "youtube-transcript" or "transcript" in rules_lower:
        required_tool_hint += "- Use the transcript tool when relevant to gather the answer.\n"
    mandatory_tools: list[str] = []
    for tool_name in available_capabilities:
        normalized = str(tool_name or "").strip()
        if not normalized:
            continue
        if normalized.lower() in challenge_text_lower and normalized not in mandatory_tools:
            mandatory_tools.append(normalized)
    if challenge_type in {"web-search", "market-research"}:
        for tool_name in available_capabilities:
            normalized = str(tool_name or "").strip()
            if "search" in normalized.lower() and normalized not in mandatory_tools:
                mandatory_tools.append(normalized)
    for tool_name in mandatory_tools:
        required_tool_hint += (
            f"- MANDATORY: If `{tool_name}` is available, call it at least once "
            "before your final answer.\n"
        )

    strategy_notes = str(getattr(STRATEGY, "text_strategy_notes", "") or "").strip()
    strategy_block = f"Additional strategy notes:\n{strategy_notes}\n\n" if strategy_notes else ""
    available_capabilities_text = ", ".join(available_capabilities) or "no additional tools"

    return (
        f"{strategy_block}"
        f"{STRATEGY.build_solver_prompt(text_ctx)}\n\n"
        f"Available runtime capabilities: {available_capabilities_text}\n\n"
        f"Execution requirements:\n"
        f"{required_tool_hint}"
        f"- Use the available tools when they improve answer quality.\n"
        f"- Make sure every clue is satisfied before you finalize the ordering.\n"
        f"- Keep reasoning extremely short (3 sentences max).\n"
        f"- Output the final answer on its own line exactly as: ANSWER: <your answer>\n"
        f"- If the rules demand a stricter one-line output, follow them exactly after the ANSWER label.\n"
        f"- The ANSWER line is mandatory.\n"
    )


def _build_image_task_description(
    challenge,
    image_ctx: ChallengeContext,
    planning_tool_names: list[str],
    image_tool_specs: list[ToolSpec],
) -> str:
    image_strategy_notes = str(getattr(STRATEGY, "image_strategy_notes", "") or "").strip()
    strategy_block = f"Image strategy notes:\n{image_strategy_notes}\n\n" if image_strategy_notes else ""
    strategy_choices, selection_map = _build_image_tool_selection_map(image_tool_specs)
    preferred_choice = STRATEGY.plan_image_tool(image_ctx, strategy_choices)
    preferred_tool = selection_map.get(_normalize_tool_key(preferred_choice))
    strategy_prompt = STRATEGY.build_image_prompt(image_ctx).strip() or (
        getattr(challenge, "prompt", "") or challenge.description or ""
    ).strip()
    available_capabilities = ", ".join(planning_tool_names) or "no additional planning tools"
    action_tool_lines = "\n".join(
        f"- {spec.original_name}: {_describe_image_tool(spec)}"
        for spec in image_tool_specs
    )
    if not action_tool_lines:
        action_tool_lines = "- No executable image-producing tool was inferred from the discovered schemas."
    preferred_tool_name = preferred_tool.original_name if preferred_tool else (preferred_choice or "auto")
    return (
        f"{strategy_block}"
        f"Image planning hint:\n{strategy_prompt}\n\n"
        f"Challenge snapshot:\n"
        f"- Type: {challenge.challenge_type}\n"
        f"- Description: {challenge.description}\n"
        f"- Prompt: {getattr(challenge, 'prompt', '')}\n"
        f"- Reference notes: {getattr(challenge, 'reference_notes', '')}\n"
        f"- Available planning tools: {available_capabilities}\n"
        f"- Final image-capable tools:\n{action_tool_lines}\n\n"
        f"Execution requirements:\n"
        f"- Preferred final image tool: {preferred_tool_name}\n"
        f"- Your job is to PLAN the best image action, not to submit it directly.\n"
        f"- The runtime will execute the final image tool call and submit the result after you respond.\n"
        f"- If extra planning tools are available and helpful, you may use them before deciding.\n"
        f"- Return exactly these lines:\n"
        f"TOOL: <exact tool name from the discovered final image-capable list>\n"
        f"INSTRUCTION: <text to place in that tool's main instruction field>\n"
        f"SUMMARY: <one short sentence>\n"
    )


def _extract_usage_metrics(result: object) -> dict[str, int]:
    usage = getattr(result, "token_usage", None)
    if usage is None:
        return {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}
    return {
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
    }


async def _resolve_image_candidate(
    *,
    initial_image_uri: str,
    tool_state: ArenaToolState,
    challenge,
    image_tool_specs: list[ToolSpec],
    ranked_models: list[str],
    mcp_url: str,
    api_key: str | None,
    agent_id: str,
    planned_tool: str = "",
    planned_instruction: str = "",
) -> tuple[str, str]:
    image_uri = initial_image_uri.strip()
    if not image_uri and tool_state.latest_image_uri:
        image_uri = tool_state.latest_image_uri
    selected_tool = tool_state.last_image_tool or "framework-output"
    if image_uri:
        return image_uri, selected_tool

    ordered_image_specs = _order_image_tool_specs(
        image_tool_specs,
        has_input_image=bool(getattr(challenge, "input_image_uri", "")),
    )
    image_ctx = _build_context(
        challenge_type=challenge.challenge_type,
        description=challenge.description,
        rules=_challenge_rules_text(challenge, "image"),
        max_time_s=challenge.max_time_s,
        available_models=ranked_models,
        image_url=getattr(challenge, "input_image_uri", None),
    )
    preferred_spec = _choose_image_tool_spec(image_ctx, ordered_image_specs, planned_tool)
    prompt_text = planned_instruction.strip() or STRATEGY.build_image_prompt(image_ctx).strip() or (
        getattr(challenge, "prompt", "") or challenge.description or ""
    ).strip()

    candidate_specs: list[ToolSpec] = []
    if preferred_spec:
        candidate_specs.append(preferred_spec)
    for spec in ordered_image_specs:
        if preferred_spec and spec.original_name == preferred_spec.original_name:
            continue
        candidate_specs.append(spec)

    tool_result: dict[str, object] = {}
    for spec in candidate_specs:
        payload: dict[str, object] = {}
        payload.update(tool_state.tool_argument_overrides.get(spec.original_name, {}))
        if spec.runtime_hints.accepts_agent_id:
            payload["agent_id"] = agent_id
        if spec.runtime_hints.image_input_field:
            challenge_image_uri = str(getattr(challenge, "input_image_uri", "") or "").strip()
            fallback_image_uri = tool_state.current_image_uri()
            image_input_uri = challenge_image_uri or fallback_image_uri
            if image_input_uri:
                payload[spec.runtime_hints.image_input_field] = image_input_uri
        if spec.instruction_field and prompt_text:
            payload[spec.instruction_field] = prompt_text

        try:
            async with McpArenaClient(mcp_url, api_key) as submit_mcp:
                tool_result = await submit_mcp.call_tool(spec.original_name, payload)
        except Exception as exc:
            print(f"   Image tool '{spec.original_name}' failed: {exc}")
            continue

        tool_state.record_result(spec.original_name, tool_result)
        image_uri = tool_state.latest_image_uri or _extract_image_uri_from_tool_result(tool_result)
        selected_tool = spec.original_name
        if image_uri:
            return image_uri, selected_tool

    if not image_uri and getattr(challenge, "input_image_uri", ""):
        image_uri = str(getattr(challenge, "input_image_uri")).strip()
    if not image_uri:
        image_uri = BLANK_PNG_DATA_URI
    return image_uri, selected_tool


async def main() -> int:
    if not _check_dependencies():
        return 1

    from crewai import Agent as CrewAgent, Crew, LLM, Task

    ensure_connected()

    api_base = get_api_base()
    mcp_url = get_mcp_url()
    llm_host = get_proxy_host()
    api_key = get_arena_api_key()
    llm_api_key = api_key

    agent_id = (
        os.getenv("AGENT_ID")
        or str(getattr(STRATEGY, "agent_id", "")).strip()
        or DEFAULT_AGENT_ID
    )
    agent_name = (
        os.getenv("AGENT_NAME")
        or str(getattr(STRATEGY, "agent_name", "")).strip()
        or DEFAULT_AGENT_NAME
    )
    os.environ["AGENT_ID"] = agent_id
    os.environ["AGENT_NAME"] = agent_name

    print(f"  {agent_name} starting...")
    print(f"   Agent ID: {agent_id}")
    print(f"   API: {api_base}")
    print(f"   MCP: {mcp_url}")
    print(f"   LLM: {llm_host}")
    print()

    http_client = HttpArenaClient(api_base=api_base, api_key=api_key, timeout=90.0)

    print("Registering with Agent Gauntlet...")
    session = await asyncio.to_thread(http_client.register, agent_id, agent_name)
    print(f"   Session: {session.session_id}")
    print(f"   Status: {session.status}")
    print()
    session_monitor = monitor_session(http_client, agent_id).start()

    await _wait_for_start_gate(http_client, agent_id)
    usage_scope = await asyncio.to_thread(http_client.fetch_usage_scope) or resolve_usage_scope()
    if usage_scope:
        os.environ["ARENA_USAGE_SCOPE"] = usage_scope
    proxy_headers = build_proxy_headers(agent_id, usage_scope)
    print()

    print("Discovering runtime tools...")
    required_tool_evidence_lines: list[str] = []
    clue_texts: list[str] = []
    async with McpArenaClient(mcp_url, api_key) as arena_mcp:
        discovered_tools = await arena_mcp.list_tools()
        tool_defs = await arena_mcp.list_tool_defs()
        modality = McpArenaClient.detect_modality(discovered_tools)
        challenge = await _fetch_challenge(arena_mcp, modality, agent_id)
        if modality == "text":
            try:
                clue_texts = await arena_mcp.get_all_clue_texts(agent_id)
            except McpArenaError:
                clue_texts = []
            if not clue_texts:
                clue_texts = list(getattr(challenge, "clues", []) or [])
            required_tool_evidence_lines = await _enforce_required_tool_calls(
                arena_mcp=arena_mcp,
                agent_id=agent_id,
                challenge_type=str(challenge.challenge_type or ""),
                description=str(challenge.description or ""),
                rules=str(getattr(challenge, "rules", "") or ""),
                clues=clue_texts,
                available_tool_names=list(discovered_tools or []),
                http_client=http_client,
                extra_context=str(getattr(challenge, "extra_text", "") or ""),
            )

    all_tool_specs = discover_tool_specs(tool_defs)
    tool_argument_overrides: dict[str, dict[str, str]] = {}
    control_tool_names = {
        spec.original_name
        for spec in all_tool_specs
        if _is_runtime_control_tool(spec.original_name)
    }
    executable_image_tool_specs = [
        spec
        for spec in all_tool_specs
        if classify_image_tool(spec) in {"edit", "generate"}
        and not unsupported_required_fields(spec)
    ]
    if not executable_image_tool_specs:
        executable_image_tool_specs = [
            spec
            for spec in all_tool_specs
            if spec.image_related and not unsupported_required_fields(spec)
        ]

    challenge_rules_text = _challenge_rules_text(challenge, modality)
    print(f"   Modality: {modality}")
    print(f"   Challenge type: {challenge.challenge_type}")
    print(f"   Puzzle: {challenge.puzzle_id}")
    print(f"   Time limit: {challenge.max_time_s}s")
    print()

    available_models = require_available_models(fetch_available_models(llm_host, llm_api_key))
    selection_ctx = _build_context(
        challenge_type=challenge.challenge_type,
        description=challenge.description,
        rules=challenge_rules_text,
        clues=clue_texts,
        max_time_s=challenge.max_time_s,
        available_models=available_models,
        time_remaining_s=float(getattr(challenge, "time_remaining_s", 0.0) or 0.0),
        image_url=getattr(challenge, "input_image_uri", None),
    )
    ranked_models = STRATEGY.rank_models(selection_ctx, available_models)
    planner_context = _planner_selection_context(selection_ctx, modality)
    planner_ranked_models = STRATEGY.rank_models(planner_context, available_models)
    strategy_model = STRATEGY.pick_model(
        "solve", planner_ranked_models, planner_context
    )
    selector_model = ""
    candidate_models = _build_planner_candidate_models(
        modality=modality,
        strategy_model=strategy_model,
        selector_model=selector_model,
        ranked_models=planner_ranked_models,
        available_models=available_models,
    )
    if not candidate_models:
        raise ModelSelectionError(
            "CrewAI agent did not choose a valid model. Override `pick_model()` "
            "or ensure `rank_models()` returns at least one valid alias from "
            "the proxy roster."
        )
    tool_argument_overrides = {
        tool_name: build_image_tool_arguments(
            tool_defs,
            tool_name,
            selected_model=(
                ranked_models[0] if ranked_models else candidate_models[0]
            ),
            llm_api_key=llm_api_key,
            arena_api_key=api_key,
        )
        for tool_name in ("image_edit", "image_generate", "image_analyze")
    }
    print(f"   Candidate models: {', '.join(candidate_models)}")
    await asyncio.to_thread(
        http_client.broadcast_thought,
        agent_id,
        f"Candidate models: {', '.join(candidate_models[:4])}",
    )

    llm_params = STRATEGY.get_llm_params(selection_ctx)
    llm_temperature = float(llm_params.get("temperature", 0.0) or 0.0)
    llm_max_tokens = int(llm_params.get("max_tokens", 3072) or 3072)

    if modality == "text":
        excluded_tools = set(control_tool_names)
        excluded_tools.update(
            spec.original_name
            for spec in all_tool_specs
            if spec.image_related
        )
    else:
        excluded_tools = set(control_tool_names)
        excluded_tools.update(spec.original_name for spec in executable_image_tool_specs)
    crew_tool_specs = [
        spec
        for spec in all_tool_specs
        if spec.original_name not in excluded_tools
    ]
    if not _native_crewai_tools_enabled(modality):
        crew_tool_specs = []
        excluded_tools.update(spec.original_name for spec in all_tool_specs)
    crew_tools, tool_state = build_crewai_tools(
        tool_defs,
        agent_id=agent_id,
        mcp_url=mcp_url,
        api_key=api_key,
        challenge_image_uri=str(getattr(challenge, "input_image_uri", "") or ""),
        tool_argument_overrides=tool_argument_overrides,
        exclude_tools=excluded_tools,
    )
    tool_listing = ", ".join(
        f"{tool.name}->{tool_state.tool_name_map.get(tool.name, '?')}"
        for tool in crew_tools
    )
    print(f"   CrewAI tools: {tool_listing or '(none)'}")
    print()

    if modality == "text":
        task_description = _build_text_task_description(
            challenge,
            selection_ctx,
            [spec.original_name for spec in crew_tool_specs],
        )
        expected_output = "Return the final answer as: ANSWER: <your answer>"
    else:
        task_description = _build_image_task_description(
            challenge,
            selection_ctx,
            [spec.original_name for spec in crew_tool_specs],
            executable_image_tool_specs,
        )
        expected_output = "Return TOOL, INSTRUCTION, and SUMMARY lines."

    print("Solving with CrewAI native tools...")
    await asyncio.to_thread(http_client.update_status, agent_id, "thinking")
    await asyncio.to_thread(
        http_client.broadcast_thought,
        agent_id,
        "Starting CrewAI agent with native Agent Gauntlet tools...",
    )

    start_ms = time.time() * 1000
    result = None
    active_model_name = ""
    last_error: Exception | None = None
    max_iterations, planner_budget_s = _crewai_execution_limits(
        modality,
        int(challenge.max_time_s or 0),
    )
    planner_deadline = time.monotonic() + planner_budget_s
    for attempt_index, candidate_model in enumerate(candidate_models, start=1):
        remaining_planner_s = max(0, int(planner_deadline - time.monotonic()))
        if remaining_planner_s < 5:
            last_error = TimeoutError(
                f"CrewAI planner budget exhausted after {planner_budget_s}s."
            )
            break
        active_model_name = candidate_model
        _pin_crewai_model_env(candidate_model)
        if attempt_index == 1:
            print(f"   Starting with model: {candidate_model}")
            await asyncio.to_thread(
                http_client.broadcast_thought,
                agent_id,
                f"selected model: {candidate_model}",
            )
        else:
            print(f"   Retrying with model: {candidate_model}")
            await asyncio.to_thread(
                http_client.broadcast_thought,
                agent_id,
                f"Retrying with model: {candidate_model}",
            )

        llm = _make_crewai_llm(
            LLM,
            model=candidate_model,
            api_key=llm_api_key,
            base_url=llm_host,
            temperature=llm_temperature,
            max_tokens=llm_max_tokens,
            proxy_headers=proxy_headers,
            timeout_s=float(remaining_planner_s),
        )
        solver_agent = CrewAgent(
            role="Arena Challenge Solver",
            goal="Solve the current Agent Gauntlet challenge accurately and quickly.",
            backstory=(
                "You are a competitive AI agent solving timed Arena challenges. "
                "Use tools selectively, keep reasoning concise, and respect the required answer format."
            ),
            llm=llm,
            function_calling_llm=llm,
            tools=crew_tools,
            verbose=True,
            allow_delegation=False,
            max_iter=max_iterations,
            max_execution_time=remaining_planner_s,
        )
        solve_task = Task(
            description=task_description,
            expected_output=expected_output,
            agent=solver_agent,
        )
        crew = Crew(agents=[solver_agent], tasks=[solve_task], verbose=True)
        try:
            result = await crew.kickoff_async()
            if modality == "text":
                raw_attempt = (getattr(result, "raw", str(result)) or "").strip()
                if _looks_empty_raw_output(raw_attempt):
                    last_error = RuntimeError(
                        f"CrewAI returned empty raw output for model '{candidate_model}'."
                    )
                    print(f"   Model '{candidate_model}' produced empty output; trying next model.")
                    if attempt_index < len(candidate_models):
                        continue
                    result = None
            break
        except Exception as exc:
            last_error = exc
            print(f"   Model '{candidate_model}' failed: {exc}")
            if "model 'default' not allowed" in str(exc).lower():
                await asyncio.to_thread(
                    http_client.broadcast_thought,
                    agent_id,
                    f"Model routing fallback hit 'default' for {candidate_model}; retrying with pinned model env.",
                )
            if attempt_index < len(candidate_models):
                continue

    if result is None:
        final_error = last_error or RuntimeError("CrewAI solve failed for all candidate models.")
        # Generic fallback path for text challenges if CrewAI orchestration fails.
        if modality == "text":
            fallback_model = active_model_name or candidate_models[0]
            answer = ""
            if required_tool_evidence_lines:
                answer = await asyncio.to_thread(
                    _derive_answer_from_evidence,
                    llm_host=llm_host,
                    llm_api_key=llm_api_key,
                    model_name=fallback_model,
                    challenge_description=str(challenge.description or ""),
                    challenge_rules=str(getattr(challenge, "rules", "") or ""),
                    evidence_lines=required_tool_evidence_lines,
                    agent_id=agent_id,
                    usage_scope=usage_scope,
                )
            if _is_invalid_answer_candidate(answer):
                answer = ""
            if not answer:
                answer = await asyncio.to_thread(
                    _direct_text_fallback_answer,
                    llm_host=llm_host,
                    llm_api_key=llm_api_key,
                    model_name=fallback_model,
                    challenge_description=str(challenge.description or ""),
                    challenge_rules=str(getattr(challenge, "rules", "") or ""),
                    clues=clue_texts,
                    agent_id=agent_id,
                    usage_scope=usage_scope,
                )
            if _is_invalid_answer_candidate(answer):
                answer = ""
            if not answer:
                answer = "unknown"
            total_time_ms = int(time.time() * 1000 - start_ms)
            await asyncio.to_thread(
                http_client.broadcast_thought,
                agent_id,
                f"CrewAI orchestration failed; submitting fallback answer from evidence path. Error: {final_error}",
            )
            submit_result = await asyncio.to_thread(
                http_client.submit,
                agent_id,
                answer,
                {
                    "model_name": fallback_model,
                    "total_tokens": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_time_ms": total_time_ms,
                },
                "text",
            )
            print(f"   Accepted: {submit_result.accepted}")
            print(f"   Status: {submit_result.status}")
            if submit_result.score:
                print(f"   Score: {submit_result.score.get('final_score', 0)}")
            await asyncio.to_thread(http_client.update_status, agent_id, "submitted")
            await session_monitor.stop()
            return 0
        fallback_raw = _planner_failure_fallback(modality)
        if fallback_raw is None:
            await asyncio.to_thread(http_client.update_status, agent_id, "failed")
            await asyncio.to_thread(
                http_client.broadcast_thought,
                agent_id,
                f"CrewAI solve failed: {final_error}",
            )
            raise final_error
        print(
            "   CrewAI image planner failed; using deterministic tool fallback: "
            f"{final_error}"
        )
        await asyncio.to_thread(
            http_client.broadcast_thought,
            agent_id,
            "CrewAI image planner failed; continuing with dedicated image tool: "
            f"{final_error}",
        )
        result = fallback_raw

    total_time_ms = int(time.time() * 1000 - start_ms)
    raw_content = getattr(result, "raw", str(result))
    usage_metrics = _extract_usage_metrics(result)
    print(f"   Solved with model: {active_model_name}")
    print(f"   Raw output: {raw_content[:200]}...")
    print(f"   Total time: {total_time_ms}ms")
    print(f"   Tokens: {usage_metrics['total_tokens']}")
    print()

    if modality == "text":
        answer = extract_answer(raw_content)
        if _is_invalid_answer_candidate(answer):
            answer = ""
        if not answer and required_tool_evidence_lines:
            answer = await asyncio.to_thread(
                _derive_answer_from_evidence,
                llm_host=llm_host,
                llm_api_key=llm_api_key,
                model_name=active_model_name or candidate_models[0],
                challenge_description=str(challenge.description or ""),
                challenge_rules=str(getattr(challenge, "rules", "") or ""),
                evidence_lines=required_tool_evidence_lines,
                agent_id=agent_id,
                usage_scope=usage_scope,
            )
        if _is_invalid_answer_candidate(answer):
            answer = ""
        if not answer:
            print("   Extraction failed; retrying with strict formatter...")
            repair_model = active_model_name or candidate_models[0]
            answer = await asyncio.to_thread(
                _repair_text_answer,
                raw_content=raw_content,
                rules=str(getattr(challenge, "rules", "") or ""),
                llm_host=llm_host,
                llm_api_key=llm_api_key,
                repair_model=repair_model,
                agent_id=agent_id,
                usage_scope=usage_scope,
            )
        if _is_invalid_answer_candidate(answer):
            answer = ""
        if not answer:
            answer = await asyncio.to_thread(
                _direct_text_fallback_answer,
                llm_host=llm_host,
                llm_api_key=llm_api_key,
                model_name=active_model_name or candidate_models[0],
                challenge_description=str(challenge.description or ""),
                challenge_rules=str(getattr(challenge, "rules", "") or ""),
                clues=clue_texts,
                agent_id=agent_id,
                usage_scope=usage_scope,
            )
        if _is_invalid_answer_candidate(answer):
            answer = ""
        if not answer:
            answer = "unknown"
            await asyncio.to_thread(
                http_client.broadcast_thought,
                agent_id,
                "No valid answer extracted; submitting fallback 'unknown'.",
            )
        print(f"Submitting answer: {answer}")
        await asyncio.to_thread(http_client.broadcast_thought, agent_id, f"Answer: {answer}")
        submit_result = await asyncio.to_thread(
            http_client.submit,
            agent_id,
            answer,
            {
                "model_name": active_model_name,
                "total_tokens": usage_metrics["total_tokens"],
                "prompt_tokens": usage_metrics["prompt_tokens"],
                "completion_tokens": usage_metrics["completion_tokens"],
                "total_time_ms": total_time_ms,
            },
            "text",
        )
        print(f"   Accepted: {submit_result.accepted}")
        print(f"   Status: {submit_result.status}")
        if submit_result.score:
            print(f"   Score: {submit_result.score.get('final_score', 0)}")
        await asyncio.to_thread(http_client.update_status, agent_id, "submitted")
        await asyncio.to_thread(http_client.broadcast_thought, agent_id, "Text challenge submitted.")
    else:
        image_uri = extract_image_uri(raw_content)
        planned_tool, planned_instruction, planned_summary = extract_image_plan(raw_content)
        image_uri, selected_tool = await _resolve_image_candidate(
            initial_image_uri=image_uri,
            tool_state=tool_state,
            challenge=challenge,
            image_tool_specs=executable_image_tool_specs,
            ranked_models=ranked_models,
            mcp_url=mcp_url,
            api_key=api_key,
            agent_id=agent_id,
            planned_tool=planned_tool,
            planned_instruction=planned_instruction,
        )
        async with McpArenaClient(mcp_url, api_key) as submit_mcp:
            submit_result = await submit_mcp.submit_image(
                agent_id=agent_id,
                image_uri=image_uri,
                client_metrics={
                    "model_name": active_model_name,
                    "planner_tool": selected_tool,
                    "total_tokens": usage_metrics["total_tokens"],
                    "prompt_tokens": usage_metrics["prompt_tokens"],
                    "completion_tokens": usage_metrics["completion_tokens"],
                    "total_time_ms": total_time_ms,
                },
                rationale="CrewAI native tool run",
            )
        submit_log = dict(submit_result) if isinstance(submit_result, dict) else {"result": submit_result}
        submit_log.pop("edited_image", None)
        submit_log.pop("image_uri", None)
        print(f"   Image submission: {submit_log}")
        await asyncio.to_thread(http_client.update_status, agent_id, "submitted")
        if planned_summary:
            await asyncio.to_thread(http_client.broadcast_thought, agent_id, f"Image plan summary: {planned_summary}")
        await asyncio.to_thread(http_client.broadcast_thought, agent_id, "Image challenge completed and submitted.")

    print("\nAgent completed!")
    await session_monitor.stop()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except asyncio.CancelledError:
        raise SystemExit(0)
    except ModelSelectionError as exc:
        print(f"   {exc}")
        raise SystemExit(1)
