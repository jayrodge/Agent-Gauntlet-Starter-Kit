"""Example Agent that can solve text and image challenges."""

from __future__ import annotations

import asyncio
import json
import os
import re
import socket
import sys
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterator
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

load_dotenv(REPO_ROOT / ".env")

from arena_clients import (
    ArenaAPIError,
    ArenaConnectionError,
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
    require_explicit_model,
)
from my_strategy import MyStrategy


DEFAULT_AGENT_ID = "team-reference-agent"
DEFAULT_AGENT_NAME = "Team Reference Agent"
DEFAULT_TEXT_SYSTEM_PROMPT = (
    "You are a text challenge solver. "
    "Your first line must always be: ANSWER: <final answer>. "
    "Follow challenge rules exactly, especially strict output formats. "
    "Do not output <think> tags. "
    "If you add reasoning, keep it to at most 2 short lines after the ANSWER line. "
    "Never output 'unknown'."
)
DEFAULT_TEXT_TEMPERATURE = 0.0
DEFAULT_TEXT_MAX_TOKENS = 320
DEFAULT_IMAGE_STRATEGY_NOTES = ""
STRATEGY = MyStrategy()


def _truncate(text: str, max_len: int = 500) -> str:
    if len(text) <= max_len:
        return text
    return f"{text[:max_len]}...<truncated>"


def _build_strategy_context(
    *,
    challenge_type: str,
    difficulty: str = "unknown",
    challenge_text: str = "",
    description: str = "",
    rules: str = "",
    clues: list[str] | None = None,
    time_remaining_s: float = 0.0,
    max_time_s: int = 0,
    available_models: list[str] | None = None,
    tools_used: list[str] | None = None,
    tokens_used: int = 0,
    required_tools: list[str] | None = None,
    image_url: str | None = None,
) -> ChallengeContext:
    return ChallengeContext(
        challenge_type=(challenge_type or "text").strip().lower(),
        difficulty=(difficulty or "unknown").strip().lower(),
        challenge_text=challenge_text or "",
        description=description or "",
        rules=rules or "",
        clues=list(clues or []),
        time_remaining_s=float(time_remaining_s or 0.0),
        max_time_s=int(max_time_s or 0),
        available_models=list(available_models or []),
        tools_used=list(tools_used or []),
        tokens_used=int(tokens_used or 0),
        required_tools=list(required_tools or []),
        image_url=image_url,
    )


def _resolve_llm_api_key() -> str:
    """Use ARENA_API_KEY for LLM proxy auth; server expects the same key as Agent Gauntlet."""
    return (get_arena_api_key() or "").strip()


def _build_prompt(
    clue_texts: list[str],
    *,
    challenge_type: str,
    challenge_description: str,
    challenge_rules: str,
) -> str:
    lines = "\n".join(f"- {clue}" for clue in clue_texts if clue.strip())
    if not lines:
        lines = "- (No clues provided.)"

    challenge_type = challenge_type.strip() or "text"
    challenge_description = challenge_description.strip() or "No description provided."
    challenge_rules = challenge_rules.strip() or "No additional rules provided."

    return (
        "Solve the text challenge below.\n\n"
        f"Challenge Type: {challenge_type}\n"
        f"Description: {challenge_description}\n"
        f"Rules: {challenge_rules}\n\n"
        f"Clues:\n{lines}\n\n"
        "Output requirements:\n"
        "- First line must be exactly: ANSWER: <final answer>\n"
        "- Follow strict formatting constraints from Rules exactly.\n"
        "- Optional: up to 2 brief reasoning lines after ANSWER."
    )


def _coerce_float(value: str | None, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_positive_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


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


def _build_live_metrics(
    model_name: str,
    usage: dict[str, Any] | None,
    elapsed_ms: int,
) -> dict[str, Any]:
    usage = usage or {}
    prompt_tokens = _coerce_nonnegative_int(usage.get("prompt_tokens"))
    completion_tokens = _coerce_nonnegative_int(usage.get("completion_tokens"))
    total_tokens = _coerce_nonnegative_int(usage.get("total_tokens"))
    return {
        "model_name": model_name,
        "total_tokens": str(total_tokens),
        "prompt_tokens": str(prompt_tokens),
        "completion_tokens": str(completion_tokens),
        "total_time_ms": int(max(0, elapsed_ms)),
    }


_ANSWER_PREFIX_RE = re.compile(
    r"^(final\s+answer|answer)\s*[:\-]\s*(.+)$", re.IGNORECASE,
)

_INVALID_JUDGE_ANSWERS = {
    "none",
    "unknown",
    "n/a",
    "null",
    "<think>",
    "</think>",
    "think",
    "answer",
    "final answer",
}


_NAME_TOKEN_RE = re.compile(r"^[A-Za-z][A-Za-z' -]{0,30}$")


def _clean_judge_answer_candidate(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""
    cleaned = cleaned.strip("\"'` ")
    cleaned = cleaned.replace("```", "")
    cleaned = re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE).strip()
    if cleaned.endswith((".", ";")):
        cleaned = cleaned[:-1].strip()
    return cleaned


def _looks_like_name_list(text: str) -> bool:
    # Judge extraction in this path is intended to return ordered names.
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if len(parts) < 3:
        return False
    return all(bool(_NAME_TOKEN_RE.fullmatch(part)) for part in parts)


def _is_valid_judge_answer_candidate(text: str) -> bool:
    cleaned = _clean_judge_answer_candidate(text)
    if not cleaned:
        return False
    lowered = cleaned.lower()
    if lowered in _INVALID_JUDGE_ANSWERS:
        return False
    if "<think" in lowered or "</think" in lowered:
        return False
    if lowered.startswith(
        (
            "let me",
            "i will",
            "i'll",
            "we should",
            "we need",
            "i need",
            "first,",
            "first ",
            "step ",
        )
    ):
        return False
    if lowered.startswith(("reasoning:", "analysis:", "thought:")):
        return False
    if len(cleaned) > 220:
        return False
    if _looks_like_name_list(cleaned):
        return True
    if any(
        token in lowered
        for token in (
            "first,",
            "second,",
            "third,",
            "the text mentions",
            "let's",
            "because",
            "therefore",
            "step 1",
            "step one",
        )
    ) and len(cleaned.split()) > 10:
        return False
    if not re.search(r"[A-Za-z0-9]", cleaned):
        return False
    return True


def _quick_extract_answer(text: str) -> str | None:
    """Fast-path: look for an explicit 'ANSWER:' line (no LLM call needed)."""
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        match = _ANSWER_PREFIX_RE.match(line)
        if match:
            return match.group(2).strip()
    return None


def _extract_explicit_output_template(rules: str | None) -> str | None:
    if not rules:
        return None
    match = re.search(
        r"format(?: your answer)?(?: exactly)?(?: as| like)\s*:\s*['\"]([^'\"]+)['\"]",
        rules,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    template = match.group(1).strip()
    return template or None


def _matches_explicit_output_template(answer: str, rules: str | None) -> bool:
    template = _extract_explicit_output_template(rules)
    if not template:
        return True
    tokenized = re.sub(r"[A-Za-z0-9_]+", "__TOKEN__", template)
    pattern = re.escape(tokenized).replace("__TOKEN__", r"[A-Za-z0-9 .&/_'-]+")
    return bool(re.fullmatch(pattern, answer.strip()))


def _is_valid_submission_candidate(answer: str | None, rules: str | None) -> bool:
    candidate = _clean_judge_answer_candidate(str(answer or ""))
    if not _is_valid_judge_answer_candidate(candidate):
        return False
    if _is_invalid_meta_answer_candidate(candidate):
        return False
    if not _matches_explicit_output_template(candidate, rules):
        return False
    return True


def _extract_name_list_candidate(text: str | None) -> str | None:
    if not text:
        return None

    for line in reversed(text.splitlines()):
        cleaned = _clean_judge_answer_candidate(line)
        if _looks_like_name_list(cleaned):
            return cleaned

    match = re.search(
        r"([A-Z][a-z]+(?:\s*,\s*[A-Z][a-z]+){4})",
        text,
    )
    if match:
        candidate = _clean_judge_answer_candidate(match.group(1))
        if _looks_like_name_list(candidate):
            return candidate
    return None


def _extract_think_section(content: str) -> tuple[str | None, str]:
    """Separate <think>...</think> reasoning from the rest of the content.

    Handles several variants produced by reasoning models:
      - Full tags:       <think>reasoning</think>answer
      - Opening only:    <think>reasoning (truncated)
      - Closing only:    reasoning</think>answer   (NVIDIA style)
    """
    if "<think>" in content and "</think>" in content:
        think_start = content.find("<think>") + len("<think>")
        think_end = content.find("</think>")
        reasoning = content[think_start:think_end].strip()
        remaining = content[think_end + len("</think>"):].strip()
        return reasoning, remaining
    if "<think>" in content:
        think_start = content.find("<think>") + len("<think>")
        reasoning = content[think_start:].strip()
        return reasoning, ""
    # Handle </think> without <think> (NVIDIA reasoning models embed the
    # opening tag implicitly and only emit the closing tag in content).
    if "</think>" in content:
        think_end = content.find("</think>")
        reasoning = content[:think_end].strip()
        remaining = content[think_end + len("</think>"):].strip()
        return reasoning, remaining
    return None, content


# ---------------------------------------------------------------------------
# LLM-as-Judge answer extraction
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = (
    "You extract answers from raw text. "
    "Return ONLY the final answer text. "
    "No explanation. No reasoning. No <think> tags.\n\n"
    "If the text contains an ANSWER: line, return only what appears after ANSWER:. "
    "If no answer can be extracted, respond: NONE."
)


def _llm_judge_extract(raw_output: str, judge_model: str) -> str | None:
    """Call a lightweight LLM to extract the answer from raw reasoning output.

    Returns the extracted answer string, or *None* if the judge could not
    identify one.
    """
    proxy_host = get_proxy_host()
    judge_model = str(judge_model or "").strip()
    if not judge_model:
        return None
    url = f"{proxy_host.rstrip('/')}/chat/completions"
    api_key = _resolve_llm_api_key()

    # Use only the LAST portion of the output — the conclusion is almost
    # always near the end.  Keeping it short also reduces the judge model's
    # reasoning overhead so it has budget left for the actual answer.
    truncated = raw_output[-800:] if len(raw_output) > 800 else raw_output

    payload = {
        "model": judge_model,
        "messages": [
            {"role": "system", "content": _JUDGE_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Extract the final answer from this text:\n\n"
                    f"{truncated}\n\n"
                    f"Answer:"
                ),
            },
        ],
        "temperature": 0.0,
        "max_tokens": 64,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    headers.update(build_proxy_headers(os.getenv("AGENT_ID", "").strip()))

    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urlopen(request, timeout=30.0) as response:
            body = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        print(f"[judge] extraction call failed: {exc}", flush=True)
        return None

    message = body.get("choices", [{}])[0].get("message", {})
    # Non-reasoning models return the answer directly in 'content'.
    # Reasoning models may split into 'content' + 'reasoning_content'.
    answer_text = (message.get("content") or "").strip()
    if not answer_text:
        # Some models put everything in reasoning_content
        answer_text = (
            message.get("reasoning_content") or message.get("reasoning") or ""
        ).strip()

    if not answer_text:
        return None

    # Strip <think>...</think> wrapper if the judge model added one.
    if "<think" in answer_text.lower() or "</think>" in answer_text.lower():
        _, after_think = _extract_think_section(answer_text)
        if after_think:
            answer_text = after_think
        else:
            # Pure tag/reasoning output is not a valid final answer.
            answer_text = _clean_judge_answer_candidate(answer_text)

    if answer_text.upper() == "NONE":
        return None

    # Try quick-extract for "ANSWER:" prefix first
    quick = _quick_extract_answer(answer_text)
    if quick and quick.upper() != "NONE":
        quick = _clean_judge_answer_candidate(quick)
        if _is_valid_judge_answer_candidate(quick):
            return quick

    # Otherwise take the first non-empty line that isn't meta-text
    for line in answer_text.splitlines():
        line = _clean_judge_answer_candidate(line)
        if not line or line.upper() == "NONE":
            continue
        if not _is_valid_judge_answer_candidate(line):
            continue
        # Skip lines that look like reasoning
        lower = line.lower()
        if any(w in lower for w in ("let me", "okay", "the answer", "based on", "i think", "step")):
            continue
        m = _ANSWER_PREFIX_RE.match(line)
        candidate = m.group(2).strip() if m else line
        candidate = _clean_judge_answer_candidate(candidate)
        if _is_valid_judge_answer_candidate(candidate):
            return candidate
    return None


async def _async_judge_extract(raw_output: str, judge_model: str) -> str | None:
    """Async wrapper around the synchronous judge extraction call."""
    return await asyncio.to_thread(_llm_judge_extract, raw_output, judge_model)


def _extract_answer(
    content: str | None,
    reasoning: str | None,
) -> tuple[str | None, str | None]:
    """Try quick extraction from content and reasoning.

    Returns (answer, reasoning_text).  Does NOT call the LLM judge – the
    caller should fall through to the judge when this returns answer=None.
    """
    # Try content first
    if content:
        answer = _quick_extract_answer(content)
        if answer:
            return answer, reasoning or content
        candidate = _extract_name_list_candidate(content)
        if candidate:
            return candidate, reasoning or content
        for line in reversed(content.splitlines()):
            candidate = _clean_judge_answer_candidate(line)
            if _is_valid_judge_answer_candidate(candidate):
                return candidate, reasoning or content
    # Try reasoning text
    if reasoning:
        answer = _quick_extract_answer(reasoning)
        if answer:
            return answer, reasoning
        candidate = _extract_name_list_candidate(reasoning)
        if candidate:
            return candidate, reasoning
        for line in reversed(reasoning.splitlines()):
            candidate = _clean_judge_answer_candidate(line)
            if _is_valid_judge_answer_candidate(candidate):
                return candidate, reasoning
    # Return what we have; caller decides next step
    return None, reasoning or content


def _iter_reasoning_chunks(reasoning: str, max_len: int = 220) -> list[str]:
    chunks: list[str] = []
    for line in reasoning.splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        if len(cleaned) <= max_len:
            chunks.append(cleaned)
            continue
        for start in range(0, len(cleaned), max_len):
            chunks.append(cleaned[start:start + max_len])
    return chunks


def _find_search_tool(tools: list[Any]) -> str | None:
    """Find a search tool from the discovered tool list."""
    for tool in tools:
        name = getattr(tool, "name", "")
        if "search" in name.lower():
            return name
    return None


def _coerce_search_query_param(tools: list[Any], search_tool_name: str) -> str | None:
    """Find the likely query parameter name for a search tool."""
    for tool in tools:
        if getattr(tool, "name", None) != search_tool_name:
            continue
        schema = getattr(tool, "inputSchema", None) or {}
        props = schema.get("properties") if isinstance(schema, dict) else None
        if not isinstance(props, dict) or not props:
            return "query"
        keys = list(props.keys())
        for preferred in ("query", "q", "text", "search"):
            if preferred in props:
                return preferred
        return keys[0]
    return None


def _challenge_text_blob(
    challenge_type: str,
    description: str,
    rules: str,
    extra_context: str = "",
) -> str:
    return " ".join(
        [challenge_type or "", description or "", rules or "", extra_context or ""]
    ).lower()


def _query_candidates(
    description: str,
    rules: str | None,
    clues: list[str] | None,
    extra_context: str = "",
) -> list[str]:
    candidates: list[str] = []
    if description and description.strip():
        candidates.append(description.strip())
    for clue in clues or []:
        if isinstance(clue, str) and clue.strip():
            candidates.append(clue.strip())
    if isinstance(rules, str) and rules.strip():
        for part in re.split(r"[.;\n]+", rules):
            part = part.strip()
            if 8 <= len(part) <= 240:
                candidates.append(part)
    for text in [description, rules or "", extra_context, *(clues or [])]:
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


def _is_invalid_meta_answer_candidate(value: str | None) -> bool:
    text = str(value or "").strip()
    if not text:
        return True
    lowered = text.lower()
    if lowered.startswith(("<toolcall", "toolcall", "<think", "reasoning:", "analysis:", "action:")):
        return True
    blocked_tokens = ("<toolcall", "</toolcall", '"tool_calls"', "function_call")
    return any(token in lowered for token in blocked_tokens)


def _score_candidate(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return -1
    score = 0
    if re.search(r"[A-Za-z]", stripped):
        score += 2
    if re.search(r"\d", stripped):
        score += 2
    if len(stripped) <= 120:
        score += 1
    if re.search(r"https?://", stripped):
        score -= 2
    numeric_only = stripped.replace(",", "")
    if numeric_only.isdigit() and len(numeric_only) >= 6:
        score -= 3
    return score


def _iter_text_candidates(result: Any) -> list[str]:
    if result is None:
        return []
    if isinstance(result, str):
        parts: list[str] = []
        for line in result.splitlines():
            line = line.strip()
            if not line:
                continue
            parts.append(line)
            for chunk in re.split(r"\s[|•·]\s|\s-\s", line):
                chunk = chunk.strip()
                if chunk and chunk not in parts:
                    parts.append(chunk)
        return parts
    if isinstance(result, dict):
        parts = []
        for key in ("answer", "value", "text", "snippet", "title", "content", "raw"):
            if key in result:
                parts.extend(_iter_text_candidates(result.get(key)))
        for value in result.values():
            parts.extend(_iter_text_candidates(value))
        return parts
    if isinstance(result, list):
        parts = []
        for item in result:
            parts.extend(_iter_text_candidates(item))
        return parts
    return []


def _format_search_results_for_llm(search_result: Any, max_items: int = 3) -> str:
    if not isinstance(search_result, dict):
        return _truncate(json.dumps(search_result, ensure_ascii=True, default=str), 1500)
    web_results = search_result.get("web")
    if not isinstance(web_results, list):
        return _truncate(json.dumps(search_result, ensure_ascii=True, default=str), 1500)
    lines: list[str] = []
    for idx, item in enumerate(web_results[:max_items], start=1):
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        snippet = str(item.get("description") or item.get("snippet") or "").strip()
        url = str(item.get("url") or "").strip()
        if not title and not snippet:
            continue
        lines.append(f"Result {idx}:")
        if title:
            lines.append(f"Title: {title}")
        if snippet:
            lines.append(f"Snippet: {snippet}")
        if url:
            lines.append(f"URL: {url}")
        lines.append("")
    formatted = "\n".join(lines).strip()
    return formatted or _truncate(json.dumps(search_result, ensure_ascii=True, default=str), 1500)


def _extract_first_json_object(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    if not cleaned:
        return None
    if cleaned.startswith("<think>") and "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1].strip()
    if cleaned.startswith("{"):
        try:
            data = json.loads(cleaned)
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            pass
    start = cleaned.find("{")
    if start == -1:
        return None
    depth = 0
    for idx, char in enumerate(cleaned[start:], start=start):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                snippet = cleaned[start : idx + 1]
                try:
                    data = json.loads(snippet)
                    return data if isinstance(data, dict) else None
                except json.JSONDecodeError:
                    return None
    return None


def _extract_answer_field(text: str) -> str | None:
    match = re.search(r'"answer"\s*:\s*"([^"]+)', text)
    if not match:
        return None
    value = match.group(1).strip()
    return value if value else None


def _answer_matches_rules(answer: str, rules: str | None) -> bool:
    if not rules:
        return True
    rules_lower = rules.lower()
    if "year" in rules_lower and not re.search(r"\b(19|20)\d{2}\b", answer):
        return False
    if any(token in rules_lower for token in ("number", "count", "amount", "total")):
        if not re.search(r"\d", answer):
            return False
    return True


def _requires_names_only_output(rules: str | None) -> bool:
    if not rules:
        return False
    rules_lower = rules.lower()
    return any(
        token in rules_lower
        for token in (
            "names only",
            "provider names only",
            "one line with provider names",
            "cheapest-to-most-expensive order",
            "most-expensive",
        )
    )


_NAME_ORDER_PART_RE = re.compile(r"^[A-Za-z][A-Za-z0-9 .&/_-]{0,40}$")


def _normalize_name_order_answer(answer: str) -> str | None:
    parts = [part.strip() for part in answer.split(",") if part.strip()]
    if len(parts) < 2:
        return None
    if not all(_NAME_ORDER_PART_RE.fullmatch(part) for part in parts):
        return None
    # Reject obvious title/sentence style answers.
    lowered = answer.lower()
    if any(token in lowered for token in ("comparison", "pricing", "report", "article", ":")):
        return None
    if re.search(r"\b(19|20)\d{2}\b", answer):
        return None
    return ", ".join(parts)


def _extract_name_order_from_rules(rules: str | None) -> str | None:
    if not rules:
        return None
    for quoted in re.findall(r"'([^']+)'", rules):
        normalized = _normalize_name_order_answer(quoted.strip())
        if normalized:
            return normalized
    return None


def _extract_answer_from_search_result(result: Any) -> str | None:
    """Heuristic extraction for web-search results without query-specific rules."""
    candidates = _iter_text_candidates(result)
    if not candidates:
        return None
    filtered = [
        candidate.strip()
        for candidate in candidates
        if candidate.strip()
        and len(candidate.strip()) <= 220
        and not _is_invalid_meta_answer_candidate(candidate.strip())
    ]
    if not filtered:
        return None
    best = max(filtered, key=_score_candidate)
    return best if _score_candidate(best) > 0 else None


def _repair_final_answer_with_llm(
    *,
    question: str,
    rules: str | None,
    clues: list[str],
    current_answer: str | None,
    reasoning: str | None,
    model_name: str,
) -> tuple[str | None, dict[str, Any] | None, int]:
    clues_block = "\n".join(f"- {clue}" for clue in clues if clue.strip()) or "- (No clues provided.)"
    prompt = (
        "Return exactly one final answer line that follows the challenge rules.\n\n"
        f"Question: {question}\n"
        f"Rules: {rules or ''}\n"
        f"Current candidate answer: {current_answer or ''}\n\n"
        f"Clues:\n{clues_block}\n\n"
        f"Reasoning/context:\n{reasoning or ''}\n\n"
        "Output only the final answer line."
    )
    message, usage, ttft_ms = _request_llm_message(
        prompt=prompt,
        system_prompt=(
            "You are a strict final-answer formatter and validator. "
            "Follow the required output shape exactly. "
            "Do not include explanation, labels, or reasoning."
        ),
        model_name=model_name,
        temperature=0.0,
        max_tokens=192,
    )
    if not message:
        return None, usage, ttft_ms
    content = (message.get("content") or "").strip()
    if not content:
        return None, usage, ttft_ms
    quick = _quick_extract_answer(content)
    if quick:
        cleaned = _clean_judge_answer_candidate(quick)
        if _is_valid_submission_candidate(cleaned, rules):
            return cleaned, usage, ttft_ms
    for line in content.splitlines():
        cleaned = _clean_judge_answer_candidate(line)
        if _is_valid_submission_candidate(cleaned, rules):
            return cleaned, usage, ttft_ms
    return None, usage, ttft_ms


_IMAGE_TOOL_PLAN_SYSTEM = (
    "You are selecting MCP image tools for an autonomous agent. "
    "You must choose ONE primary tool from: image_edit, image_generate, image_analyze. "
    "Return ONLY valid JSON with keys: tool, prompt, question, rationale. "
    "If an input image is available, prefer image_edit for transformation tasks. "
    "Use image_generate when there is no input image or a fresh image is needed. "
    "Use image_analyze for understanding/inspection tasks. "
    "When crafting image prompts, request standard resolution only "
    "(about 1024 pixels on the longest side, or the source size if smaller). "
    "Do not request HD, 4K, or upscaling."
)

_BLANK_PNG_DATA_URI = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7/"
    "S7sAAAAASUVORK5CYII="
)


def _default_image_tool_plan(
    *,
    available_tools: list[str],
    challenge_prompt: str,
    challenge_description: str,
    challenge_type: str,
    has_input_image: bool,
    max_time_s: int = 0,
) -> dict[str, str]:
    ctx = _build_strategy_context(
        challenge_type=challenge_type,
        challenge_text=challenge_prompt,
        description=challenge_description,
        rules="",
        max_time_s=max_time_s,
        image_url="input://challenge" if has_input_image else None,
    )
    selected_tool = STRATEGY.plan_image_tool(ctx, available_tools)
    prompt = STRATEGY.build_image_prompt(ctx).strip() or (
        challenge_prompt.strip() or "Generate a clear image response for the task."
    )
    if selected_tool == "image_edit" and has_input_image and "image_edit" in available_tools:
        return {
            "tool": "image_edit",
            "prompt": prompt,
            "question": "",
            "rationale": "Selected by strategy image tool planner.",
        }
    if selected_tool == "image_generate" and "image_generate" in available_tools:
        return {
            "tool": "image_generate",
            "prompt": prompt,
            "question": "",
            "rationale": "Selected by strategy image tool planner.",
        }
    if selected_tool == "image_analyze" and has_input_image and "image_analyze" in available_tools:
        return {
            "tool": "image_analyze",
            "prompt": "",
            "question": prompt,
            "rationale": "Selected by strategy image tool planner.",
        }
    if has_input_image and "image_edit" in available_tools:
        return {
            "tool": "image_edit",
            "prompt": prompt,
            "question": "",
            "rationale": "Input image present; edit is the safest default path.",
        }
    if "image_generate" in available_tools:
        return {
            "tool": "image_generate",
            "prompt": prompt,
            "question": "",
            "rationale": "No editable image path available; generating output image.",
        }
    if has_input_image and "image_analyze" in available_tools:
        return {
            "tool": "image_analyze",
            "prompt": "",
            "question": prompt,
            "rationale": "Only analysis tool available.",
        }
    return {
        "tool": "",
        "prompt": prompt,
        "question": prompt,
        "rationale": "No supported image MCP tools available.",
    }


def _plan_image_tool_call(
    *,
    model_name: str,
    challenge_type: str,
    challenge_description: str,
    challenge_prompt: str,
    reference_notes: str,
    has_input_image: bool,
    available_tools: list[str],
) -> tuple[dict[str, str], dict[str, Any] | None, int]:
    default_plan = _default_image_tool_plan(
        available_tools=available_tools,
        challenge_prompt=challenge_prompt,
        challenge_description=challenge_description,
        challenge_type=challenge_type,
        has_input_image=has_input_image,
        max_time_s=0,
    )
    if not available_tools:
        return default_plan, None, 0

    prompt = (
        "Choose one MCP tool and return JSON.\n\n"
        f"Challenge type: {challenge_type}\n"
        f"Description: {challenge_description[:500]}\n"
        f"Task prompt: {challenge_prompt[:500]}\n"
        f"Reference notes: {reference_notes[:500]}\n"
        f"Input image available: {'yes' if has_input_image else 'no'}\n"
        f"Available tools: {', '.join(available_tools)}\n\n"
        "Output schema:\n"
        "{\n"
        '  "tool": "image_edit|image_generate|image_analyze",\n'
        '  "prompt": "text prompt for edit/generate (can be empty for analyze)",\n'
        '  "question": "question for analyze (can be empty for edit/generate)",\n'
        '  "rationale": "short reason"\n'
        "}\n"
    )
    message, usage, ttft_ms = _request_llm_message(
        prompt=prompt,
        system_prompt=_IMAGE_TOOL_PLAN_SYSTEM,
        model_name=model_name,
        temperature=0.0,
        max_tokens=220,
    )
    if not message:
        return default_plan, usage, ttft_ms

    content = (message.get("content") or "").strip()
    if not content:
        content = (message.get("reasoning_content") or message.get("reasoning") or "").strip()
    payload = _extract_first_json_object(content) if content else None
    if not isinstance(payload, dict):
        return default_plan, usage, ttft_ms

    tool = payload.get("tool")
    if not isinstance(tool, str):
        return default_plan, usage, ttft_ms
    tool = tool.strip().lower()
    if tool not in available_tools:
        return default_plan, usage, ttft_ms

    plan = {
        "tool": tool,
        "prompt": str(payload.get("prompt") or "").strip(),
        "question": str(payload.get("question") or "").strip(),
        "rationale": str(payload.get("rationale") or "").strip(),
    }
    if not plan["prompt"]:
        plan["prompt"] = challenge_prompt.strip()
    if not plan["question"]:
        plan["question"] = challenge_prompt.strip()
    if not plan["rationale"]:
        plan["rationale"] = "Selected by LLM planner."
    return plan, usage, ttft_ms


def _request_llm_message(
    *,
    prompt: str,
    system_prompt: str,
    model_name: str,
    temperature: float = 0.2,
    max_tokens: int = 256,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, int]:
    proxy_host = get_proxy_host()
    if not proxy_host:
        print(
            "[agent] LLM proxy host not configured; competitor agents only support Agent Gauntlet proxy access.",
            flush=True,
        )
        return None, None, 0
    url = f"{proxy_host.rstrip('/')}/chat/completions"
    api_key = _resolve_llm_api_key()
    if not api_key:
        print("[agent] Agent Gauntlet API key not set; skipping LLM", flush=True)
        return None, None, 0

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    headers.update(build_proxy_headers(os.getenv("AGENT_ID", "").strip()))

    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    timeout_value = 180.0
    timeout_env = os.getenv("LLM_TIMEOUT_S")
    if timeout_env:
        try:
            parsed = float(timeout_env)
            if parsed > 0:
                timeout_value = parsed
        except ValueError:
            timeout_value = 180.0

    try:
        with urlopen(request, timeout=timeout_value) as response:
            raw_body = response.read().decode("utf-8")
    except HTTPError as exc:
        error_body = ""
        try:
            error_body = exc.read().decode("utf-8")
        except Exception:
            error_body = ""
        print(
            f"[agent] LLM HTTP error status={exc.code} body={_truncate(error_body)}",
            flush=True,
        )
        return None, None, 0
    except (TimeoutError, socket.timeout) as exc:
        print(f"[agent] LLM timeout after {timeout_value}s: {exc}", flush=True)
        return None, None, 0
    except URLError as exc:
        print(f"[agent] LLM URL error: {exc}", flush=True)
        return None, None, 0
    except Exception as exc:
        print(f"[agent] LLM request failed: {exc}", flush=True)
        return None, None, 0

    try:
        body = json.loads(raw_body)
    except json.JSONDecodeError as exc:
        print(f"[agent] LLM JSON parse error: {exc}", flush=True)
        print(f"[agent] LLM raw body: {_truncate(raw_body)}", flush=True)
        return None, None, 0

    message = body.get("choices", [{}])[0].get("message", {}) or {}
    usage = body.get("usage")
    ttft_ms = body.get("ttft_ms", 0)
    if isinstance(ttft_ms, (int, float)) and ttft_ms >= 0:
        ttft_ms = int(ttft_ms)
    else:
        ttft_ms = 0

    return message, usage, ttft_ms


def try_extract_search_answer_with_llm(
    *,
    question: str,
    rules: str | None,
    search_result: Any,
    model_name: str,
) -> tuple[str | None, dict[str, Any] | None, int]:
    result_text = _format_search_results_for_llm(search_result)
    rules_text = rules or "Answer with the final result only."
    prompt = (
        "Extract the precise answer to the question below from the search results.\n\n"
        f"Question: {question}\n\n"
        f"Rules: {rules_text}\n\n"
        f"Search Results:\n{result_text}\n\n"
        "INSTRUCTIONS:\n"
        "1. Read the search results carefully\n"
        "2. Extract ONLY the direct answer to the question\n"
        "3. Follow the rules exactly (e.g., if rules say 'company name only', give ONLY the company name)\n"
        "4. Do NOT include URLs, titles, extra context, or full sentences\n"
        "5. Output format: {\"answer\": \"your_answer\"}\n\n"
        "Now extract the answer:"
    )
    message, usage, ttft_ms = _request_llm_message(
        prompt=prompt,
        system_prompt=(
            "You are a precise answer extractor. "
            "Read search results and return ONLY the requested information in JSON format. "
            "Follow the rules exactly. Output only valid JSON with no extra text."
        ),
        model_name=model_name,
        temperature=0.0,
        max_tokens=128,
    )
    if not message:
        return None, usage, ttft_ms
    content = (message.get("content") or "").strip()
    if not content:
        return None, usage, ttft_ms
    payload = _extract_first_json_object(content)
    if not isinstance(payload, dict):
        recovered = _extract_answer_field(content)
        if (
            recovered
            and not _is_invalid_meta_answer_candidate(recovered)
            and _answer_matches_rules(recovered, rules)
        ):
            return recovered, usage, ttft_ms
        return None, usage, ttft_ms
    answer = payload.get("answer")
    if (
        isinstance(answer, str)
        and answer.strip()
        and not _is_invalid_meta_answer_candidate(answer)
        and _answer_matches_rules(answer, rules)
    ):
        return answer.strip(), usage, ttft_ms
    return None, usage, ttft_ms


def _derive_llm_solve_timeout_s(max_time_s: int) -> float:
    configured_timeout_s = 180.0
    timeout_env = os.getenv("LLM_TIMEOUT_S")
    if timeout_env:
        try:
            parsed = float(timeout_env)
            if parsed > 0:
                configured_timeout_s = parsed
        except ValueError:
            configured_timeout_s = 180.0

    normalized_max_time_s = max(0.0, float(max_time_s or 0))
    if normalized_max_time_s <= 0:
        return max(10.0, configured_timeout_s)
    submission_reserve_s = min(30.0, normalized_max_time_s / 2)
    return max(
        10.0, min(configured_timeout_s, normalized_max_time_s - submission_reserve_s)
    )


async def _call_image_tool_before_deadline(
    mcp_client: McpArenaClient,
    tool_name: str,
    arguments: dict[str, Any],
    *,
    deadline: float,
) -> dict[str, Any]:
    """Bound image work so the agent retains time to submit a fallback."""
    remaining_s = deadline - time.monotonic()
    if remaining_s <= 0:
        return {"error": f"{tool_name} timed out before submission reserve"}
    try:
        result = await asyncio.wait_for(
            mcp_client.call_tool(tool_name, arguments),
            timeout=remaining_s,
        )
    except asyncio.TimeoutError:
        return {"error": f"{tool_name} timed out before submission reserve"}
    if isinstance(result, dict):
        return result
    return {"raw": str(result)}


def _has_final_repair_budget(remaining_s: float) -> bool:
    return max(0.0, float(remaining_s or 0.0)) >= 45.0


def _iter_sse_events(stream, deadline: float | None = None) -> Iterator[str]:
    data_lines: list[str] = []
    while True:
        if deadline is not None and time.monotonic() >= deadline:
            break
        line = stream.readline()
        if deadline is not None and time.monotonic() >= deadline:
            break
        if not line:
            break
        line = line.decode("utf-8", errors="replace").strip("\r\n")
        if not line:
            if data_lines:
                yield "\n".join(data_lines)
                data_lines = []
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if data_lines:
        yield "\n".join(data_lines)


async def _consume_sse_stream(
    response,
    broadcast: Callable[[str], Awaitable[None]],
    start_time: float,
    deadline: float | None = None,
) -> tuple[str | None, str | None, dict[str, Any] | None, int, bool]:
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    usage: dict[str, Any] | None = None
    ttft_ms = 0
    pending_reasoning = ""
    reasoning_streamed = False
    saw_reasoning_field = False
    header_sent = False

    async def flush_pending(force: bool = False) -> None:
        nonlocal pending_reasoning, reasoning_streamed
        if not pending_reasoning.strip():
            pending_reasoning = ""
            return
        if force or len(pending_reasoning) >= 240 or "\n" in pending_reasoning:
            chunks = _iter_reasoning_chunks(pending_reasoning)
            pending_reasoning = ""
            for chunk in chunks:
                if not chunk:
                    continue
                reasoning_streamed = True
                await broadcast(f"   {chunk}")
                await asyncio.sleep(0.05)

    for data in _iter_sse_events(response, deadline=deadline):
        if deadline is not None and time.monotonic() >= deadline:
            await broadcast("⏱️ LLM stream deadline reached; using partial output.")
            break
        if data.strip() == "[DONE]":
            break
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            continue
        if not ttft_ms and (event.get("choices") or event.get("usage")):
            ttft_ms = int((time.monotonic() - start_time) * 1000)
        if isinstance(event.get("usage"), dict):
            usage = event.get("usage")
        choices = event.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        reasoning_delta = delta.get("reasoning_content") or delta.get("reasoning")
        content_delta = delta.get("content")
        if reasoning_delta:
            saw_reasoning_field = True
            reasoning_parts.append(reasoning_delta)
            pending_reasoning += reasoning_delta
        if content_delta:
            content_parts.append(content_delta)
            if not saw_reasoning_field:
                pending_reasoning += content_delta
        if pending_reasoning and not header_sent:
            await broadcast("💭 LLM Reasoning:")
            header_sent = True
        await flush_pending()

    await flush_pending(force=True)

    content = "".join(content_parts).strip()
    reasoning = "".join(reasoning_parts).strip() if reasoning_parts else None

    # Separate <think> tags if present in content
    if content:
        think_text, remaining = _extract_think_section(content)
        if think_text and not reasoning:
            reasoning = think_text
        content = remaining

    # Quick extraction (no LLM call)
    answer, reasoning = _extract_answer(content, reasoning)

    if answer:
        answer = answer.strip()
    if reasoning:
        reasoning = reasoning.strip()

    return answer, reasoning, usage, ttft_ms, reasoning_streamed


async def try_solve_with_llm(
    clue_texts: list[str],
    model_name: str,
    broadcast: Callable[[str], Awaitable[None]],
    text_temperature: float,
    text_max_tokens: int,
    challenge_type: str,
    challenge_description: str,
    challenge_rules: str,
    system_prompt: str,
    solver_prompt: str | None = None,
    solve_timeout_s: float | None = None,
) -> tuple[str | None, str | None, dict[str, Any] | None, int, bool]:
    """
    Call LLM to solve the puzzle.
    Returns: (answer, reasoning, usage, ttft_ms, reasoning_streamed).
    """
    proxy_host = get_proxy_host()
    stream_enabled = os.getenv("LLM_STREAM", "1").lower() not in {"0", "false", "no"}
    if not proxy_host:
        print(
            "[agent] LLM proxy host not configured; competitor agents only support Agent Gauntlet proxy access.",
            flush=True,
        )
        return None, None, None, 0, False
    url = f"{proxy_host.rstrip('/')}/chat/completions"
    api_key = _resolve_llm_api_key()
    if not api_key:
        print("[agent] Agent Gauntlet API key not set; skipping LLM", flush=True)
        return None, None, None, 0, False
    prompt = solver_prompt or _build_prompt(
        clue_texts,
        challenge_type=challenge_type,
        challenge_description=challenge_description,
        challenge_rules=challenge_rules,
    )
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": text_temperature,
        "max_tokens": text_max_tokens,
    }
    if stream_enabled:
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    headers.update(build_proxy_headers(os.getenv("AGENT_ID", "").strip()))

    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    timeout_value = _derive_llm_solve_timeout_s(0)
    if solve_timeout_s is not None and solve_timeout_s > 0:
        timeout_value = min(timeout_value, solve_timeout_s)

    start_time = time.monotonic()
    try:
        with urlopen(request, timeout=timeout_value) as response:
            status = response.getcode()
            if stream_enabled:
                return await _consume_sse_stream(
                    response,
                    broadcast,
                    start_time,
                    deadline=start_time + timeout_value,
                )
            raw_body = response.read().decode("utf-8")
    except HTTPError as exc:
        error_body = ""
        try:
            error_body = exc.read().decode("utf-8")
        except Exception:
            error_body = ""
        print(
            f"[agent] LLM HTTP error status={exc.code} body={_truncate(error_body)}",
            flush=True,
        )
        return None, None, None, 0, False
    except (TimeoutError, socket.timeout) as exc:
        print(f"[agent] LLM timeout after {timeout_value}s: {exc}", flush=True)
        return None, None, None, 0, False
    except URLError as exc:
        print(f"[agent] LLM URL error: {exc}", flush=True)
        return None, None, None, 0, False
    except (ConnectionResetError, ConnectionError, BrokenPipeError) as exc:
        print(f"[agent] LLM connection error: {exc}", flush=True)
        return None, None, None, 0, False

    print(f"[agent] LLM response status={status}", flush=True)
    try:
        body = json.loads(raw_body)
    except json.JSONDecodeError as exc:
        print(f"[agent] LLM JSON parse error: {exc}", flush=True)
        print(f"[agent] LLM raw body: {_truncate(raw_body)}", flush=True)
        return None, None, None, 0, False

    message = body.get("choices", [{}])[0].get("message", {})
    content = message.get("content", "")
    # Check for reasoning_content field (from LLM proxy with reasoning models)
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    usage = body.get("usage")
    ttft_ms = body.get("ttft_ms", 0)
    if isinstance(ttft_ms, (int, float)) and ttft_ms >= 0:
        ttft_ms = int(ttft_ms)
    else:
        ttft_ms = 0

    if not content and not reasoning:
        return None, None, usage, ttft_ms, False

    content = content.strip()
    extracted_think, remaining_content = _extract_think_section(content)
    if extracted_think and not reasoning:
        reasoning = extracted_think
    content = remaining_content

    # Quick extraction (no LLM call)
    answer, reasoning = _extract_answer(content, reasoning)

    return answer, reasoning, usage, ttft_ms, False


async def _broadcast_thought(
    http_client: HttpArenaClient,
    agent_id: str,
    thought: str,
) -> None:
    try:
        await asyncio.to_thread(http_client.broadcast_thought, agent_id, thought)
    except (TimeoutError, ArenaConnectionError, URLError) as exc:
        # Broadcast is best-effort; don't crash on network issues (e.g. remote server latency).
        print(f"[agent] broadcast skipped ({exc})", flush=True)


async def _save_draft(
    http_client: HttpArenaClient,
    agent_id: str,
    draft: str,
    rationale: str | None = None,
) -> None:
    await asyncio.to_thread(http_client.save_draft, agent_id, draft, rationale)


async def _register_agent(
    http_client: HttpArenaClient,
    agent_id: str,
    agent_name: str | None = None,
) -> None:
    await asyncio.to_thread(http_client.register, agent_id, agent_name)


async def _update_status(
    http_client: HttpArenaClient,
    agent_id: str,
    status: str,
    client_metrics: dict[str, Any] | None = None,
) -> None:
    try:
        await asyncio.to_thread(http_client.update_status, agent_id, status, client_metrics)
    except Exception:
        # Status endpoint is best-effort for compatibility with older servers.
        return


async def _submit_answer(
    http_client: HttpArenaClient,
    agent_id: str,
    answer: str,
    client_metrics: dict[str, Any],
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            result = await asyncio.to_thread(
                http_client.submit,
                agent_id,
                answer,
                client_metrics,
                "text",
            )
            return result.__dict__
        except (TimeoutError, ArenaConnectionError) as exc:
            last_error = exc
            if attempt < 3:
                print(
                    f"[agent] submit attempt {attempt} failed ({exc}); retrying...",
                    flush=True,
                )
                await asyncio.sleep(float(attempt))
                continue
            break
        except ArenaAPIError as exc:
            # API returned a definitive error; don't retry blindly.
            last_error = exc
            break
        except Exception as exc:
            last_error = exc
            break

    return {
        "accepted": False,
        "agent_id": agent_id,
        "answer": answer,
        "score": None,
        "status": "submit_failed",
        "error": str(last_error) if last_error else "unknown submission error",
    }


async def _wait_for_start_gate(
    http_client: HttpArenaClient,
    agent_id: str,
    broadcast: Callable[[str], Awaitable[None]],
) -> None:
    """Wait for organizer start when gate is enabled."""
    await _update_status(http_client, agent_id, "ready")
    await broadcast("✅ Connected to Agent Gauntlet")
    print("[agent] Connected to Agent Gauntlet", flush=True)

    last_phase: str | None = None
    last_countdown: Any = None
    waiting_for_next_round = False
    while True:
        try:
            competition = await asyncio.to_thread(http_client.get_competition)
        except Exception:
            # If competition endpoint is unavailable, stay backward-compatible.
            print("[agent] Competition endpoint unavailable; proceeding without gate.", flush=True)
            return

        phase = str(competition.get("phase") or "").lower()
        countdown_value = competition.get("countdown_value")
        eligible_agent_ids = competition.get("eligible_agent_ids")
        eligible_for_current_round = True
        if isinstance(eligible_agent_ids, list):
            eligible_set = {
                str(value) for value in eligible_agent_ids if isinstance(value, str) and value.strip()
            }
            # If an allowlist exists, only those connected before start should run.
            if eligible_set:
                eligible_for_current_round = agent_id in eligible_set
            else:
                eligible_for_current_round = False

        if phase == "running":
            if not eligible_for_current_round:
                if not waiting_for_next_round:
                    print(
                        "[agent] Battle already running. Waiting for next organizer start.",
                        flush=True,
                    )
                    await broadcast("⏸️ Battle already running. Waiting for next battle.")
                    waiting_for_next_round = True
                await asyncio.sleep(1.0)
                continue
            print("[agent] GO - challenge unlocked", flush=True)
            await broadcast("🏁 GO - challenge unlocked")
            await _update_status(http_client, agent_id, "running")
            return
        waiting_for_next_round = False

        if phase == "countdown":
            if countdown_value != last_countdown:
                print(f"[agent] Countdown: {countdown_value}", flush=True)
                await broadcast(f"⏳ Countdown: {countdown_value}")
                last_countdown = countdown_value
        else:
            if phase != last_phase:
                print("[agent] Waiting for organizer start", flush=True)
                await broadcast("⏸️ Waiting for organizer start")
                last_phase = phase

        await asyncio.sleep(1.0)


async def run() -> None:
    ensure_connected()

    proxy_host = get_proxy_host()
    llm_api_key = _resolve_llm_api_key()
    strategy_defaults = _build_strategy_context(challenge_type="text")
    default_params = STRATEGY.get_llm_params(strategy_defaults)
    text_temperature = _coerce_float(
        os.getenv("TEXT_TEMPERATURE"),
        _coerce_float(str(default_params.get("temperature", DEFAULT_TEXT_TEMPERATURE)), DEFAULT_TEXT_TEMPERATURE),
    )
    text_max_tokens = _coerce_positive_int(
        os.getenv("TEXT_MAX_TOKENS"),
        _coerce_positive_int(str(default_params.get("max_tokens", DEFAULT_TEXT_MAX_TOKENS)), DEFAULT_TEXT_MAX_TOKENS),
    )
    image_strategy_notes = str(getattr(STRATEGY, "image_strategy_notes", "") or "").strip()

    # Get agent ID and name from environment.
    agent_id = (
        os.getenv("AGENT_ID")
        or str(getattr(STRATEGY, "agent_id", "")).strip()
        or DEFAULT_AGENT_ID
    ).strip()
    agent_name = (
        os.getenv("AGENT_NAME")
        or str(getattr(STRATEGY, "agent_name", "")).strip()
        or DEFAULT_AGENT_NAME
    ).strip()
    if not agent_id:
        agent_id = DEFAULT_AGENT_ID
    if not agent_name:
        agent_name = agent_id
    os.environ["AGENT_ID"] = agent_id
    os.environ["AGENT_NAME"] = agent_name

    api_base = get_api_base()
    mcp_url = get_mcp_url()
    api_key = get_arena_api_key()
    timeout_env = os.getenv("ARENA_HTTP_TIMEOUT_S", "90")
    try:
        http_timeout_s = float(timeout_env)
        if http_timeout_s <= 0:
            http_timeout_s = 90.0
    except ValueError:
        http_timeout_s = 90.0

    http_client = HttpArenaClient(
        api_base=api_base,
        api_key=api_key,
        timeout=http_timeout_s,
    )

    await _register_agent(http_client, agent_id, agent_name)
    print(f"[agent] registered: {agent_id} ({agent_name})", flush=True)
    session_monitor = monitor_session(http_client, agent_id).start()

    async def local_broadcast(message: str) -> None:
        await _broadcast_thought(http_client, agent_id, message)

    await _wait_for_start_gate(http_client, agent_id, local_broadcast)
    usage_scope = http_client.fetch_usage_scope() or resolve_usage_scope()
    if usage_scope:
        os.environ["ARENA_USAGE_SCOPE"] = usage_scope

    live_prompt_tokens = 0
    live_completion_tokens = 0
    live_total_tokens = 0
    active_model_name = ""
    metrics_started_ms = 0.0
    reporter_stop = asyncio.Event()
    metrics_task: asyncio.Task[None] | None = None

    def _elapsed_metrics_ms() -> int:
        if metrics_started_ms <= 0:
            return 0
        return int(max(0.0, time.time() * 1000 - metrics_started_ms))

    async def _refresh_live_usage() -> None:
        nonlocal live_prompt_tokens, live_completion_tokens, live_total_tokens
        try:
            usage = await asyncio.to_thread(
                _fetch_proxy_usage,
                proxy_host,
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
        metrics = _build_live_metrics(
            active_model_name,
            {
                "prompt_tokens": live_prompt_tokens,
                "completion_tokens": live_completion_tokens,
                "total_tokens": live_total_tokens,
            },
            _elapsed_metrics_ms(),
        )
        await _update_status(http_client, agent_id, status, metrics)

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

    async with McpArenaClient(mcp_url) as mcp_client:
        tools = await mcp_client.list_tools()
        tool_defs = await mcp_client.list_tool_defs()
        print(f"[agent] available tools: {tools}", flush=True)

        modality = McpArenaClient.detect_modality(tools)
        if modality == "image":
            while True:
                try:
                    image_challenge = await mcp_client.get_image_challenge(agent_id)
                    break
                except McpArenaError as exc:
                    message = str(exc).lower()
                    if "locked" in message or "waiting for organizer start" in message:
                        print("[agent] Waiting for GO signal from organizer...", flush=True)
                        await asyncio.sleep(1.0)
                        continue
                    raise

            print(f"[agent] image challenge: {image_challenge}", flush=True)
            challenge_prompt = (image_challenge.prompt or image_challenge.description or "").strip()
            planner_context = "\n".join(
                value
                for value in (
                    image_challenge.prompt,
                    image_challenge.reference_notes,
                    image_strategy_notes,
                )
                if isinstance(value, str) and value.strip()
            )
            available_models = fetch_available_models(proxy_host, llm_api_key)
            image_model_ctx = _build_strategy_context(
                challenge_type=image_challenge.challenge_type,
                difficulty=str(getattr(image_challenge, "difficulty", "unknown")),
                challenge_text=challenge_prompt,
                description=image_challenge.description or "",
                rules=planner_context,
                max_time_s=image_challenge.max_time_s,
                available_models=available_models,
                image_url=image_challenge.input_image_uri or None,
            )
            ranked_models = STRATEGY.rank_models(image_model_ctx, available_models)
            model_name = STRATEGY.pick_model("solve", ranked_models, image_model_ctx)
            model_name = require_explicit_model(
                model_name,
                available_models,
                source="python_reference agent",
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
            print(f"[agent] selected model: {model_name}", flush=True)
            await _broadcast_thought(http_client, agent_id, f"🤖 Selected model: {model_name}")
            await _start_metrics_reporter(model_name)

            async def broadcast_image(message: str) -> None:
                try:
                    await mcp_client.broadcast_image_thought(message, agent_id)
                except Exception:
                    await _broadcast_thought(http_client, agent_id, message)

            image_edit_attempts = min(
                _coerce_positive_int(os.getenv("IMAGE_EDIT_MAX_ATTEMPTS"), 3),
                6,
            )
            start_time = time.monotonic()
            image_work_deadline = start_time + _derive_llm_solve_timeout_s(
                image_challenge.max_time_s
            )

            async def call_image_edit_with_retry(prompt: str) -> dict[str, Any]:
                last_result: dict[str, Any] = {"error": "image_edit failed"}
                for attempt in range(1, image_edit_attempts + 1):
                    try:
                        current_result = await _call_image_tool_before_deadline(
                            mcp_client,
                            "image_edit",
                            {
                                "image_uri": image_challenge.input_image_uri,
                                "prompt": prompt,
                                "agent_id": agent_id,
                                **image_tool_proxy_args.get("image_edit", {}),
                            },
                            deadline=image_work_deadline,
                        )
                    except Exception as exc:
                        current_result = {"error": str(exc)}
                    if isinstance(current_result, dict):
                        maybe_uri = current_result.get("image_uri")
                        if isinstance(maybe_uri, str) and maybe_uri.strip():
                            return current_result
                        if current_result.get("error"):
                            last_result = current_result
                        else:
                            last_result = {
                                **current_result,
                                "error": "image_edit returned no image_uri",
                            }
                    else:
                        return {"raw": str(current_result)}
                    if attempt < image_edit_attempts:
                        delay_s = float(2 ** (attempt - 1))
                        error_text = _truncate(str(last_result.get("error") or "unknown error"), 150)
                        await broadcast_image(
                            f"⚠️ image_edit attempt {attempt}/{image_edit_attempts} failed; "
                            f"retrying in {delay_s:.0f}s. ({error_text})"
                        )
                        await asyncio.sleep(delay_s)
                return last_result

            await broadcast_image("🖼️ Starting image challenge...")

            image_tools = [
                tool_name
                for tool_name in ("image_edit", "image_generate", "image_analyze")
                if tool_name in tools
            ]
            image_tool_ctx = _build_strategy_context(
                challenge_type=image_challenge.challenge_type,
                difficulty=str(getattr(image_challenge, "difficulty", "unknown")),
                challenge_text=challenge_prompt,
                description=image_challenge.description or "",
                rules=planner_context,
                max_time_s=image_challenge.max_time_s,
                available_models=ranked_models,
                image_url=image_challenge.input_image_uri or None,
            )
            strategy_tool = STRATEGY.plan_image_tool(image_tool_ctx, image_tools)
            strategy_image_prompt = STRATEGY.build_image_prompt(image_tool_ctx).strip()
            plan, usage, ttft_ms = _plan_image_tool_call(
                model_name=model_name,
                challenge_type=image_challenge.challenge_type,
                challenge_description=image_challenge.description,
                challenge_prompt=challenge_prompt,
                reference_notes=image_challenge.reference_notes,
                has_input_image=bool(image_challenge.input_image_uri),
                available_tools=image_tools,
            )
            selected_tool = (strategy_tool or plan.get("tool") or "").strip()
            selected_prompt = (
                strategy_image_prompt
                or (plan.get("prompt") or challenge_prompt).strip()
            )
            selected_question = (plan.get("question") or selected_prompt or challenge_prompt).strip()
            selected_rationale = (
                plan.get("rationale")
                or "Selected by strategy image tool planner."
            ).strip()

            await broadcast_image(
                f"🧭 Planned tool: {selected_tool or 'none'}"
                + (f" ({selected_rationale})" if selected_rationale else "")
            )

            tool_result: dict[str, Any] = {}
            if (
                selected_tool == "image_analyze"
                and "image_analyze" in image_tools
                and image_challenge.input_image_uri
            ):
                analyze_result = await _call_image_tool_before_deadline(
                    mcp_client,
                    "image_analyze",
                    {
                        "image_uri": image_challenge.input_image_uri,
                        "question": selected_question or challenge_prompt,
                        "agent_id": agent_id,
                        **image_tool_proxy_args.get("image_analyze", {}),
                    },
                    deadline=image_work_deadline,
                )
                analysis_text = str(analyze_result.get("text") or "").strip()
                if analysis_text:
                    await broadcast_image(f"🔍 Analysis: {analysis_text[:180]}")
                enriched_prompt = selected_prompt or challenge_prompt
                if analysis_text:
                    enriched_prompt = (
                        f"{enriched_prompt}\n\nImage analysis context:\n{analysis_text}"
                    ).strip()
                if image_challenge.input_image_uri and "image_edit" in image_tools:
                    tool_result = await call_image_edit_with_retry(enriched_prompt)
                    selected_tool = "image_edit"
                elif "image_generate" in image_tools:
                    tool_result = await _call_image_tool_before_deadline(
                        mcp_client,
                        "image_generate",
                        {
                            "prompt": enriched_prompt,
                            "agent_id": agent_id,
                            **image_tool_proxy_args.get("image_generate", {}),
                        },
                        deadline=image_work_deadline,
                    )
                    selected_tool = "image_generate"
                else:
                    tool_result = analyze_result
            elif (
                selected_tool == "image_edit"
                and "image_edit" in image_tools
                and image_challenge.input_image_uri
            ):
                tool_result = await call_image_edit_with_retry(selected_prompt or challenge_prompt)
            elif selected_tool == "image_generate" and "image_generate" in image_tools:
                tool_result = await _call_image_tool_before_deadline(
                    mcp_client,
                    "image_generate",
                    {
                        "prompt": selected_prompt or challenge_prompt,
                        "agent_id": agent_id,
                        **image_tool_proxy_args.get("image_generate", {}),
                    },
                    deadline=image_work_deadline,
                )
            elif image_challenge.input_image_uri and "image_edit" in image_tools:
                selected_tool = "image_edit"
                tool_result = await call_image_edit_with_retry(challenge_prompt)
            elif "image_generate" in image_tools:
                selected_tool = "image_generate"
                tool_result = await _call_image_tool_before_deadline(
                    mcp_client,
                    "image_generate",
                    {
                        "prompt": challenge_prompt,
                        "agent_id": agent_id,
                        **image_tool_proxy_args.get("image_generate", {}),
                    },
                    deadline=image_work_deadline,
                )

            if isinstance(tool_result, dict) and tool_result.get("error"):
                await broadcast_image(f"⚠️ {selected_tool} failed: {tool_result.get('error')}")

            output_image_uri = ""
            tool_model_name = ""
            if isinstance(tool_result, dict):
                maybe_uri = tool_result.get("image_uri")
                if isinstance(maybe_uri, str):
                    output_image_uri = maybe_uri.strip()
                maybe_model_name = tool_result.get("model")
                if isinstance(maybe_model_name, str):
                    tool_model_name = maybe_model_name.strip()

            if not output_image_uri and image_challenge.input_image_uri:
                output_image_uri = image_challenge.input_image_uri
                await broadcast_image("⚠️ Using original image as fallback output.")
            if not output_image_uri:
                output_image_uri = _BLANK_PNG_DATA_URI
                await broadcast_image("⚠️ Using blank fallback image output.")

            total_time_ms = int((time.monotonic() - start_time) * 1000)
            await _stop_metrics_reporter()
            effective_model_name = tool_model_name or model_name
            client_metrics: dict[str, Any] = {
                "model_name": effective_model_name,
                "planner_model_name": model_name,
                "planner_tool": selected_tool,
                "total_tokens": str(live_total_tokens),
                "prompt_tokens": str(live_prompt_tokens),
                "completion_tokens": str(live_completion_tokens),
                "ttft_ms": ttft_ms,
                "total_time_ms": total_time_ms,
            }
            await broadcast_image("📤 Submitting image output...")
            try:
                result = await mcp_client.submit_image(
                    agent_id=agent_id,
                    image_uri=output_image_uri,
                    client_metrics=client_metrics,
                    rationale=selected_rationale,
                )
            except Exception as exc:
                await broadcast_image(f"❌ Image submission failed: {exc}")
                await _update_status(http_client, agent_id, "failed")
                print(f"[agent] image submission failed: {exc}", flush=True)
                return
            if isinstance(result, dict):
                log_result = dict(result)
                log_result.pop("edited_image", None)
                log_result.pop("image_uri", None)
                print(f"[agent] image submission result: {log_result}", flush=True)
            else:
                print(f"[agent] image submission result: {result}", flush=True)
            await session_monitor.stop()
            return

        while True:
            try:
                challenge = await mcp_client.get_challenge(agent_id)
                break
            except McpArenaError as exc:
                message = str(exc).lower()
                if "locked" in message or "waiting for organizer start" in message:
                    print("[agent] Waiting for GO signal from organizer...", flush=True)
                    await asyncio.sleep(1.0)
                    continue
                raise
        print(f"[agent] challenge: {challenge}", flush=True)
        available_models = fetch_available_models(proxy_host, llm_api_key)
        model_ctx = _build_strategy_context(
            challenge_type=challenge.challenge_type,
            difficulty=str(getattr(challenge, "difficulty", "unknown")),
            challenge_text=challenge.description or "",
            description=challenge.description or "",
            rules=challenge.rules or "",
            max_time_s=challenge.max_time_s,
            available_models=available_models,
        )
        ranked_models = STRATEGY.rank_models(model_ctx, available_models)
        model_name = STRATEGY.pick_model("solve", ranked_models, model_ctx)
        model_name = require_explicit_model(
            model_name,
            available_models,
            source="python_reference agent",
        )
        print(f"[agent] selected model: {model_name}", flush=True)
        await _broadcast_thought(http_client, agent_id, f"🤖 Selected model: {model_name}")
        await _start_metrics_reporter(model_name)

        start_time = time.monotonic()
        await _broadcast_thought(http_client, agent_id, "🚀 Starting challenge...")

        challenge_type = (challenge.challenge_type or "logic-puzzle").lower()
        rules_text = challenge.rules or ""
        extra_context = str(getattr(challenge, "extra_text", "") or "")
        challenge_blob = _challenge_text_blob(
            challenge_type,
            challenge.description or "",
            rules_text,
            extra_context,
        )
        available_tool_names = [
            str(getattr(tool, "name", "") or "").strip()
            for tool in tools
            if str(getattr(tool, "name", "") or "").strip()
        ]
        required_runtime_tools = [
            tool_name
            for tool_name in available_tool_names
            if tool_name.lower() in challenge_blob
        ]
        if challenge_type in {"web-search", "market-research"}:
            for tool_name in available_tool_names:
                if "search" in tool_name.lower() and tool_name not in required_runtime_tools:
                    required_runtime_tools.append(tool_name)
        requires_search = bool(required_runtime_tools)
        if requires_search:
            question = (
                challenge.description
                or challenge.rules
                or extra_context
                or ""
            )
            rules = challenge.rules or None

            await _broadcast_thought(
                http_client,
                agent_id,
                "🔎 This challenge requires search/tool verification.",
            )
            await _broadcast_thought(http_client, agent_id, f"🧠 Searching for: {question}")

            tool_defs = await mcp_client.list_tool_defs()
            search_tool_name = _find_search_tool(tool_defs)
            if not search_tool_name and required_runtime_tools:
                search_tool_name = required_runtime_tools[0]
            search_runs: list[dict[str, Any]] = []
            clues_list = [str(c) for c in (challenge.clues or []) if isinstance(c, str)]
            query_candidates = _query_candidates(question, rules, clues_list, extra_context)
            if not query_candidates:
                query_candidates = [question]
            if search_tool_name:
                query_param = _coerce_search_query_param(tool_defs, search_tool_name) or "query"
                max_calls = min(len(query_candidates), 3) if "search" in search_tool_name.lower() else 1
                successful_calls = 0
                for query in query_candidates[:max_calls]:
                    payload_variants = [
                        {"agent_id": agent_id, query_param: query},
                        {query_param: query},
                        {"agent_id": agent_id, "query": query},
                        {"query": query},
                        {"agent_id": agent_id, "url": query},
                        {"url": query},
                        {"agent_id": agent_id, "video_url": query},
                        {"video_url": query},
                        {"agent_id": agent_id},
                        {},
                    ]
                    called = False
                    for args in payload_variants:
                        try:
                            current_result = await mcp_client.call_tool(search_tool_name, args)
                            search_runs.append(
                                {
                                    "tool": search_tool_name,
                                    "query": query,
                                    "result": current_result,
                                }
                            )
                            if isinstance(current_result, dict) and current_result.get("error"):
                                await _broadcast_thought(
                                    http_client,
                                    agent_id,
                                    f"⚠️ Search tool returned error: {current_result.get('error')}",
                                )
                            else:
                                successful_calls += 1
                            called = True
                            break
                        except Exception:
                            continue
                    if not called:
                        await _broadcast_thought(
                            http_client,
                            agent_id,
                            f"⚠️ Could not call tool for query: {_truncate(query, 80)}",
                        )
                await _broadcast_thought(
                    http_client,
                    agent_id,
                    f"🔧 Tool coverage {search_tool_name}: {successful_calls}/{max_calls}",
                )
            else:
                await _broadcast_thought(
                    http_client,
                    agent_id,
                    "⚠️ No search tool is available on this server.",
                )
            if not search_runs:
                search_runs = [{"tool": search_tool_name or "", "query": question, "result": {}}]
            search_result: dict[str, Any] = {
                "tool": search_tool_name or "",
                "query": question,
                "runs": search_runs,
            }

            await _broadcast_thought(
                http_client,
                agent_id,
                "✅ Got web search results. Extracting the answer...",
            )

            answer: str | None = None
            usage: dict[str, Any] | None = None
            ttft_ms = 0

            answer, usage, ttft_ms = try_extract_search_answer_with_llm(
                question=question,
                rules=rules,
                search_result=search_result,
                model_name=model_name,
            )
            if _is_invalid_meta_answer_candidate(answer):
                answer = None
            if answer and _requires_names_only_output(rules):
                answer = _normalize_name_order_answer(answer)
            if not answer and _requires_names_only_output(rules):
                answer = _extract_name_order_from_rules(rules)
            if not answer:
                answer = _extract_answer_from_search_result(search_result)
            if _is_invalid_meta_answer_candidate(answer):
                answer = None
            if answer and _requires_names_only_output(rules):
                answer = _normalize_name_order_answer(answer)
            if not answer:
                answer = "unknown"

            time_remaining = await mcp_client.time_remaining(agent_id)
            remaining_s = float(time_remaining.get("time_remaining_s", 0.0))
            search_submit_ctx = _build_strategy_context(
                challenge_type=challenge.challenge_type or "text",
                difficulty=str(getattr(challenge, "difficulty", "unknown")),
                challenge_text=question,
                description=challenge.description or "",
                rules=challenge.rules or "",
                max_time_s=challenge.max_time_s,
                available_models=ranked_models,
                time_remaining_s=remaining_s,
                tools_used=[search_tool_name] if search_tool_name else [],
                tokens_used=int(usage.get("total_tokens", 0)) if usage else 0,
            )
            if STRATEGY.should_submit_early(answer, search_submit_ctx):
                await _broadcast_thought(http_client, agent_id, "⚡ Strategy submitting early.")
            revised_answer = STRATEGY.on_time_warning(remaining_s, answer, search_submit_ctx)
            if isinstance(revised_answer, str) and revised_answer.strip():
                answer = revised_answer.strip()
            if not _is_valid_submission_candidate(answer, rules):
                repaired, repair_usage, repair_ttft = _repair_final_answer_with_llm(
                    question=question,
                    rules=rules,
                    clues=clues_list,
                    current_answer=answer,
                    reasoning=_format_search_results_for_llm(search_result),
                    model_name=model_name,
                )
                if repaired and _is_valid_submission_candidate(repaired, rules):
                    answer = repaired
                    if repair_usage:
                        usage = repair_usage
                    if repair_ttft:
                        ttft_ms = repair_ttft
            if not _is_valid_submission_candidate(answer, rules):
                answer = "unknown"

            await _broadcast_thought(http_client, agent_id, f"🎯 Final answer: {answer}")
            await _save_draft(http_client, agent_id, answer, "Draft saved before submit.")

            await _stop_metrics_reporter()
            total_time_ms = int((time.monotonic() - start_time) * 1000)
            client_metrics: dict[str, Any] = {
                "model_name": model_name,
                "total_tokens": str(live_total_tokens),
                "prompt_tokens": str(live_prompt_tokens),
                "completion_tokens": str(live_completion_tokens),
                "ttft_ms": ttft_ms,
                "total_time_ms": total_time_ms,
            }

            print(f"[agent] submitting with metrics: {client_metrics}", flush=True)
            result = await _submit_answer(http_client, agent_id, answer, client_metrics)
            print(f"[agent] submission result: {result}", flush=True)
            return

        clue_ids = await mcp_client.list_clues(agent_id)
        clue_texts = []
        for clue_id in clue_ids:
            clue = await mcp_client.get_clue(clue_id, agent_id)
            clue_texts.append(clue.text)

        await _broadcast_thought(
            http_client,
            agent_id,
            f"📝 Received {len(clue_texts)} clues to analyze",
        )
        solve_ctx = _build_strategy_context(
            challenge_type=challenge.challenge_type or "text",
            difficulty=str(getattr(challenge, "difficulty", "unknown")),
            challenge_text=challenge.description or "",
            description=challenge.description or "",
            rules=challenge.rules or "",
            clues=clue_texts,
            max_time_s=challenge.max_time_s,
            available_models=ranked_models,
            required_tools=list(getattr(challenge, "required_tools", []) or []),
        )
        llm_params = STRATEGY.get_llm_params(solve_ctx)
        text_temperature = _coerce_float(
            os.getenv("TEXT_TEMPERATURE"),
            _coerce_float(
                str(llm_params.get("temperature", text_temperature)),
                text_temperature,
            ),
        )
        text_max_tokens = _coerce_positive_int(
            os.getenv("TEXT_MAX_TOKENS"),
            _coerce_positive_int(
                str(llm_params.get("max_tokens", text_max_tokens)),
                text_max_tokens,
            ),
        )
        system_prompt = (
            STRATEGY.build_system_prompt(solve_ctx).strip() or DEFAULT_TEXT_SYSTEM_PROMPT
        )
        solver_prompt = STRATEGY.build_solver_prompt(solve_ctx)
        ttft_ms = 0

        answer = "unknown"
        reasoning: str | None = None
        usage: dict[str, Any] | None = None

        async def broadcast(message: str) -> None:
            await _broadcast_thought(http_client, agent_id, message)

        print("[agent] attempting LLM solve", flush=True)
        await broadcast("🤖 Calling LLM to solve the puzzle...")
        answer, reasoning, usage, ttft_ms, reasoning_streamed = await try_solve_with_llm(
            clue_texts,
            model_name,
            broadcast,
            text_temperature,
            text_max_tokens,
            challenge.challenge_type or "text",
            challenge.description or "",
            challenge.rules or "",
            system_prompt,
            solver_prompt,
            solve_timeout_s=_derive_llm_solve_timeout_s(challenge.max_time_s),
        )

        if reasoning and not reasoning_streamed:
            print(
                f"[agent] reasoning: {reasoning[:500]}{'...' if len(reasoning) > 500 else ''}",
                flush=True,
            )
            await broadcast("💭 LLM Reasoning:")
            for chunk in _iter_reasoning_chunks(reasoning):
                await broadcast(f"   {chunk}")
                await asyncio.sleep(0.12)

        if _is_invalid_meta_answer_candidate(answer):
            answer = None

        if answer:
            await broadcast(f"✅ Found answer: {answer}")
        elif reasoning:
            # No clean answer extracted — use LLM judge
            await broadcast("🔍 Answer unclear — calling LLM judge to extract...")
            print("[agent] calling LLM judge for answer extraction", flush=True)
            raw_text = reasoning
            judge_model = (os.getenv("JUDGE_MODEL") or model_name).strip()
            answer = await _async_judge_extract(raw_text, judge_model)
            if _is_invalid_meta_answer_candidate(answer):
                answer = None
            if answer:
                await broadcast(f"✅ Judge extracted answer: {answer}")
                print(f"[agent] judge extracted: {answer}", flush=True)
            else:
                await broadcast("⚠️ Judge could not extract answer")
                print("[agent] judge extraction failed", flush=True)
        else:
            await broadcast("⚠️ LLM returned no output")

        if not answer:
            answer = "unknown"
            await broadcast("⚠️ No valid answer found. Submitting 'unknown'.")

        time_remaining = await mcp_client.time_remaining(agent_id)
        remaining_s = float(time_remaining.get("time_remaining_s", 0.0))
        submit_ctx = _build_strategy_context(
            challenge_type=challenge.challenge_type or "text",
            difficulty=str(getattr(challenge, "difficulty", "unknown")),
            challenge_text=challenge.description or "",
            description=challenge.description or "",
            rules=challenge.rules or "",
            clues=clue_texts,
            max_time_s=challenge.max_time_s,
            available_models=ranked_models,
            time_remaining_s=remaining_s,
            tools_used=clue_ids,
            tokens_used=int(usage.get("total_tokens", 0)) if usage else 0,
            required_tools=list(getattr(challenge, "required_tools", []) or []),
        )
        if STRATEGY.should_submit_early(answer, submit_ctx):
            await broadcast("⚡ Strategy submitting early.")
        revised_answer = STRATEGY.on_time_warning(remaining_s, answer, submit_ctx)
        if isinstance(revised_answer, str) and revised_answer.strip():
            answer = revised_answer.strip()
        if _is_invalid_meta_answer_candidate(answer):
            answer = "unknown"
        if (
            _has_final_repair_budget(remaining_s)
            and not _is_valid_submission_candidate(answer, challenge.rules or "")
        ):
            repaired, repair_usage, repair_ttft = _repair_final_answer_with_llm(
                question=challenge.description or "",
                rules=challenge.rules or "",
                clues=clue_texts,
                current_answer=answer,
                reasoning=reasoning,
                model_name=model_name,
            )
            if repaired and _is_valid_submission_candidate(repaired, challenge.rules or ""):
                answer = repaired
                if repair_usage:
                    usage = repair_usage
                if repair_ttft:
                    ttft_ms = repair_ttft
        if not _is_valid_submission_candidate(answer, challenge.rules or ""):
            answer = "unknown"

        print(f"[agent] solved answer: {answer}", flush=True)
        await broadcast(f"🎯 Final answer: {answer}")
        await _save_draft(http_client, agent_id, answer, "Draft saved before submit.")

        await _stop_metrics_reporter()
        total_time_ms = int((time.monotonic() - start_time) * 1000)
        client_metrics: dict[str, Any] = {
            "model_name": model_name,
            "total_tokens": str(live_total_tokens),
            "prompt_tokens": str(live_prompt_tokens),
            "completion_tokens": str(live_completion_tokens),
            "ttft_ms": ttft_ms,
            "total_time_ms": total_time_ms,
        }

        print(f"[agent] submitting with metrics: {client_metrics}", flush=True)
        result = await _submit_answer(http_client, agent_id, answer, client_metrics)
        print(f"[agent] submission result: {result}", flush=True)
        await session_monitor.stop()


def main() -> int:
    try:
        asyncio.run(run())
    except asyncio.CancelledError:
        return 0
    except ModelSelectionError as exc:
        print(f"[agent] {exc}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
