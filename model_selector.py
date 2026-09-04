"""Model discovery helpers for starter-kit agents.

The bundled example runtimes require teams to choose a model explicitly via
`rank_models()` / `pick_model()`. `fetch_available_models()` remains the
supported way to inspect the live proxy roster.
"""

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from arena_clients.config import get_arena_api_key, get_proxy_host

# Last /models rows from fetch_available_models / fetch_available_model_rows.
# rank_models and prefer_image_models consult this so the capabilities field
# survives the ID-only list the example agents pass through.
_LAST_MODEL_ROWS: list[dict[str, Any]] = []


class ModelSelectionError(RuntimeError):
    """Raised when the runtime does not have a valid proxy model choice."""


def _format_model_list(available_models: list[str], limit: int = 8) -> str:
    preview = [str(model).strip() for model in available_models if str(model).strip()]
    if not preview:
        return "(no models discovered)"
    if len(preview) <= limit:
        return ", ".join(preview)
    return ", ".join(preview[:limit]) + ", ..."


def require_available_models(available_models: list[str]) -> list[str]:
    """Require a non-empty proxy roster before selecting a model."""
    if available_models:
        return available_models
    raise ModelSelectionError(
        "No models are available from the LLM proxy. Verify proxy connectivity "
        "and ARENA_API_KEY/LLM_PROXY_HOST before starting the agent."
    )


def require_explicit_model(
    model_name: str,
    available_models: list[str],
    *,
    source: str = "agent",
) -> str:
    """Validate that an explicit model choice exists in the proxy roster."""
    models = require_available_models(available_models)
    normalized = str(model_name or "").strip()
    if not normalized:
        raise ModelSelectionError(
            f"{source} did not choose a model. Override `pick_model()` or ensure "
            "`rank_models()` returns at least one valid alias from `/models`."
        )
    if normalized not in models:
        raise ModelSelectionError(
            f"{source} selected '{normalized}', but it is not in the proxy model "
            f"roster: {_format_model_list(models)}"
        )
    return normalized


def resolve_proxy_api_key(api_key: str = "") -> str:
    """Resolve proxy auth key for competitor access."""
    explicit = (api_key or "").strip()
    if explicit:
        return explicit
    return get_arena_api_key()


def _parse_proxy_model_rows(payload: Any) -> list[dict[str, Any]]:
    """Parse /models payloads, keeping id and capabilities when present."""
    raw_rows: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, list):
            for item in data:
                row = _row_from_models_item(item)
                if row is not None:
                    raw_rows.append(row)
        elif isinstance(payload.get("models"), list):
            for item in payload.get("models", []):
                row = _row_from_models_item(item)
                if row is not None:
                    raw_rows.append(row)
    elif isinstance(payload, list):
        for item in payload:
            row = _row_from_models_item(item)
            if row is not None:
                raw_rows.append(row)

    seen: set[str] = set()
    ordered: list[dict[str, Any]] = []
    for row in raw_rows:
        model_id = str(row.get("id") or "").strip()
        if not model_id or model_id in seen:
            continue
        seen.add(model_id)
        ordered.append(row)
    return ordered


def _row_from_models_item(item: Any) -> dict[str, Any] | None:
    if isinstance(item, str) and item.strip():
        return {"id": item.strip()}
    if not isinstance(item, dict):
        return None
    model_id = item.get("id")
    if not isinstance(model_id, str) or not model_id.strip():
        return None
    row: dict[str, Any] = {"id": model_id.strip()}
    if "capabilities" in item:
        row["capabilities"] = item["capabilities"]
    return row


def _parse_proxy_model_ids(payload: Any) -> list[str]:
    return [
        str(row["id"]).strip()
        for row in _parse_proxy_model_rows(payload)
        if str(row.get("id") or "").strip()
    ]


def cached_model_rows() -> list[dict[str, Any]]:
    """Return the last /models rows captured by fetch_available_models."""
    return list(_LAST_MODEL_ROWS)


def fetch_available_model_rows(
    proxy_host: str | None = None, api_key: str = ""
) -> list[dict[str, Any]]:
    """Fetch /models rows, preserving id and capabilities."""
    global _LAST_MODEL_ROWS
    resolved_proxy_host = get_proxy_host(proxy_host)
    url = f"{resolved_proxy_host.rstrip('/')}/models"
    headers = {"Accept": "application/json"}
    resolved_key = resolve_proxy_api_key(api_key)
    if resolved_key:
        headers["Authorization"] = f"Bearer {resolved_key}"
    request = Request(url, headers=headers, method="GET")

    try:
        with urlopen(request, timeout=3.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (URLError, TimeoutError, json.JSONDecodeError, HTTPError):
        _LAST_MODEL_ROWS = []
        return []

    rows = _parse_proxy_model_rows(payload)
    _LAST_MODEL_ROWS = rows
    return list(rows)


def fetch_available_models(proxy_host: str | None = None, api_key: str = "") -> list[str]:
    """Fetch available model IDs from the proxy /models endpoint."""
    return [
        str(row["id"]).strip()
        for row in fetch_available_model_rows(proxy_host, api_key)
        if str(row.get("id") or "").strip()
    ]


def prefer_image_models(
    available_models: list[str],
    *,
    model_rows: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Return image-capable aliases, preserving the input order.

    When a /models row carries ``capabilities``, keep the alias only if
    ``"image"`` is listed. When the field is absent (practice LiteLLM), fall
    back to the ``*-image`` suffix convention.
    """
    rows = list(model_rows) if model_rows is not None else list(_LAST_MODEL_ROWS)
    caps_by_id: dict[str, set[str] | None] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        model_id = str(row.get("id") or "").strip()
        if not model_id:
            continue
        if "capabilities" not in row:
            caps_by_id[model_id] = None
            continue
        caps = row.get("capabilities")
        if not isinstance(caps, list):
            caps_by_id[model_id] = None
            continue
        caps_by_id[model_id] = {
            str(capability).strip().casefold()
            for capability in caps
            if str(capability).strip()
        }

    preferred: list[str] = []
    seen: set[str] = set()
    for model in available_models:
        name = str(model or "").strip()
        if not name or name in seen:
            continue
        caps = caps_by_id.get(name)
        if caps is not None:
            keep = "image" in caps
        else:
            alias = name.rsplit("/", 1)[-1].casefold()
            keep = alias.endswith("-image") or "-image-" in alias
        if keep:
            seen.add(name)
            preferred.append(name)
    return preferred


def select_model(
    challenge_type: str,
    challenge_description: str,
    challenge_rules: str,
    max_time_s: int,
    available_models: list[str],
    proxy_host: str | None = None,
    api_key: str = "",
) -> str:
    """Opt-in autonomous selection; available only with organizer heuristics.

    The published participant package does not ship scoring/classification
    heuristics. Override `rank_models()` / `pick_model()` instead.
    """
    available_models = require_available_models(available_models)
    try:
        from model_selector_heuristics import select_model as _select_model
    except ImportError as exc:
        raise ModelSelectionError(
            "Autonomous model-selection heuristics are not part of the published "
            "starter kit. Override `rank_models()` / `pick_model()` to choose a "
            "model from `fetch_available_models()`."
        ) from exc

    return _select_model(
        challenge_type=challenge_type,
        challenge_description=challenge_description,
        challenge_rules=challenge_rules,
        max_time_s=max_time_s,
        available_models=available_models,
        proxy_host=proxy_host,
        api_key=api_key,
    )
