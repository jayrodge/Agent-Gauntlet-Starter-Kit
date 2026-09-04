from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any

from crewai.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr, create_model

from arena_clients import McpArenaClient

MAX_TOOL_RESULT_CHARS = 8000
INSTRUCTION_FIELD_PREFERENCES = (
    "prompt",
    "question",
    "instruction",
    "instructions",
    "request",
    "query",
    "text",
    "task",
    "message",
    "caption",
    "edit_prompt",
)
IMAGE_HINT_KEYWORDS = (
    "image",
    "photo",
    "picture",
    "visual",
    "vision",
    "face",
    "blur",
    "mask",
    "crop",
    "render",
    "draw",
    "inpaint",
    "outpaint",
    "upscale",
)
IMAGE_OUTPUT_HINT_KEYWORDS = (
    "generate",
    "generated",
    "create",
    "created",
    "edit",
    "edited",
    "render",
    "draw",
    "paint",
    "blur",
    "mask",
    "crop",
    "transform",
    "stylize",
    "upscale",
    "inpaint",
    "outpaint",
)
IMAGE_ANALYSIS_HINT_KEYWORDS = (
    "analyze",
    "analysis",
    "describe",
    "detect",
    "detection",
    "classify",
    "caption",
    "ocr",
    "read",
    "count",
    "extract",
    "identify",
    "segment",
    "question",
)


def _truncate_text(text: str, max_chars: int = MAX_TOOL_RESULT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]} ...<truncated>"


def _sanitize_tool_name(name: str, used: set[str]) -> str:
    sanitized = re.sub(r"[^a-z0-9_]+", "_", str(name or "").strip().lower())
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    if not sanitized:
        sanitized = "arena_tool"
    if not (sanitized[0].isalpha() or sanitized[0] == "_"):
        sanitized = f"arena_{sanitized}"
    candidate = sanitized
    suffix = 2
    while candidate in used:
        candidate = f"{sanitized}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def _schema_properties(input_schema: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    properties = input_schema.get("properties") if isinstance(input_schema, dict) else None
    if not isinstance(properties, dict):
        return {}
    normalized: dict[str, dict[str, Any]] = {}
    for field_name, field_schema in properties.items():
        normalized[str(field_name)] = field_schema if isinstance(field_schema, dict) else {}
    return normalized


def _looks_like_image_input_field(field_name: str, field_schema: dict[str, Any]) -> bool:
    lower_name = field_name.lower()
    description = str(field_schema.get("description") or "").lower()
    if "image" not in lower_name and "image" not in description:
        return False
    if lower_name in {
        "image_uri",
        "image_url",
        "input_image_uri",
        "input_image_url",
        "source_image_uri",
        "source_image_url",
    }:
        return True
    return (
        ("image" in lower_name)
        and any(token in lower_name for token in ("uri", "url", "input", "source"))
    )


@dataclass(frozen=True)
class ToolRuntimeHints:
    accepts_agent_id: bool
    accepts_proxy_api_key: bool
    image_input_field: str | None


@dataclass(frozen=True)
class ToolSpec:
    original_name: str
    sanitized_name: str
    description: str
    input_schema: dict[str, Any]
    required_fields: tuple[str, ...]
    runtime_hints: ToolRuntimeHints
    instruction_field: str | None
    image_related: bool
    likely_returns_image: bool


def _derive_runtime_hints(input_schema: dict[str, Any]) -> ToolRuntimeHints:
    properties = _schema_properties(input_schema)
    image_input_field: str | None = None

    for preferred_name in (
        "image_uri",
        "input_image_uri",
        "source_image_uri",
        "image_url",
        "input_image_url",
        "source_image_url",
    ):
        if preferred_name in properties:
            image_input_field = preferred_name
            break

    if image_input_field is None:
        for field_name, field_schema in properties.items():
            if _looks_like_image_input_field(field_name, field_schema):
                image_input_field = field_name
                break

    return ToolRuntimeHints(
        accepts_agent_id="agent_id" in properties,
        accepts_proxy_api_key="proxy_api_key" in properties,
        image_input_field=image_input_field,
    )


def _schema_type_matches(schema: dict[str, Any], expected_type: str) -> bool:
    schema_type = schema.get("type")
    if schema_type == expected_type:
        return True
    if isinstance(schema_type, list):
        return expected_type in schema_type
    return False


def _is_string_schema(schema: dict[str, Any]) -> bool:
    return _schema_type_matches(schema, "string")


def _derive_instruction_field(
    input_schema: dict[str, Any],
    runtime_hints: ToolRuntimeHints,
) -> str | None:
    properties = _schema_properties(input_schema)
    candidate_fields: list[str] = []

    for field_name, field_schema in properties.items():
        if runtime_hints.accepts_agent_id and field_name == "agent_id":
            continue
        if runtime_hints.accepts_proxy_api_key and field_name == "proxy_api_key":
            continue
        if runtime_hints.image_input_field and field_name == runtime_hints.image_input_field:
            continue
        if not _is_string_schema(field_schema):
            continue
        candidate_fields.append(field_name)

    if not candidate_fields:
        return None

    candidate_set = set(candidate_fields)
    for preferred in INSTRUCTION_FIELD_PREFERENCES:
        if preferred in candidate_set:
            return preferred

    required_fields = set(input_schema.get("required") or [])
    for field_name in candidate_fields:
        if field_name in required_fields:
            return field_name

    return candidate_fields[0]


def _tool_text_blob(
    original_name: str,
    description: str,
    input_schema: dict[str, Any],
) -> str:
    properties = _schema_properties(input_schema)
    field_parts: list[str] = []
    for field_name, field_schema in properties.items():
        field_parts.append(field_name)
        field_description = str(field_schema.get("description") or "").strip()
        if field_description:
            field_parts.append(field_description)
    return " ".join(part for part in [original_name, description, *field_parts] if part).lower()


def _has_any_keyword(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def classify_image_tool(spec: ToolSpec) -> str:
    if not spec.image_related:
        return "none"
    if spec.runtime_hints.image_input_field and spec.likely_returns_image:
        return "edit"
    if not spec.runtime_hints.image_input_field and spec.likely_returns_image:
        return "generate"
    if spec.runtime_hints.image_input_field and spec.instruction_field:
        return "analyze"
    return "other"


def unsupported_required_fields(spec: ToolSpec) -> tuple[str, ...]:
    supported_fields: set[str] = set()
    if spec.runtime_hints.accepts_agent_id:
        supported_fields.add("agent_id")
    if spec.runtime_hints.accepts_proxy_api_key:
        supported_fields.add("proxy_api_key")
    if spec.runtime_hints.image_input_field:
        supported_fields.add(spec.runtime_hints.image_input_field)
    if spec.instruction_field:
        supported_fields.add(spec.instruction_field)
    return tuple(field for field in spec.required_fields if field not in supported_fields)


def discover_tool_specs(
    tool_defs: list[Any],
    *,
    exclude_tools: set[str] | None = None,
) -> list[ToolSpec]:
    excluded = set(exclude_tools or set())
    used_names: set[str] = set()
    tool_specs: list[ToolSpec] = []

    for tool_def in tool_defs:
        original_name = str(getattr(tool_def, "name", "") or "").strip()
        if not original_name or original_name in excluded:
            continue

        sanitized_name = _sanitize_tool_name(original_name, used_names)
        input_schema = getattr(tool_def, "inputSchema", None)
        if not isinstance(input_schema, dict):
            input_schema = {}
        description = str(getattr(tool_def, "description", "") or "")
        runtime_hints = _derive_runtime_hints(input_schema)
        instruction_field = _derive_instruction_field(input_schema, runtime_hints)
        required_fields = tuple(
            str(field_name)
            for field_name in input_schema.get("required") or []
            if isinstance(field_name, str) and field_name.strip()
        )
        tool_text = _tool_text_blob(original_name, description, input_schema)
        image_related = bool(runtime_hints.image_input_field) or _has_any_keyword(
            tool_text,
            IMAGE_HINT_KEYWORDS,
        )
        likely_returns_image = _has_any_keyword(tool_text, IMAGE_OUTPUT_HINT_KEYWORDS)
        if (
            not likely_returns_image
            and image_related
            and runtime_hints.image_input_field
            and instruction_field
            and not _has_any_keyword(tool_text, IMAGE_ANALYSIS_HINT_KEYWORDS)
        ):
            likely_returns_image = True

        tool_specs.append(
            ToolSpec(
                original_name=original_name,
                sanitized_name=sanitized_name,
                description=description,
                input_schema=input_schema,
                required_fields=required_fields,
                runtime_hints=runtime_hints,
                instruction_field=instruction_field,
                image_related=image_related,
                likely_returns_image=likely_returns_image,
            )
        )

    return tool_specs


def _json_schema_to_annotation(schema: dict[str, Any] | None) -> Any:
    if not isinstance(schema, dict):
        return Any

    schema_type = schema.get("type")
    nullable = False
    if isinstance(schema_type, list):
        nullable = "null" in schema_type
        schema_type = next((item for item in schema_type if item != "null"), None)

    annotation: Any
    if schema_type == "string":
        annotation = str
    elif schema_type == "integer":
        annotation = int
    elif schema_type == "number":
        annotation = float
    elif schema_type == "boolean":
        annotation = bool
    elif schema_type == "array":
        item_schema = schema.get("items")
        annotation = list[_json_schema_to_annotation(item_schema)]
    elif schema_type == "object":
        annotation = dict[str, Any]
    else:
        annotation = Any

    if nullable:
        annotation = annotation | None
    return annotation


def _build_args_schema(
    tool_name: str,
    input_schema: dict[str, Any],
    runtime_hints: ToolRuntimeHints,
) -> type[BaseModel]:
    properties = _schema_properties(input_schema)
    required_fields = set(input_schema.get("required") or [])
    fields: dict[str, tuple[Any, Any]] = {}

    if not properties:
        schema_name = f"{tool_name.title().replace('_', '')}Args"
        return create_model(schema_name)

    for field_name, field_schema in properties.items():
        if runtime_hints.accepts_agent_id and field_name == "agent_id":
            continue
        if runtime_hints.accepts_proxy_api_key and field_name == "proxy_api_key":
            continue

        annotation = _json_schema_to_annotation(field_schema)
        description = str(field_schema.get("description") or "").strip()

        if runtime_hints.image_input_field and field_name == runtime_hints.image_input_field:
            annotation = str | None
            description = (
                f"{description} Optional: defaults to the active challenge image, or the most recent "
                "tool-produced image if one exists."
            ).strip()
            default = Field(default=None, description=description)
        elif field_name in required_fields:
            default = Field(..., description=description)
        else:
            default = Field(default=field_schema.get("default", None), description=description)

        fields[field_name] = (annotation, default)

    schema_name = f"{tool_name.title().replace('_', '')}Args"
    return create_model(schema_name, **fields)


def _sanitize_payload(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("data:image/"):
            return "<omitted data:image payload>"
        return _truncate_text(value, 4000)
    if isinstance(value, dict):
        return {str(key): _sanitize_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        trimmed = value[:20]
        sanitized_items = [_sanitize_payload(item) for item in trimmed]
        if len(value) > len(trimmed):
            sanitized_items.append(f"<{len(value) - len(trimmed)} more items omitted>")
        return sanitized_items
    return value


def _is_meaningful_image_value(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _find_input_image_field(result: dict[str, Any]) -> tuple[str | None, str]:
    if not isinstance(result, dict):
        return None, ""
    for key, value in result.items():
        value = result.get(key)
        if not _is_meaningful_image_value(value):
            continue
        lower = key.lower()
        if "image" not in lower:
            continue
        if ("input" in lower or "source" in lower) and any(
            token in lower for token in ("uri", "url", "image")
        ):
            return key, value.strip()
    return None, ""


def _find_output_image_field(result: dict[str, Any]) -> tuple[str | None, str]:
    if not isinstance(result, dict):
        return None, ""
    for key, value in result.items():
        if not _is_meaningful_image_value(value):
            continue
        lower = key.lower()
        if lower.startswith("input_") or lower.startswith("source_"):
            continue
        if value.strip().startswith("data:image/"):
            return key, value.strip()
        if "image" in lower and any(token in lower for token in ("uri", "url", "image")):
            return key, value.strip()
    return None, ""


def _extract_image_uri(result: dict[str, Any]) -> str:
    return _find_output_image_field(result)[1]


class ArenaToolState:
    def __init__(
        self,
        *,
        agent_id: str,
        mcp_url: str,
        api_key: str | None,
        challenge_image_uri: str = "",
        tool_argument_overrides: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.mcp_url = mcp_url
        self.api_key = api_key
        self.challenge_image_uri = challenge_image_uri.strip()
        self.tool_argument_overrides = {
            str(tool_name): dict(arguments)
            for tool_name, arguments in (tool_argument_overrides or {}).items()
            if isinstance(tool_name, str) and isinstance(arguments, dict)
        }
        self.latest_image_uri = ""
        self.last_image_tool = ""
        self.tool_name_map: dict[str, str] = {}
        self.latest_results: dict[str, dict[str, Any]] = {}

    def current_image_uri(self) -> str:
        return self.latest_image_uri or self.challenge_image_uri

    def seed_challenge_image(self, image_uri: str | None) -> None:
        if isinstance(image_uri, str) and image_uri.strip():
            self.challenge_image_uri = image_uri.strip()

    def record_result(self, original_tool_name: str, result: dict[str, Any]) -> None:
        self.latest_results[original_tool_name] = result
        _, input_image_uri = _find_input_image_field(result)
        if input_image_uri:
            self.seed_challenge_image(input_image_uri)
        image_uri = _extract_image_uri(result)
        if image_uri:
            self.latest_image_uri = image_uri
            self.last_image_tool = original_tool_name

    def summarize_result(self, original_tool_name: str, result: dict[str, Any]) -> str:
        payload = _sanitize_payload(result)
        input_field_name, _ = _find_input_image_field(result)
        output_field_name, output_image_uri = _find_output_image_field(result)
        if input_field_name and isinstance(payload, dict):
            payload["input_image_available"] = bool(self.challenge_image_uri)
            payload[input_field_name] = "<managed by runtime>"
        if output_field_name and isinstance(payload, dict):
            payload[output_field_name] = "<stored_by_runtime>" if output_image_uri else ""
            payload["image_available"] = bool(output_image_uri)
            payload["note"] = (
                "The runtime stored the produced image for later submission. The full data URI "
                "is intentionally omitted from the tool response."
            )
        try:
            return _truncate_text(json.dumps(payload, ensure_ascii=True, default=str))
        except TypeError:
            return _truncate_text(str(payload))


class ArenaMcpTool(BaseTool):
    _original_tool_name: str = PrivateAttr()
    _state: ArenaToolState = PrivateAttr()
    _runtime_hints: ToolRuntimeHints = PrivateAttr()

    def __init__(
        self,
        *,
        name: str,
        description: str,
        args_schema: type[BaseModel],
        original_tool_name: str,
        state: ArenaToolState,
        runtime_hints: ToolRuntimeHints,
    ) -> None:
        super().__init__(name=name, description=description, args_schema=args_schema)
        self._original_tool_name = original_tool_name
        self._state = state
        self._runtime_hints = runtime_hints

    def _prepare_arguments(self, raw_kwargs: dict[str, Any]) -> dict[str, Any]:
        kwargs = dict(self._state.tool_argument_overrides.get(self._original_tool_name, {}))
        kwargs.update(raw_kwargs)
        if self._runtime_hints.accepts_agent_id:
            kwargs["agent_id"] = self._state.agent_id
        image_input_field = self._runtime_hints.image_input_field
        if image_input_field and not kwargs.get(image_input_field):
            image_uri = self._state.current_image_uri()
            if image_uri:
                kwargs[image_input_field] = image_uri
        return kwargs

    async def _call_tool(self, **kwargs: Any) -> str:
        payload = self._prepare_arguments(kwargs)
        try:
            async with McpArenaClient(self._state.mcp_url, self._state.api_key) as client:
                result = await client.call_tool(self._original_tool_name, payload)
        except Exception as exc:
            error_payload = {
                "error": f"Tool {self._original_tool_name} failed: {exc}",
                "tool": self._original_tool_name,
            }
            return json.dumps(error_payload, ensure_ascii=True)

        self._state.record_result(self._original_tool_name, result)
        return self._state.summarize_result(self._original_tool_name, result)

    def _run(self, **kwargs: Any) -> str:
        return asyncio.run(self._call_tool(**kwargs))

    async def _arun(self, **kwargs: Any) -> str:
        return await self._call_tool(**kwargs)


def _build_tool_description(
    original_name: str,
    description: str,
    runtime_hints: ToolRuntimeHints,
) -> str:
    parts: list[str] = []
    cleaned_description = description.strip()
    if cleaned_description:
        parts.append(cleaned_description)
    parts.append(f"Original arena MCP tool name: `{original_name}`.")
    if runtime_hints.accepts_agent_id:
        parts.append("The runtime injects `agent_id` automatically.")
    if runtime_hints.accepts_proxy_api_key:
        parts.append("The runtime injects `proxy_api_key` automatically when supported.")
    if runtime_hints.image_input_field:
        parts.append(
            f"If `{runtime_hints.image_input_field}` is omitted, the runtime uses the active "
            "challenge image or latest generated image."
        )
        parts.append(
            "If this tool returns an image payload, the runtime stores it for later use and omits "
            "the full data URI from the tool response."
        )
    elif "image" in original_name.lower() or "image" in cleaned_description.lower():
        parts.append(
            "If this tool returns an image payload, the runtime stores it for later use and omits "
            "large image payloads from the tool response."
        )
    return " ".join(parts)


def build_crewai_tools(
    tool_defs: list[Any],
    *,
    agent_id: str,
    mcp_url: str,
    api_key: str | None,
    challenge_image_uri: str | None = None,
    tool_argument_overrides: dict[str, dict[str, Any]] | None = None,
    exclude_tools: set[str] | None = None,
) -> tuple[list[BaseTool], ArenaToolState]:
    state = ArenaToolState(
        agent_id=agent_id,
        mcp_url=mcp_url,
        api_key=api_key,
        challenge_image_uri=challenge_image_uri or "",
        tool_argument_overrides=tool_argument_overrides,
    )
    tools: list[BaseTool] = []

    for tool_spec in discover_tool_specs(tool_defs, exclude_tools=exclude_tools):
        args_schema = _build_args_schema(
            tool_spec.sanitized_name,
            tool_spec.input_schema,
            tool_spec.runtime_hints,
        )
        description = _build_tool_description(
            tool_spec.original_name,
            tool_spec.description,
            tool_spec.runtime_hints,
        )
        state.tool_name_map[tool_spec.sanitized_name] = tool_spec.original_name
        tools.append(
            ArenaMcpTool(
                name=tool_spec.sanitized_name,
                description=description,
                args_schema=args_schema,
                original_tool_name=tool_spec.original_name,
                state=state,
                runtime_hints=tool_spec.runtime_hints,
            )
        )

    return tools, state
