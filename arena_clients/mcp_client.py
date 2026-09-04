"""MCP client for Agent Gauntlet challenge tools.

This client connects to the Agent Gauntlet Platform MCP server (HTTP/SSE) to access
challenge tools like `arena.get_challenge`, `arena.clues.list`, and
`arena.time_remaining`, plus image challenge tools.

The MCP URL is derived from `ARENA_SERVER` by default, and the competitor key is
sent as the `X-Arena-API-Key` header.

Example:
    async with McpArenaClient() as client:  # or McpArenaClient("https://arena.example.com")
        challenge = await client.get_challenge("my-agent")
        clues = await client.list_clues("my-agent")
        clue = await client.get_clue("clue_0", "my-agent")
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator
from urllib.error import URLError
from urllib.request import Request, urlopen

from mcp import ClientSession
from mcp.client.sse import sse_client

from .config import get_api_base, get_arena_api_key, get_mcp_url


@dataclass
class ChallengeInfo:
    """Challenge information from get_challenge."""

    challenge_type: str
    challenge_id: str
    puzzle_id: str
    description: str
    rules: str
    max_time_s: int
    clues: list[str]
    time_remaining_s: float
    raw_data: dict[str, Any]

    def __getattr__(self, name: str) -> Any:
        """Provide dynamic access to challenge fields returned by MCP.

        This keeps agents future-proof as new challenge keys are introduced
        (for example: video_url, output_format, required_tools).
        """
        if name in self.raw_data:
            return self.raw_data[name]
        raise AttributeError(name)

    @property
    def extra_text(self) -> str:
        """Flatten non-core challenge fields into searchable text."""
        core_keys = {
            "challenge_type",
            "challenge_id",
            "puzzle_id",
            "description",
            "rules",
            "max_time_s",
            "clues",
            "time_remaining_s",
        }
        parts: list[str] = []
        for key, value in self.raw_data.items():
            if key in core_keys:
                continue
            text = _stringify_challenge_value(value)
            if text:
                parts.append(f"{key}: {text}")
        return " ".join(parts).strip()


@dataclass
class ClueInfo:
    """Information about a specific clue."""

    clue_id: str
    text: str
    time_remaining_s: float


@dataclass
class ImageChallengeInfo:
    """Image challenge information from arena.image.get_challenge."""

    challenge_type: str
    challenge_id: str
    puzzle_id: str
    difficulty: str
    description: str
    prompt: str
    reference_notes: str
    max_time_s: int
    input_image_uri: str
    time_remaining_s: float


def tool_supports_input_property(
    tool_defs: list[Any],
    tool_name: str,
    property_name: str,
) -> bool:
    """Return True when a discovered MCP tool schema advertises an input property."""
    for tool_def in tool_defs:
        if str(getattr(tool_def, "name", "") or "").strip() != tool_name:
            continue
        input_schema = getattr(tool_def, "inputSchema", None)
        if not isinstance(input_schema, dict):
            return False
        properties = input_schema.get("properties")
        return isinstance(properties, dict) and property_name in properties
    return False


def resolve_tool_proxy_api_key(
    tool_defs: list[Any],
    tool_name: str,
    *,
    llm_api_key: str,
    arena_api_key: str,
) -> str | None:
    """Return a BYO proxy key only when both env and server schema allow it."""
    normalized_llm_key = str(llm_api_key or "").strip()
    normalized_arena_key = str(arena_api_key or "").strip()
    if not normalized_llm_key or normalized_llm_key == normalized_arena_key:
        return None
    if not tool_supports_input_property(tool_defs, tool_name, "proxy_api_key"):
        return None
    return normalized_llm_key


def build_image_tool_arguments(
    tool_defs: list[Any],
    tool_name: str,
    *,
    selected_model: str,
    llm_api_key: str,
    arena_api_key: str,
) -> dict[str, str]:
    """Build optional image-tool arguments advertised by the live MCP schema."""
    arguments: dict[str, str] = {}
    normalized_model = str(selected_model or "").strip()
    if normalized_model and tool_supports_input_property(
        tool_defs,
        tool_name,
        "model",
    ):
        arguments["model"] = normalized_model
    proxy_api_key = resolve_tool_proxy_api_key(
        tool_defs,
        tool_name,
        llm_api_key=llm_api_key,
        arena_api_key=arena_api_key,
    )
    if proxy_api_key:
        arguments["proxy_api_key"] = proxy_api_key
    return arguments


class McpArenaClient:
    """MCP client for Agent Gauntlet challenge tools.

    This client connects to the Agent Gauntlet Platform MCP server via SSE transport
    and provides methods to access challenge tools.

    Usage:
        async with McpArenaClient() as client:  # or McpArenaClient("https://arena.example.com")
            challenge = await client.get_challenge("my-agent")
    """

    def __init__(
        self,
        mcp_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        """Initialize the client.

        Args:
            mcp_url: Base URL for the Agent Gauntlet Platform MCP server; `/sse` is
                appended (default: ARENA_MCP_URL or derived from ARENA_SERVER)
            api_key: Competitor key sent as the `X-Arena-API-Key` header
                (default: ARENA_API_KEY env var)
            timeout: Connection timeout in seconds
        """
        base_url = get_mcp_url(mcp_url)
        # SSE endpoint is at /sse
        self.sse_url = f"{base_url.rstrip('/')}/sse"
        resolved_api_key = get_arena_api_key(api_key)
        self._headers = (
            {"X-Arena-API-Key": resolved_api_key}
            if resolved_api_key
            else {}
        )
        self.timeout = timeout
        self._session: ClientSession | None = None
        self._context = None

    async def __aenter__(self) -> "McpArenaClient":
        """Enter async context and establish connection."""
        self._context = sse_client(self.sse_url, headers=self._headers)
        read, write = await self._context.__aenter__()
        self._session = ClientSession(read, write)
        await self._session.__aenter__()
        await self._session.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context and close connection."""
        if self._session:
            await self._session.__aexit__(exc_type, exc_val, exc_tb)
        if self._context:
            await self._context.__aexit__(exc_type, exc_val, exc_tb)

    def _parse_result(self, result) -> dict[str, Any]:
        """Parse a tool result to a dictionary."""
        if isinstance(getattr(result, "structuredContent", None), dict):
            return dict(result.structuredContent)
        if getattr(result, "structuredContent", None) is not None:
            return {"structured": result.structuredContent}
        if not result.content:
            return {}
        text = result.content[0].text
        if not text:
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"raw": text}

    @staticmethod
    def _normalize_clues(value: Any) -> list[str]:
        if not value:
            return []
        if isinstance(value, str):
            text = value.strip()
            return [text] if text else []
        if not isinstance(value, list):
            text = str(value).strip()
            return [text] if text else []

        normalized: list[str] = []
        for item in value:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    normalized.append(text)
                continue
            if isinstance(item, dict):
                clue_text = item.get("text")
                if isinstance(clue_text, str) and clue_text.strip():
                    normalized.append(clue_text.strip())
                    continue
            text = str(item).strip()
            if text:
                normalized.append(text)
        return normalized

    async def list_tools(self) -> list[str]:
        """List available tools.

        Returns:
            List of tool names
        """
        if not self._session:
            raise RuntimeError("Client not connected. Use 'async with' context.")

        result = await self._session.list_tools()
        return [tool.name for tool in result.tools]

    async def list_tool_defs(self) -> list[Any]:
        """List full available tool definitions (with schemas)."""
        if not self._session:
            raise RuntimeError("Client not connected. Use 'async with' context.")

        result = await self._session.list_tools()
        return list(result.tools)

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Call any MCP tool by name and return parsed JSON."""
        if not self._session:
            raise RuntimeError("Client not connected. Use 'async with' context.")

        payload = arguments or {}
        result = await self._session.call_tool(name, payload)
        return self._parse_result(result)

    @staticmethod
    def detect_modality(tools: list[str]) -> str:
        """Detect active challenge modality.

        Detection order:
        1) Competition API challenge_type (if reachable)
        2) Tool-set fallback heuristics
        """
        api_base = get_api_base()
        if api_base:
            request = Request(
                f"{api_base.rstrip('/')}/api/competition",
                headers={"Accept": "application/json"},
                method="GET",
            )
            try:
                with urlopen(request, timeout=1.5) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                challenge_type = str(payload.get("challenge_type") or "").lower()
                if "image" in challenge_type:
                    return "image"
                if challenge_type:
                    return "text"
            except (json.JSONDecodeError, URLError, TimeoutError):
                pass

        tool_set = set(tools)
        has_text = "arena.get_challenge" in tool_set
        has_image = "arena.image.get_challenge" in tool_set
        if has_image and not has_text:
            return "image"
        if has_text:
            return "text"
        return "text"

    async def get_challenge(self, agent_id: str = "default") -> ChallengeInfo:
        """Get the current challenge and start the timer.

        Args:
            agent_id: Unique identifier for the agent

        Returns:
            ChallengeInfo with puzzle details
        """
        if not self._session:
            raise RuntimeError("Client not connected. Use 'async with' context.")

        result = await self._session.call_tool(
            "arena.get_challenge",
            {"agent_id": agent_id},
        )
        data = self._parse_result(result)

        if "error" in data:
            raise McpArenaError(data["error"])

        return ChallengeInfo(
            challenge_type=data.get("challenge_type", ""),
            challenge_id=data.get("challenge_id", ""),
            puzzle_id=data.get("puzzle_id", ""),
            description=data.get("description", ""),
            rules=data.get("rules", ""),
            max_time_s=data.get("max_time_s", 0),
            clues=self._normalize_clues(data.get("clues")),
            time_remaining_s=data.get("time_remaining_s", 0),
            raw_data=dict(data),
        )

    async def get_image_challenge(self, agent_id: str = "default") -> ImageChallengeInfo:
        """Get the current image challenge and start its timer."""
        if not self._session:
            raise RuntimeError("Client not connected. Use 'async with' context.")

        result = await self._session.call_tool(
            "arena.image.get_challenge",
            {"agent_id": agent_id},
        )
        data = self._parse_result(result)
        if "error" in data:
            raise McpArenaError(data["error"], code=data.get("code"))

        prompt = data.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            prompt = data.get("edit_prompt", "")
        if not isinstance(prompt, str):
            prompt = ""

        return ImageChallengeInfo(
            challenge_type=data.get("challenge_type", ""),
            challenge_id=data.get("challenge_id", ""),
            puzzle_id=data.get("puzzle_id", ""),
            difficulty=data.get("difficulty", ""),
            description=data.get("description", ""),
            prompt=prompt,
            reference_notes=data.get("reference_notes", ""),
            max_time_s=data.get("max_time_s", 0),
            input_image_uri=data.get("input_image_uri", ""),
            time_remaining_s=data.get("time_remaining_s", 0),
        )

    async def list_clues(self, agent_id: str = "default") -> list[str]:
        """List available clue IDs.

        Args:
            agent_id: Unique identifier for the agent

        Returns:
            List of clue IDs
        """
        if not self._session:
            raise RuntimeError("Client not connected. Use 'async with' context.")

        result = await self._session.call_tool(
            "arena.clues.list",
            {"agent_id": agent_id},
        )
        data = self._parse_result(result)

        if "error" in data:
            raise McpArenaError(data["error"])

        return data.get("clue_ids", [])

    async def get_clue(self, clue_id: str, agent_id: str = "default") -> ClueInfo:
        """Get a specific clue by ID.

        Args:
            clue_id: The clue ID (e.g., "clue_0")
            agent_id: Unique identifier for the agent

        Returns:
            ClueInfo with clue text
        """
        if not self._session:
            raise RuntimeError("Client not connected. Use 'async with' context.")

        result = await self._session.call_tool(
            "arena.clues.get",
            {"clue_id": clue_id, "agent_id": agent_id},
        )
        data = self._parse_result(result)

        if "error" in data:
            raise McpArenaError(data["error"])

        return ClueInfo(
            clue_id=data.get("clue_id", clue_id),
            text=data.get("text", ""),
            time_remaining_s=data.get("time_remaining_s", 0),
        )

    async def get_all_clue_texts(self, agent_id: str = "default") -> list[str]:
        """Fetch every listed clue in server order."""
        clue_ids = await self.list_clues(agent_id)
        clues = [await self.get_clue(clue_id, agent_id) for clue_id in clue_ids]
        return [clue.text for clue in clues if clue.text.strip()]

    async def time_remaining(self, agent_id: str = "default") -> dict[str, Any]:
        """Get remaining time for the current match.

        Args:
            agent_id: Unique identifier for the agent

        Returns:
            Dictionary with time_remaining_s, elapsed_s, max_time_s, expired
        """
        if not self._session:
            raise RuntimeError("Client not connected. Use 'async with' context.")

        result = await self._session.call_tool(
            "arena.time_remaining",
            {"agent_id": agent_id},
        )
        data = self._parse_result(result)

        if "error" in data:
            raise McpArenaError(data["error"])

        return data

    async def broadcast_image_thought(
        self,
        thought: str,
        agent_id: str = "default",
    ) -> dict[str, Any]:
        """Broadcast thought text through the image challenge channel."""
        if not self._session:
            raise RuntimeError("Client not connected. Use 'async with' context.")

        result = await self._session.call_tool(
            "arena.image.broadcast_thought",
            {"thought": thought, "agent_id": agent_id},
        )
        data = self._parse_result(result)
        raw_message = data.get("raw")
        if isinstance(raw_message, str):
            lowered = raw_message.strip().lower()
            if lowered.startswith("error executing tool"):
                raise McpArenaError(raw_message)
        if "error" in data:
            raise McpArenaError(data["error"])
        return data

    async def submit_image(
        self,
        agent_id: str,
        image_uri: str,
        client_metrics: dict[str, Any] | None = None,
        rationale: str = "",
    ) -> dict[str, Any]:
        """Submit an image output via arena.image.submit_edit."""
        if not self._session:
            raise RuntimeError("Client not connected. Use 'async with' context.")

        result = await self._session.call_tool(
            "arena.image.submit_edit",
            {
                "edited_image": image_uri,
                "client_metrics": client_metrics or {},
                "rationale": rationale,
                "agent_id": agent_id,
            },
        )
        data = self._parse_result(result)
        if "error" in data:
            raise McpArenaError(data["error"])
        return data


class McpArenaError(Exception):
    """Error from the Agent Gauntlet Platform MCP server."""

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        raw_code = str(code or "").strip()
        self.code = raw_code or None
        self.terminal = self.code == "not_image_challenge"

    @property
    def is_terminal(self) -> bool:
        """True when the agent should stop retrying this challenge tool."""
        return self.code == "not_image_challenge"


def _stringify_challenge_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        parts = [_stringify_challenge_value(item) for item in value]
        return " ".join(part for part in parts if part).strip()
    if isinstance(value, dict):
        if isinstance(value.get("text"), str):
            text = value.get("text", "").strip()
            if text:
                return text
        try:
            return json.dumps(value, ensure_ascii=True, sort_keys=True)
        except Exception:
            return str(value)
    return str(value)


@asynccontextmanager
async def connect_arena_mcp(
    mcp_url: str | None = None,
) -> AsyncIterator[McpArenaClient]:
    """Convenience context manager for connecting to Agent Gauntlet Platform MCP.

    Example:
        async with connect_arena_mcp() as client:  # or connect_arena_mcp("https://arena.example.com")
            challenge = await client.get_challenge("my-agent")
    """
    client = McpArenaClient(mcp_url)
    async with client:
        yield client
