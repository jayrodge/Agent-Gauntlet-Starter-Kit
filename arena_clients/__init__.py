"""Self-contained Agent Gauntlet clients for starter kit agents."""

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
    SessionInfo,
    SubmitResult,
)
from .mcp_client import (
    build_image_tool_arguments,
    ChallengeInfo,
    ClueInfo,
    ImageChallengeInfo,
    McpArenaClient,
    McpArenaError,
    connect_arena_mcp,
    resolve_tool_proxy_api_key,
    tool_supports_input_property,
)
from .session_monitor import (
    DEFAULT_SESSION_POLL_INTERVAL_S,
    TERMINAL_SESSION_STATUSES,
    SessionMonitor,
    SessionStopReason,
    get_session_stop_reason,
    monitor_session,
    resolve_session_poll_interval,
)

__all__ = [
    "ArenaAPIError",
    "ArenaConnectionError",
    "build_image_tool_arguments",
    "ChallengeInfo",
    "ClueInfo",
    "DEFAULT_SESSION_POLL_INTERVAL_S",
    "ensure_connected",
    "get_session_stop_reason",
    "get_api_base",
    "get_arena_api_key",
    "get_mcp_url",
    "get_proxy_host",
    "ImageChallengeInfo",
    "HttpArenaClient",
    "McpArenaClient",
    "McpArenaError",
    "SessionInfo",
    "SessionMonitor",
    "SessionStopReason",
    "SubmitResult",
    "TERMINAL_SESSION_STATUSES",
    "connect_arena_mcp",
    "monitor_session",
    "resolve_tool_proxy_api_key",
    "resolve_session_poll_interval",
    "tool_supports_input_property",
]
