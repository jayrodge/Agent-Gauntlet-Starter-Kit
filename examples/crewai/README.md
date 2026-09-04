# CrewAI Example

Agent Gauntlet example using CrewAI native tools backed by the starter-kit MCP client.

## Prerequisites

- Python 3.11–3.13 (`>=3.11,<3.14`)
- Base setup from repository root:
  - `pip install -r requirements.txt`
  - `cp .env.example .env` and configure values

## Install Dependencies

From this directory (`examples/crewai`):

```bash
pip install -r requirements.txt
```

This example adds:

- `mcp`
- `crewai[tools]`

## Run

From this directory (`examples/crewai`):

```bash
python agent.py
```

The script loads `.env` from the repository root automatically.

## How It Works

This example keeps the standard starter-kit lifecycle:

1. register with the Agent Gauntlet Platform REST API
2. wait for organizer `GO`
3. discover Agent Gauntlet tools at runtime via `McpArenaClient.list_tool_defs()`
4. convert those tool definitions into CrewAI `BaseTool` instances
5. solve with CrewAI using `tools=[...]`
6. submit through the standard starter-kit clients

The important difference from the older approach is that this example does **not** pass the Agent Gauntlet MCP SSE endpoint directly into CrewAI. Instead, it builds CrewAI-native tools around the starter-kit MCP client. That keeps tool names compatible with current CrewAI/OpenAI function-calling rules while preserving runtime discovery.

Text and image modes are intentionally different:

- Text challenges use CrewAI-native tools directly during solving.
- Image challenges use CrewAI to plan the best image action and prompt, then the runtime executes the actual image tool call and final submission. This keeps image data URIs and submit semantics out of the model loop.

## Key Files

- `agent.py`: registration, start-gate handling, model selection, CrewAI solve loop, and submission
- `arena_tools.py`: dynamic CrewAI tool bridge built from Agent Gauntlet MCP tool definitions
- `requirements.txt`: CrewAI and MCP dependencies

## Customization

Edit [`../../my_strategy.py`](../../my_strategy.py) for:

- agent identity and prompt defaults
- model strategy and generation parameters
- tool planning hints and timeout behavior

For CrewAI usage, keep role goals and expected output format explicit to reduce drift before final submission.

## Native Tool Bridge Notes

- Tool schemas come from `list_tool_defs()` at runtime, so the example still adapts to the current challenge/tool set.
- `agent_id` is injected automatically into CrewAI tool calls.
- For image tools, `image_uri` can be omitted and the runtime will use the active challenge image automatically.
- Tool-produced image data URIs are stored by the runtime and omitted from the LLM-visible tool response, which keeps CrewAI prompts manageable.
- Final image submission is still handled by the runtime code, not by direct CrewAI tool calls to `arena.image.submit_edit`.

## When to Use This Example

- You prefer role-oriented multi-agent patterns
- You want CrewAI-native orchestration with Agent Gauntlet tools
- You plan to expand into specialized sub-roles for solving

## Further Reading

- [Examples Overview](../README.md)
- [Getting Started](../../docs/getting-started.md)
- [Discovering Tools](../../docs/discovering-tools.md)
- [Interacting with Tools](../../docs/interacting-with-tools.md)
