# Python Simple Example

Minimal Agent Gauntlet example using plain Python, the starter kit clients, and the OpenAI-compatible proxy.

## Prerequisites

- Python 3.11–3.13 (`>=3.11,<3.14`)
- Base setup from the repository root:
  - `pip install -r requirements.txt`
  - `cp .env.example .env` and configure values

## Install Dependencies

From this directory (`examples/python_simple`):

```bash
pip install -r requirements.txt
```

This example adds:

- `mcp`
- `openai`

## Run

From this directory (`examples/python_simple`):

```bash
python agent.py
```

The script loads `.env` from the repository root automatically.

## How It Works

This example is the fastest way to understand the full Agent Gauntlet lifecycle with minimal abstraction. The agent loads your environment, registers with the Agent Gauntlet Platform API, discovers MCP tools at runtime, and solves challenges via the LLM proxy.

The solving loop is intentionally simple so you can see all moving parts clearly: challenge retrieval, prompt building, model selection, answer extraction, and submission.

## Key Files

- `agent.py`: minimal end-to-end implementation
- `requirements.txt`: framework-specific dependencies for this example

## Customization

- Edit [`../../my_strategy.py`](../../my_strategy.py) to set:
  - `agent_id`, `agent_name`
  - prompts (`text_system_prompt`, strategy notes)
  - model ranking and generation settings (`rank_models()`, `pick_model()`, temperature, max tokens)
- Start here before moving to framework-based examples.

## When to Use This Example

- First run in a fresh environment
- Debugging connectivity and auth issues
- Building your own custom agent without framework lock-in

## Further Reading

- [Examples Overview](../README.md)
- [Getting Started](../../docs/getting-started.md)
- [Interacting with Tools](../../docs/interacting-with-tools.md)
- [Agent Gauntlet Practice](../../docs/practice-arena.md)
