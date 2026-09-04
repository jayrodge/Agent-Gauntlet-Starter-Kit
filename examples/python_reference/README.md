# Python Reference Example (Advanced / Optional)

Production-style Agent Gauntlet baseline with explicit orchestration in pure Python.

**Not the starting point.** Use [`python_simple`](../python_simple/README.md) first.
Come back here only when you want retries, streaming, and a heavier extraction
loop as an advanced reference — not as a drop-in competitive baseline.

## Prerequisites

- Python 3.11–3.13 (`>=3.11,<3.14`)
- Base setup from repository root:
  - `pip install -r requirements.txt`
  - `cp .env.example .env` and configure values
  - Prefer having `python_simple` working before exploring this example

## Install Dependencies

From this directory (`examples/python_reference`):

```bash
pip install -r requirements.txt
```

This example only requires:

- `mcp`

## Run

From this directory (`examples/python_reference`):

```bash
python agent.py
```

The script loads `.env` from the repository root automatically.

## How It Works

This is a large, explicit orchestration reference (~2.5k lines) with retries,
streaming, and an LLM extraction judge. It is useful when you want to study
advanced control flow — not as the recommended first agent to customize.

## Key Files

- `agent.py`: full reference implementation and orchestration loop
- `requirements.txt`: minimal dependency list (`mcp`)

## Customization

Edit [`../../my_strategy.py`](../../my_strategy.py) and focus on:

- prompt shaping (`build_system_prompt`, `build_solver_prompt`)
- model policy (`rank_models`, `pick_model`)
- timeout and submission behavior (`should_submit_early`, `on_time_warning`)
- image-specific planning (`plan_image_tool`, `build_image_prompt`)

Note: `plan_tools` / `on_tool_result` are not invoked by this runtime today.

## When to Use This Example

- You already understand the simple agent flow
- You want to study retries, streaming, or answer-extraction patterns
- You plan to steal selective pieces into your own agent — not run this as-is

## Further Reading

- [Examples Overview](../README.md)
- [Getting Started](../../docs/getting-started.md)
- [Discovering Tools](../../docs/discovering-tools.md)
- [Interacting with Tools](../../docs/interacting-with-tools.md)
