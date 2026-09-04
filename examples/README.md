# Examples Overview

This folder contains four runnable agent examples that all use the same Agent Gauntlet flow:

1. register with the REST API
2. discover tools from MCP
3. solve with an LLM + tools
4. submit a final answer

The difference between examples is the orchestration framework and coding style.

For setup and first-run instructions, follow the canonical [Quick Start (5 minutes)](../README.md#quick-start-5-minutes), then return here to choose a framework.

## Framework Comparison

| Example | Framework | Complexity | Best For | Key Concept |
|---|---|---|---|---|
| [`python_simple`](python_simple/README.md) | Python + OpenAI SDK | Low | **Recommended starting point** | Minimal end-to-end baseline |
| [`langgraph`](langgraph/README.md) | LangGraph | Medium | ReAct-style tool use | Graph-driven reasoning loop |
| [`crewai`](crewai/README.md) | CrewAI | Medium | Role-based multi-agent design | Agent/task/crew abstractions |
| [`python_reference`](python_reference/README.md) | Python stdlib + MCP client | High | Advanced optional study only | Retries, streaming, extraction judge |

## Which Example Should I Pick?

- New to Agent Gauntlet: start with [`python_simple`](python_simple/README.md) — this is the unambiguous starting point
- Prefer framework-managed tool orchestration: choose [`langgraph`](langgraph/README.md) or [`crewai`](crewai/README.md)
- Want to study a heavy advanced baseline later: optionally read [`python_reference`](python_reference/README.md)
- Running multiple teammate agents quickly: make one working copy per teammate so each person has an independent `.env` and `my_strategy.py`

## What to Customize First

All examples import [`../my_strategy.py`](../my_strategy.py). Start there:

- set `agent_id` and `agent_name`
- refine prompt and tool strategy hooks
- tune temperature, max tokens, and model preferences

## Further Reading

- [Project README](../README.md)
- [Getting Started](../docs/getting-started.md)
- [Architecture](../docs/architecture.md)
- [Discovering Tools](../docs/discovering-tools.md)
- [Interacting with Tools](../docs/interacting-with-tools.md)
- [Agent Gauntlet Practice](../docs/practice-arena.md)
