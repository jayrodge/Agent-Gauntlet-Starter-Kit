# CLAUDE.md

Guidance for building your Agent Gauntlet competitor agent with Claude Code,
Cursor, or any other coding agent.

## Read AGENTS.md first

[`AGENTS.md`](AGENTS.md) is the single source of truth for this kit: endpoints
and auth headers, setup commands, the strategy hook surface, how you are scored,
and a troubleshooting table. This file deliberately does not repeat any of it —
if the two ever disagree, `AGENTS.md` wins.

Claude Code reads this file automatically. Cursor and other agents may not, so
point them at the reference explicitly:

```text
Read AGENTS.md in the repository root before making changes.
```

## Edit one file

`my_strategy.py` is the file you customize. It subclasses `BaseStrategy` from
`base_strategy.py`, and every bundled example imports it, so a change there
applies to all of them.

Tell your agent to keep edits inside `my_strategy.py` unless you have a specific
reason to fork an example. Changing `arena_clients/` risks breaking the
registration, MCP, and telemetry contracts the organizer's server expects.

## One hook trap worth knowing

`plan_tools` and `on_tool_result` exist on `BaseStrategy`, but **no bundled
runtime calls them**. An agent that "implements tool planning" by overriding
those hooks will produce code that runs, passes review, and changes nothing.

The hooks the shipped examples actually invoke are listed in `AGENTS.md` under
Agent Development Conventions. Ask your coding agent to check that list before
it proposes a hook-based change.

## Before you claim it works

Run the readiness preflight and one real agent run, and read the submit response
rather than trusting the model's summary of it:

```bash
python -m arena_clients.doctor
cd examples/python_simple && python agent.py
```

`AGENTS.md` has the full validation checklist and the meaning of each failure.
