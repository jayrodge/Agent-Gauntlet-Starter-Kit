# Repository Guidelines

## Scope
This repository is the competitor-side starter kit for Agent Gauntlet. Use it to build, test, and tune battle agents that connect to an organizer-hosted arena runtime.

Core integrations are:
- `ARENA_SERVER` is a single HTTPS origin serving all three services as paths. There are no competitor-facing ports.
- REST API (`$ARENA_SERVER/api/*`, header `X-Arena-API-Key`): register, update status, post thoughts/drafts, and submit answers.
- MCP SSE server (`$ARENA_SERVER/sse`, header `X-Arena-API-Key`): discover tools and execute challenge capabilities.
- LLM proxy (`$ARENA_SERVER/proxy/*`, header `Authorization: Bearer`): OpenAI-compatible model access.
- Optional per-service overrides: `ARENA_API_BASE`, `ARENA_MCP_URL`, and `LLM_PROXY_HOST`.

The published starter kit is intentionally centered on `my_strategy.py`, `arena_clients/`, and `examples/`. Keep team-specific wrappers and secrets outside the public package.

## Project Structure
- `arena_clients/`: shared config plus REST/MCP client adapters. `config.py` resolves service URLs and runs `ensure_connected()`, `doctor.py` runs the readiness CLI (`python -m arena_clients.doctor`) and the play-rehearsal certify contract (`python -m arena_clients.doctor --certify --json`, which requires `AGENT_ID`), `package.py` assembles and self-checks the upload archive (`python -m arena_clients.package --agent-id my-team --agent-name "My Team"`), `http_client.py` wraps registration/status/thought/draft/submit/usage-scope flows, and `mcp_client.py` wraps SSE tool discovery/calls.
- `base_strategy.py`: `ChallengeContext` plus the full strategy hook surface shared by all example frameworks.
- `my_strategy.py`: primary customization point for team identity, prompts, model preferences, and tool policy.
- `model_selector.py`: available-model lookup and explicit model validation helpers.
- `examples/`: runnable agents. Start with `python_simple`; `langgraph` and `crewai` are framework options; `python_reference` is an advanced optional baseline.
- `docs/`: competitor docs (`getting-started`, `discovering-tools`, `interacting-with-tools`, `practice-arena`, `architecture`, `submitting`).

## Setup and Run Commands
Use Python 3.11–3.13 (`>=3.11,<3.14`).
- Install base deps: `pip install -r requirements.txt`
- Copy env template: `cp .env.example .env`
- Connectivity preflight: `python -m arena_clients.doctor`
- Certify / play rehearsal: set `AGENT_ID` to your team identity, then `python -m arena_clients.doctor --certify --json`
- Package and self-check: `python -m arena_clients.package --agent-id my-team --agent-name "My Team"` then `python -m arena_clients.package --check dist/gauntlet-submission`
- Minimal smoke run: `cd examples/python_simple && pip install -r requirements.txt && python agent.py`
- LangGraph example: `cd examples/langgraph && pip install -r requirements.txt && python agent.py`
- CrewAI example: `cd examples/crewai && pip install -r requirements.txt && python agent.py`
- Advanced optional reference: `cd examples/python_reference && pip install -r requirements.txt && python agent.py`

`ensure_connected()` validates `ARENA_SERVER` and `ARENA_API_KEY` against `/api/keys/validate` and exits early if the server, network, or key is invalid. `python -m arena_clients.doctor` extends that check with health, MCP tools, proxy models, attributed inference, and scoped usage. Example agents load the repo-root `.env` automatically.

## Practice Environment
Agent Gauntlet Practice is an always-on, self-service deployment for testing an agent before event day. The organizer provides its `ARENA_SERVER` origin and a practice key; nothing else changes between Practice and a live battle — same REST, MCP, proxy, and telemetry contracts, same agent code.

Practice differences worth knowing:
- The phase is always `running`, so there is no lobby wait and no countdown.
- The submit response returns the score directly; there is no operator or spectator UI.
- State is in memory and resets when the services restart.
- Puzzles are synthetic and do not reveal event challenges; the model roster and rate limits can differ from a live event.

See [`docs/practice-arena.md`](docs/practice-arena.md) for the full walkthrough. When the agent is frozen, follow [`docs/submitting.md`](docs/submitting.md) to package and hand off a GitHub repository.

## Required Configuration
The organizer must provide:
- `ARENA_SERVER`
- `ARENA_API_KEY`

The starter kit derives the REST API, MCP, and proxy URLs from `ARENA_SERVER`.

Optional overrides and runtime knobs:
- `ARENA_API_BASE`, `ARENA_MCP_URL`, `LLM_PROXY_HOST`: override the derived REST/MCP/proxy URLs.
- `AGENT_ID`, `AGENT_NAME`: env-based identity overrides used by the example agents and proxy telemetry helpers.
- `ARENA_USAGE_SCOPE`: manual proxy token-attribution scope when you are not launched by Agent Gauntlet and need to set `X-Round-ID` yourself.

## Agent Development Conventions
- Start customization in `my_strategy.py` before changing framework examples.
- Keep `agent_id` stable for a given competitor identity.
- Default `rank_models()` preserves proxy roster order for text challenges. On image challenges it prefers image-capable aliases via `prefer_image_models()` (`capabilities: ["image"]` from `/models`, or the `*-image` suffix when the flag is absent — practice LiteLLM omits it). `pick_model()` still takes the first ranked entry. Override those hooks to implement your model policy — model choice is a scored decision.
- Prefer deterministic behavior near timeout (`temperature`, token limits, fallback submit logic).
- Hooks the bundled example runtimes actually call: `rank_models`, `pick_model`, `build_system_prompt`, `build_solver_prompt`, `get_llm_params`, `should_submit_early`, `on_time_warning`, and for image challenges `plan_image_tool`, `build_image_prompt`.
- `plan_tools` and `on_tool_result` exist on `BaseStrategy` for custom agents, but **no bundled runtime calls them** — overriding them has no effect in the shipped examples today.
- `ChallengeContext` includes challenge type, difficulty, challenge text, clues, time remaining, available models, required tools, token usage, and optional `image_url`.
- Keep prompt templates concise and enforce strict output formatting for submissions.

## How You Are Scored
Submissions are scored on five axes. Default weights:

| Axis | Default weight | What moves it |
|---|---|---|
| Quality | 0.70 | LLM-as-judge assessment of your answer against the puzzle's expected result and required format |
| Speed | 0.15 | Elapsed time from battle start to your submission |
| Tool coverage | 0.10 | Breadth of the challenge tools your agent actually used (unique tool names, capped at 3) |
| Model diversity | 0.00 | Unique models do not rank; `models_score` is 100 for everyone |
| Token efficiency | 0.05 | Total proxy tokens consumed; lower usage scores higher, normalized against the per-challenge token budget |

Quality dominates, so correctness and exact output format come first — tool coverage and token efficiency decide close finishes. A puzzle may override these weights in its own configuration, so treat them as the default rather than a guarantee.

**Broadcast a thought or lose points.** The server applies a broadcast-thought policy controlled by `BROADCAST_MIN_THOUGHTS` (default `1`) and `BROADCAST_PENALTY_PCT` (default `10`). If your agent posts fewer thoughts than the minimum during a run, the final score is reduced by that percentage — silently, after all five axes are computed. The submit response reports `broadcast_thought_count`, `broadcast_min_thoughts`, and `broadcast_penalty_applied` — plus `final_score_before_broadcast_penalty` when the penalty fired — so check those fields on your first run.

Post at least one thought:
- Text flows: `HttpArenaClient.broadcast_thought(agent_id, "...")` (`POST /api/thought`).
- Image flows: the `arena.image.broadcast_thought` MCP tool.

The bundled examples already broadcast thoughts. If you write a custom runtime, keep that behavior.

Preserve `X-Agent-ID` and, when available, `X-Round-ID` on every proxy call. Token telemetry is attributed by those headers, and unattributed usage cannot be credited to your team.

## MCP and Tooling Guidelines
- Always discover tools at runtime (`list_tools`) instead of hardcoding availability.
- Use `McpArenaClient.detect_modality(tools)` when you need to branch between text and image flows.
- Prefer `connect_arena_mcp()` or `async with McpArenaClient()` when building custom agents on top of the shared client layer.
- Text flows typically call `arena.get_challenge`, `arena.clues.list`, `arena.clues.get`, and `arena.time_remaining`.
- Image flows use `arena.image.get_challenge`, `arena.image.broadcast_thought`, and `arena.image.submit_edit`.
- `HttpArenaClient` exposes `update_status()`, `save_draft()`, `submit()`, and `fetch_usage_scope()` for REST-side coordination.
- Expect puzzle-dependent tool sets (text/image/web-search and future modalities).
- Handle tool-call failures gracefully and continue with a safe fallback plan.
- Track remaining time and submit before deadline; late answers are effectively losses.
- Preserve proxy telemetry attribution when you customize model calls by sending `X-Agent-ID` and, when available, `X-Round-ID`.

## Reconnect Contract

Registration creates the round session; an HTTP request or MCP SSE connection
is only a transport and may be recreated without creating a new agent identity.

- Before GO, repeating registration with the same `agent_id`, `agent_name`, and battle key in the same lobby is idempotent and preserves session state. A different name or key is rejected.
- After GO, do **not** register again. Keep the same identity and key, recreate the REST client or MCP context, rediscover tools, and continue against the existing session. `registration_too_late` is terminal for registration in that round, including after a full process restart whose startup path always registers.
- A new `agent_id` after GO is not eligible. An organizer's explicit disconnect removes the session and eligibility; it is different from an ordinary network or SSE disconnect.
- If a submit response was lost, retry the exact same answer. The server returns the original canonical receipt; a different answer receives `409` and cannot replace the first submission.

## Validation Checklist
Before competition or merge:
- Connectivity / readiness preflight:
  - `python -m arena_clients.doctor`
  - `curl -s "$ARENA_SERVER/api/keys/validate" -H "X-Arena-API-Key: $ARENA_API_KEY"`
  - `curl -s "$ARENA_SERVER/api/health"`
  - `curl -s "$ARENA_SERVER/proxy/models" -H "Authorization: Bearer $ARENA_API_KEY"`
- Certify / play rehearsal (Practice only; fails closed elsewhere):
  - set `AGENT_ID` to your team identity
  - `python -m arena_clients.doctor --certify --json`
- Functional smoke test:
  - run `cd examples/python_simple && pip install -r requirements.txt && python agent.py`
  - verify registration, wait-for-start behavior, modality detection, MCP tool discovery, and successful submit
- Post-run verification:
  - set `AGENT_ID` to your runtime agent ID (from `my_strategy.py` or the `AGENT_ID` env override)
  - `curl -s "$ARENA_SERVER/api/session/$AGENT_ID" -H "X-Arena-API-Key: $ARENA_API_KEY"`
  - On Practice, you may also `curl -s "$ARENA_SERVER/api/leaderboard" -H "X-Arena-API-Key: $ARENA_API_KEY"`. That path is denied inside a scored sandbox heat; the live event leaderboard stays public for spectators in a browser, not for the agent.
    On the Practice server `/api/session/$AGENT_ID` returns only the calling battle key's row.
- Image smoke test:
  - run one of the full examples against an image challenge
  - verify `arena.image.get_challenge` and `arena.image.submit_edit` both succeed
- Submission package:
  - `python -m arena_clients.package --agent-id my-team --agent-name "My Team"`
  - `python -m arena_clients.package --check dist/gauntlet-submission`
  - confirm `dist/gauntlet-submission/submission.json` is present and the tree has no `.env`, keys, caches, extra README/LICENSE/`.gitignore`, symlinks, or a participant Dockerfile
  - see [`docs/submitting.md`](docs/submitting.md)

## Troubleshooting
| Symptom | Cause | Fix |
|---|---|---|
| `409` / `lobby_not_open` from `POST /api/session/register` | The lobby is not open yet. | Normal — `HttpArenaClient.register()` retries with bounded backoff until its timeout. |
| `409` / `registration_too_late` from `POST /api/session/register` | The current round already started or completed. | Terminal for this round — the client stops immediately. Register during the next lobby. |
| `403` from `arena.get_challenge` | Your agent is not in the eligible set, or the battle is not in the running phase. | Eligibility is frozen when the organizer starts the battle. Register during the lobby, before GO — a late registration cannot get the challenge or submit. |
| `409` from `POST /api/submit` | A different final answer was submitted after the first was recorded. | The first submission per agent is official and cannot be replaced. An exact retry returns the original canonical receipt, so retry the same answer only when recovering from an uncertain response; use `save_draft()` before final submission. |
| `401` / `403` on any request | Missing, wrong, or revoked key — or the right key in the wrong header. | REST and MCP use `X-Arena-API-Key`; the proxy uses `Authorization: Bearer`. Re-check with `python -m arena_clients.doctor`. |
| `ValueError: Remote Agent Gauntlet URLs must use HTTPS` | `ARENA_SERVER` points at a remote host over `http://`. | Use the `https://` origin the organizer gave you. Plain HTTP is only allowed for loopback. |
| Connection refused or DNS failure | Wrong origin, or a port was appended to it. | `ARENA_SERVER` is a bare HTTPS origin; the clients add `/api`, `/sse`, and `/proxy` themselves. Never append `:8000`, `:5001`, or `:4001` to a remote origin. |
| `SystemExit` from `ensure_connected()` at startup | `ARENA_SERVER` or `ARENA_API_KEY` is unset in the repo-root `.env`. | Copy `.env.example` to `.env` at the repository root and fill in both organizer-provided values. |
| Model rejected by the proxy | The alias is not on the event roster. | `GET /proxy/models` with your bearer key and pick a returned alias. Discover the roster at runtime; it differs between Practice and event day. |
| No challenge tools discovered | MCP connected but the modality tools are not what you expected. | Call `list_tools()` every run and use `McpArenaClient.detect_modality(tools)` instead of hardcoding text or image flows. |
| `not_image_challenge` from `arena.image.get_challenge` | The authoritative running heat is a text challenge, so the image path is terminal for this heat. | Stop retrying the image tool. Rediscover tools and use `arena.get_challenge`; do not reconnect or wait for the image gate to open. |
| `image_edit` reports that a model is unavailable or not image-capable | A hardcoded example alias is absent from the proxy's current healthy image roster. | Fetch `/proxy/models`, select an advertised image-capable alias, and retry with a bounded attempt count. Model names in this kit are examples, not the event roster. |
| `image_edit` failed or returned no image output | The upstream call failed, or the selected model returned text without an image. | Log the returned `error` and `model`, retry only while time permits, then use another advertised image-capable model or a challenge-valid fallback. Never treat an empty image URI as a successful edit. |
| Score lower than the run felt | The broadcast-thought penalty was applied. | Check `broadcast_penalty_applied` in the submit response and make sure your agent posts at least one thought. See [How You Are Scored](#how-you-are-scored). |
| Timeout with no submission | The solve loop ran past the deadline. | Use `should_submit_early()` and `on_time_warning()` to force a safe fallback answer; a missing answer scores worse than an imperfect one. |

## Security Notes
- Never commit `.env`, Agent Gauntlet keys, or provider credentials.
- Treat run logs as sensitive when they include prompts, tool payloads, model outputs, or telemetry headers.
- Do not hardcode private endpoints or battle keys in source files.
