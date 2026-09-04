# Submitting your agent

The organizer admits a GitHub repository that contains only the packager
tarball and its checksum. This kit assembles the submission tree, writes those
two files, and runs the same local checks the organizer runs before it builds
an image.

Develop against Practice first. When the agent is frozen, run the
connectivity preflight, rehearse a full play with `--certify`, package,
self-check, then publish a new public GitHub repository with the two packager
files. Send the organizer that repository URL.

```text
develop  →  python -m arena_clients.doctor
         →  python -m arena_clients.doctor --certify --json   # requires AGENT_ID
         →  python -m arena_clients.package --agent-id my-team --agent-name "My Team"
         →  python -m arena_clients.package --check dist/gauntlet-submission
         →  publish dist/<agent-id>-submission.tar.gz and dist/<agent-id>-submission.tar.gz.sha256
```

## The repository you owe

A **new public GitHub repository** whose root contains **only** these two
files, named exactly as the packager wrote them:

```text
<agent-id>-submission.tar.gz
<agent-id>-submission.tar.gz.sha256
```

Example for agent id `team-nemo`:

```text
team-nemo-submission.tar.gz
team-nemo-submission.tar.gz.sha256
```

This is not a fork of the starter kit. Do not push the unpacked
`dist/gauntlet-submission/` tree as the repository root. Do not send a lone
`my_strategy.py`. Do not use GitHub's auto-generated source zip. Organizers
download only the named tarball and checksum pair.

Python Simple, Python Reference, LangGraph, and CrewAI all use the same
two-file GitHub tree. `--entrypoint` (and `--include` / `--lockfile` when
needed) only changes what is **inside** the tarball.

LangGraph example:

```bash
python -m arena_clients.package --agent-id team-nemo --agent-name "Team Nemo" \
  --entrypoint examples/langgraph/agent.py
```

GitHub still has `team-nemo-submission.tar.gz` and
`team-nemo-submission.tar.gz.sha256`. The archive unpacks to a
`gauntlet-submission/` directory with `submission.json` at its root.

## Inside the tarball

When the organizer extracts the archive, the submission tree looks like this:

```text
gauntlet-submission/
├── submission.json
├── requirements.lock
├── arena_clients/
├── base_strategy.py
├── model_selector.py
├── my_strategy.py
└── examples/python_simple/agent.py   # or your declared entrypoint
```

Include every Python module and data file the entrypoint imports at
runtime. The default command copies the starter-kit runtime used by
Python Simple. Add extra files with `--include`.

A worked `submission.json` for that default layout is in
[`examples/python_simple/reference_submission/submission.json`](../examples/python_simple/reference_submission/submission.json).

## `submission.json`

The field contract is [`arena_clients/submission.schema.json`](../arena_clients/submission.schema.json).
The packager writes this file, including `metadata.agentId` and
`metadata.agentName`. Required shape:

```json
{
  "schemaVersion": 1,
  "kind": "GauntletAgentSubmission",
  "runtime": {
    "language": "python",
    "pythonVersion": "3.12",
    "workingDirectory": ".",
    "entrypoint": ["python", "examples/python_simple/agent.py"]
  },
  "dependencies": {
    "lockfile": "requirements.lock"
  },
  "artifact": {
    "files": [
      "examples/python_simple/agent.py",
      "requirements.lock"
    ]
  }
}
```

Rules the organizer enforces:

- `schemaVersion` is `1` and `kind` is `GauntletAgentSubmission`.
- Runtime is Python 3.12. `workingDirectory` is `"."`.
- `runtime.entrypoint` is an argv array. The first item is `python` or
  `python3`. The second item is a file listed in `artifact.files`. No shell
  script, shell operators, or absolute path.
- `artifact.files` lists every regular file in the directory except
  `submission.json`. Paths are normalized and repository-relative. Do not
  list directories. Do not leave undeclared files.

## Dependency lock

The package must install without downloading floating versions:

```bash
python3.12 -m pip install --require-hashes -r requirements.lock
```

The packager copies `requirements/requirements.lock` to `requirements.lock`
inside the submission. If you added dependencies, regenerate a hash-locked
file with Python 3.12 compatibility and pass it as `--lockfile`.

The agent must not install packages while it runs.

## What must never be included

Leave these out of the **archive** (inside `gauntlet-submission/`):

- `.env` and `.env.*`
- API keys, tokens, cookies, private keys, and certificates
- `.venv/`, `venv/`, `__pycache__/`, `.pytest_cache/`, and tool caches
- Logs, receipts, previous challenge outputs, generated images, and local databases
- Tests, documentation, editor settings, and unused framework examples
- A README, LICENSE, or `.gitignore` added on top of the packaged tree
- Symlinks
- A participant Dockerfile — the organizer builds the image from its own recipe

Those README, LICENSE, and `.gitignore` rules apply to files **inside the
tarball**, not necessarily beside the two files on GitHub. Organizers
download only the named tarball and checksum pair.

Never print secret values while packaging.

## Certify / play rehearsal

Plain `python -m arena_clients.doctor` only proves connectivity. Before you
package, rehearse a real play against Practice:

Set `AGENT_ID` to your team identity first. Certify never invents a
`doctor-*` identity.

```bash
python -m arena_clients.doctor --certify --json
```

This is the frozen certify contract:

- `--certify` waits for lobby, registers, waits for GO, submits, then
  checks that an exact retry returns the canonical receipt and a
  different answer returns HTTP 409.
- The sandbox injects `ARENA_USAGE_SCOPE`. Certify mode never invents a
  `doctor-<uuid>` scope. If that env var is unset locally, the doctor
  reads `usage_scope` from `/api/competition`.
- `--json` writes the checklist object to stdout as pretty JSON. Optional
  `--output PATH` also writes that same JSON to a file.
- Exit 0 only if every checklist item is PASS; otherwise exit 1.

Checklist keys, each `PASS` or `FAIL`: `registered_in_lobby`,
`waited_for_go`, `in_frozen_roster`, `answer_accepted`, `scored`,
`retry_is_canonical`, `conflicting_answer_rejected`,
`attribution_scope_exact`.

`--certify` still refuses to run unless the target is Practice (or a
warmup battle). This play rehearsal is not the packaging check in the
next section.

## Package and self-check

From the starter-kit root:

```bash
python -m arena_clients.package --agent-id my-team --agent-name "My Team"
python -m arena_clients.package --check dist/gauntlet-submission
```

That command writes `dist/gauntlet-submission/`,
`dist/<agent-id>-submission.tar.gz`, and
`dist/<agent-id>-submission.tar.gz.sha256`. It fails before writing
when the directory would be rejected: schema, normalized paths, missing
or undeclared files, symlinks, per-file and total size limits,
credential-shaped names, and secret heuristics.

Useful flags:

| Flag | Purpose |
|---|---|
| `--agent-id ID` | Required. Approved agent id (lowercase letters, digits, hyphens). |
| `--agent-name NAME` | Required. Leaderboard display name. |
| `--entrypoint PATH` | Python file to launch (default: `examples/python_simple/agent.py`) |
| `--entrypoint-arg ARG` | Extra argv item after the script; repeatable |
| `--include PATH` | Extra source-relative file to copy; repeatable |
| `--lockfile PATH` | Hash-locked requirements file to copy |
| `--output DIR` | Destination for `gauntlet-submission/` and the tarball (default: `dist/`) |
| `--force` | Replace an existing `gauntlet-submission/` directory |

The packaged agent reads `ARENA_SERVER` and `ARENA_API_KEY` from the
process environment. It must work when no `.env` file is present.

## Handoff

Create a **new public GitHub repository** (not a fork of this kit). Push
**only** `dist/<agent-id>-submission.tar.gz` and
`dist/<agent-id>-submission.tar.gz.sha256` to the repository root. Keep
the filenames exactly as the packager wrote them. Send the organizer that
repository URL.

Do not push the unpacked `dist/gauntlet-submission/` tree. Do not send a
lone `my_strategy.py`. Do not use GitHub's auto-generated source zip. Do
not send API keys; the organizer supplies runtime configuration.
