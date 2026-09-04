# Starter Kit Security Policy

## Supported versions

Use the latest published Starter Kit together with the matching Agent Gauntlet
runtime. Security fixes target the current default branch and the generated
`requirements/requirements.lock` / `requirements/requirements-all.lock` dependency profiles.

## Reporting a vulnerability

Use the repository host's private security-reporting channel or a confidential
maintainer issue. Do not post battle keys, `.env` files, private endpoints,
prompts, model outputs, or run logs in a public issue.

Include the affected revision, a secret-free reproduction, observed impact, and
whether the issue affects REST, MCP, the proxy, or a framework example.

## Competitor safety requirements

- Remote `ARENA_SERVER` values must use HTTPS; loopback HTTP is development-only.
- Send the battle key in `X-Arena-API-Key` for REST and MCP, never in a URL.
- Send the battle key as `Authorization: Bearer` only to the organizer's proxy.
- Never provide upstream provider keys to the Starter Kit or commit `.env`.
- Install from the hash-locked requirement entrypoints. Change `.in` sources and
  regenerate locks when updating dependencies.
- Treat logs and saved artifacts as sensitive event data.
