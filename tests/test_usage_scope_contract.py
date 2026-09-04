"""Competitor clients must keep reading usage_scope from the public competition payload."""

from __future__ import annotations

import json
import unittest
from unittest import mock

from arena_clients.http_client import HttpArenaClient

MINTED_SCOPE = "battle-scope-w0-redaction"


class UsageScopeContractTests(unittest.TestCase):
    def test_fetch_usage_scope_resolves_top_level_field_from_redacted_payload(self) -> None:
        client = HttpArenaClient(
            api_base="https://arena.example",
            api_key="team-key",
        )
        payload = {
            "status": "running",
            "phase": "running",
            "challenge_type": "text",
            "puzzle_id": "breakfast_logic_001",
            "usage_scope": MINTED_SCOPE,
            "run_id": "text-abc123",
            "runs": {
                "text": {
                    "run_id": "text-abc123",
                    "run_type": "text",
                    "status": "running",
                    "started_at": 100.0,
                    "ended_at": None,
                    "exit_code": None,
                    "puzzle_id": "breakfast_logic_001",
                    "usage_scope": MINTED_SCOPE,
                }
            },
        }
        client._request = mock.Mock(return_value=payload)

        scope = client.fetch_usage_scope()

        self.assertEqual(scope, MINTED_SCOPE)
        client._request.assert_called_once_with("GET", "/api/competition")
        serialized = json.dumps(payload)
        self.assertNotIn("arena_api_key", serialized)
        self.assertNotIn("command", serialized)
        self.assertNotIn("log_path", serialized)
        self.assertIs(client.fetch_usage_scope(), scope)
