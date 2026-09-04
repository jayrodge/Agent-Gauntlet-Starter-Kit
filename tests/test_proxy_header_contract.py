from __future__ import annotations

import os
import unittest
from unittest import mock

from arena_clients.proxy_headers import build_proxy_headers, resolve_usage_scope


class ProxyHeaderContractTests(unittest.TestCase):
    def test_build_proxy_headers_uses_explicit_values(self) -> None:
        self.assertEqual(
            build_proxy_headers("team-cipher", "round-1"),
            {"X-Agent-ID": "team-cipher", "X-Round-ID": "round-1"},
        )

    def test_build_proxy_headers_falls_back_to_environment(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"AGENT_ID": "team-env", "ARENA_USAGE_SCOPE": "scope-env"},
        ):
            self.assertEqual(
                build_proxy_headers(),
                {"X-Agent-ID": "team-env", "X-Round-ID": "scope-env"},
            )

    def test_resolve_usage_scope_returns_none_for_empty_scope(self) -> None:
        with mock.patch.dict(os.environ, {"ARENA_USAGE_SCOPE": "   "}):
            self.assertIsNone(resolve_usage_scope())


if __name__ == "__main__":
    unittest.main()
