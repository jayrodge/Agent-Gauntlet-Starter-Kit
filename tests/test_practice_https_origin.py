from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arena_clients.config import get_api_base, get_mcp_url, get_proxy_host


class PracticeHttpsOriginTests(unittest.TestCase):
    def test_sandboxed_agent_resolves_single_https_origin(self) -> None:
        """A sandboxed agent given https://ARENA_SERVER must hit /proxy.

        Practice terminates TLS at one origin. Caddy (deploy/Caddyfile.template)
        serves /api on :8000, /sse on :5001, and strips /proxy toward :4001.
        """
        origin = "https://practice.example.com"
        with mock.patch.dict(os.environ, {"ARENA_SERVER": origin}, clear=True):
            self.assertEqual(get_api_base(), origin)
            self.assertEqual(get_mcp_url(), origin)
            self.assertEqual(get_proxy_host(), f"{origin}/proxy")


if __name__ == "__main__":
    unittest.main()
