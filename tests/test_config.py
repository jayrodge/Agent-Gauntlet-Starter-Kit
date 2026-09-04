from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arena_clients.config import ensure_connected, get_api_base, get_mcp_url, get_proxy_host


class ArenaUrlResolutionTests(unittest.TestCase):
    def test_loopback_arena_server_keeps_legacy_service_ports(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"ARENA_SERVER": "127.0.0.1"},
            clear=True,
        ):
            self.assertEqual(get_api_base(), "http://127.0.0.1:8000")
            self.assertEqual(get_mcp_url(), "http://127.0.0.1:5001")
            self.assertEqual(get_proxy_host(), "http://127.0.0.1:4001")

    def test_bare_remote_server_uses_https_gateway_origin(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"ARENA_SERVER": "arena.example.com"},
            clear=True,
        ):
            self.assertEqual(get_api_base(), "https://arena.example.com")
            self.assertEqual(get_mcp_url(), "https://arena.example.com")
            self.assertEqual(get_proxy_host(), "https://arena.example.com/proxy")

    def test_explicit_remote_http_is_rejected(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"ARENA_SERVER": "http://arena.example.com"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "HTTPS"):
                get_api_base()

    def test_https_arena_server_uses_reverse_proxy_origin(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"ARENA_SERVER": "https://arena.example.com"},
            clear=True,
        ):
            self.assertEqual(get_api_base(), "https://arena.example.com")
            self.assertEqual(get_mcp_url(), "https://arena.example.com")
            self.assertEqual(get_proxy_host(), "https://arena.example.com/proxy")

    def test_proxy_host_accepts_arena_server_template(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "ARENA_SERVER": "https://arena.example.com",
                "LLM_PROXY_HOST": "ARENA_SERVER/proxy",
            },
            clear=True,
        ):
            self.assertEqual(get_proxy_host(), "https://arena.example.com/proxy")

    def test_key_validation_uses_header_without_query_credential(self) -> None:
        response = mock.Mock(status_code=200)
        response.json.return_value = {"valid": True}
        with (
            mock.patch.dict(
                os.environ,
                {
                    "ARENA_SERVER": "https://arena.example.com",
                    "ARENA_API_KEY": "secret/key",
                },
                clear=True,
            ),
            mock.patch("arena_clients.config.requests.get", return_value=response) as get,
        ):
            ensure_connected.cache_clear()
            ensure_connected()

        url = get.call_args.args[0]
        headers = get.call_args.kwargs["headers"]
        self.assertEqual(url, "https://arena.example.com/api/keys/validate")
        self.assertEqual(headers["X-Arena-API-Key"], "secret/key")


if __name__ == "__main__":
    unittest.main()
