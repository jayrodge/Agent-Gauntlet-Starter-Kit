from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from arena_clients.mcp_client import McpArenaClient, McpArenaError, build_image_tool_arguments


class MCPAuthenticationTransportTests(unittest.TestCase):
    def test_credential_is_carried_in_header_not_url(self) -> None:
        client = McpArenaClient(
            "https://practice.example",
            api_key="secret/key",
        )

        self.assertEqual(client.sse_url, "https://practice.example/sse")
        self.assertEqual(
            client._headers,
            {"X-Arena-API-Key": "secret/key"},
        )


class ImageToolArgumentTests(unittest.TestCase):
    def test_builds_selected_model_and_byo_proxy_key_from_live_schema(self) -> None:
        tool_defs = [
            SimpleNamespace(
                name="image_generate",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "model": {"type": ["string", "null"]},
                        "proxy_api_key": {"type": ["string", "null"]},
                    },
                },
            )
        ]

        arguments = build_image_tool_arguments(
            tool_defs,
            "image_generate",
            selected_model="gpt-5-image-mini",
            llm_api_key="provider-key",
            arena_api_key="team-key",
        )

        self.assertEqual(
            arguments,
            {
                "model": "gpt-5-image-mini",
                "proxy_api_key": "provider-key",
            },
        )


class ClueHydrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_fetches_every_listed_clue_in_server_order(self) -> None:
        client = McpArenaClient("https://practice.example", api_key="team-key")
        client.list_clues = mock.AsyncMock(return_value=["clue_0", "clue_1"])
        client.get_clue = mock.AsyncMock(
            side_effect=[
                SimpleNamespace(text="First clue"),
                SimpleNamespace(text="Second clue"),
            ]
        )

        clues = await client.get_all_clue_texts("agent-one")

        self.assertEqual(clues, ["First clue", "Second clue"])
        self.assertEqual(
            client.get_clue.await_args_list,
            [
                mock.call("clue_0", "agent-one"),
                mock.call("clue_1", "agent-one"),
            ],
        )


class NotImageChallengeClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_get_image_challenge_raises_terminal_not_image_challenge(self) -> None:
        client = McpArenaClient("https://practice.example", api_key="team-key")
        client._session = mock.Mock()
        client._session.call_tool = mock.AsyncMock(return_value=object())
        client._parse_result = mock.Mock(
            return_value={
                "error": "Authoritative assignment is not an image challenge.",
                "code": "not_image_challenge",
                "accepted": False,
                "challenge_type": "text",
            }
        )

        with self.assertRaises(McpArenaError) as ctx:
            await client.get_image_challenge("alpha")

        self.assertEqual(ctx.exception.code, "not_image_challenge")
        self.assertTrue(ctx.exception.is_terminal)
        self.assertNotIn("locked", str(ctx.exception).lower())

    def test_locked_error_is_not_terminal(self) -> None:
        exc = McpArenaError("Challenge locked. Waiting for organizer to open lobby.")
        self.assertIsNone(exc.code)
        self.assertFalse(exc.is_terminal)


if __name__ == "__main__":
    unittest.main()
