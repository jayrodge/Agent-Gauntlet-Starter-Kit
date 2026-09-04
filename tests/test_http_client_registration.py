from __future__ import annotations

import os
import unittest
from unittest import mock

from arena_clients.http_client import (
    ArenaAPIError,
    ArenaConnectionError,
    HttpArenaClient,
)


class RegistrationRetryTests(unittest.TestCase):
    def test_registration_uses_bounded_exponential_backoff(self) -> None:
        client = HttpArenaClient(
            api_base="https://practice.example",
            api_key="team-key",
        )
        client._request = mock.Mock(
            side_effect=[
                ArenaAPIError(409, "lobby closed"),
                ArenaAPIError(409, "lobby closed"),
                {
                    "session_id": "session-one",
                    "agent_id": "agent-one",
                    "agent_name": "Agent One",
                    "status": "connected",
                    "started_at": 100.0,
                },
            ]
        )

        with (
            mock.patch.dict(os.environ, {"ARENA_REGISTRATION_TIMEOUT_S": "600"}),
            mock.patch("arena_clients.http_client.time.monotonic", return_value=100.0),
            mock.patch("arena_clients.http_client.time.sleep") as sleep,
        ):
            result = client.register("agent-one", "Agent One")

        self.assertEqual(result.agent_id, "agent-one")
        self.assertEqual(sleep.call_args_list, [mock.call(1.0), mock.call(2.0)])

    def test_registration_timeout_fails_instead_of_retrying_forever(self) -> None:
        client = HttpArenaClient(
            api_base="https://practice.example",
            api_key="team-key",
        )
        client._request = mock.Mock(side_effect=ArenaAPIError(409, "lobby closed"))

        with (
            mock.patch.dict(os.environ, {"ARENA_REGISTRATION_TIMEOUT_S": "3"}),
            mock.patch(
                "arena_clients.http_client.time.monotonic",
                side_effect=[100.0, 100.0, 101.0, 103.0],
            ),
            mock.patch("arena_clients.http_client.time.sleep"),
        ):
            with self.assertRaisesRegex(ArenaConnectionError, "3 seconds"):
                client.register("agent-one", "Agent One")

    def test_registration_too_late_stops_without_polling(self) -> None:
        client = HttpArenaClient(
            api_base="https://practice.example",
            api_key="team-key",
        )
        error = ArenaAPIError(
            409,
            (
                '{"detail":{"code":"registration_too_late",'
                '"message":"Register during the next lobby."}}'
            ),
        )
        client._request = mock.Mock(side_effect=error)

        with mock.patch(
            "arena_clients.http_client.time.sleep"
        ) as sleep, self.assertRaisesRegex(
            ArenaConnectionError,
            "closed for the current round",
        ):
            client.register("agent-one", "Agent One")

        self.assertEqual(error.code, "registration_too_late")
        client._request.assert_called_once()
        sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
