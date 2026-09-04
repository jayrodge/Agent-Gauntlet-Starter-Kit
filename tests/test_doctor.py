from __future__ import annotations

from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arena_clients import doctor


class DoctorHelperTests(unittest.TestCase):
    def test_usage_total_tokens_reads_nested_usage(self) -> None:
        self.assertEqual(
            doctor._usage_total_tokens({"usage": {"total_tokens": 12}}),
            12,
        )
        self.assertEqual(doctor._usage_total_tokens({"prompt_tokens": 3}), 3)
        self.assertEqual(doctor._usage_total_tokens({"usage": {}}), 0)

    def test_check_resolved_urls_prints_ok(self) -> None:
        with (
            mock.patch.dict(
                "os.environ",
                {"ARENA_SERVER": "https://arena.example.com"},
                clear=True,
            ),
            mock.patch("builtins.print") as printed,
        ):
            doctor.check_resolved_urls()
        joined = " ".join(str(call.args[0]) for call in printed.call_args_list)
        self.assertIn("OK  resolved URLs", joined)

    def test_check_api_key_surfaces_ensure_connected_failure(self) -> None:
        with (
            mock.patch.object(
                doctor.ensure_connected,
                "cache_clear",
                create=True,
            ),
            mock.patch.object(
                doctor,
                "ensure_connected",
                side_effect=SystemExit("bad key"),
            ),
            mock.patch("builtins.print"),
            self.assertRaises(SystemExit) as raised,
        ):
            doctor.check_api_key()
        self.assertEqual(raised.exception.code, 1)


PRACTICE_HEALTH = {
    "status": "ok",
    "service": "practice-arena-api",
    "phase": "running",
    "practice_gate_mode": "off",
}
OPERATOR_HEALTH = {"status": "healthy", "service": "agent-arena-api"}


class DoctorFullGateTests(unittest.TestCase):
    """The --full round trip must only run where a submission cannot count."""

    def _gate(self, health: object, competition: object) -> str:
        responses = {"/api/health": health, "/api/competition": competition}
        with (
            mock.patch.object(doctor, "_fetch_json_or_none", side_effect=lambda path: responses[path]),
            mock.patch("builtins.print"),
        ):
            return doctor.check_full_gate()

    def _refusal_message(self, health: object, competition: object) -> str:
        responses = {"/api/health": health, "/api/competition": competition}
        with (
            mock.patch.object(doctor, "_fetch_json_or_none", side_effect=lambda path: responses[path]),
            mock.patch("builtins.print") as printed,
            self.assertRaises(SystemExit) as raised,
        ):
            doctor.check_full_gate()
        self.assertEqual(raised.exception.code, 1)
        return " ".join(str(call.args[0]) for call in printed.call_args_list)

    def test_practice_service_is_allowed(self) -> None:
        self.assertEqual(self._gate(PRACTICE_HEALTH, {"phase": "running"}), "practice")

    def test_practice_gate_mode_field_is_allowed_without_service_name(self) -> None:
        health = {"status": "ok", "phase": "running", "practice_gate_mode": "cycle"}
        self.assertEqual(self._gate(health, None), "practice")

    def test_operator_warmup_battle_is_allowed(self) -> None:
        competition = {"phase": "running", "warmup": True}
        self.assertEqual(self._gate(OPERATOR_HEALTH, competition), "warmup")

    def test_operator_live_battle_is_refused(self) -> None:
        competition = {"phase": "running", "warmup": False}
        message = self._refusal_message(OPERATOR_HEALTH, competition)
        self.assertIn("Refusing --full", message)
        self.assertIn("official", message)

    def test_unreachable_health_is_refused(self) -> None:
        message = self._refusal_message(None, None)
        self.assertIn("/api/health could not be read", message)

    def test_unreachable_competition_on_unknown_server_is_refused(self) -> None:
        message = self._refusal_message(OPERATOR_HEALTH, None)
        self.assertIn("Refusing --full", message)

    def test_truthy_non_boolean_warmup_is_refused(self) -> None:
        # Fail closed: only an explicit warmup=true or phase=warmup counts.
        competition = {"phase": "running", "warmup": "maybe"}
        message = self._refusal_message(OPERATOR_HEALTH, competition)
        self.assertIn("Refusing --full", message)

    def test_run_doctor_full_refuses_before_any_round_trip(self) -> None:
        with (
            mock.patch.object(doctor, "_load_env"),
            mock.patch.object(doctor, "check_resolved_urls"),
            mock.patch.object(doctor, "check_api_health"),
            mock.patch.object(doctor, "check_api_key"),
            mock.patch.object(doctor, "check_mcp_tools"),
            mock.patch.object(doctor, "check_proxy_models", return_value=["m"]),
            mock.patch.object(doctor, "check_attributed_inference_and_usage"),
            mock.patch.object(doctor, "_fetch_json_or_none", return_value=OPERATOR_HEALTH),
            mock.patch.object(doctor, "check_full_round_trip") as round_trip,
            mock.patch("builtins.print"),
            self.assertRaises(SystemExit),
        ):
            doctor.run_doctor(full=True)
        round_trip.assert_not_called()

    def test_run_doctor_full_runs_round_trip_on_practice(self) -> None:
        with (
            mock.patch.object(doctor, "_load_env"),
            mock.patch.object(doctor, "check_resolved_urls"),
            mock.patch.object(doctor, "check_api_health"),
            mock.patch.object(doctor, "check_api_key"),
            mock.patch.object(doctor, "check_mcp_tools"),
            mock.patch.object(doctor, "check_proxy_models", return_value=["m"]),
            mock.patch.object(doctor, "check_attributed_inference_and_usage"),
            mock.patch.object(doctor, "_fetch_json_or_none", return_value=PRACTICE_HEALTH),
            mock.patch.object(doctor, "check_full_round_trip") as round_trip,
            mock.patch("builtins.print"),
        ):
            self.assertEqual(doctor.run_doctor(full=True), 0)
        round_trip.assert_called_once_with("practice")

    def test_default_run_skips_the_round_trip_entirely(self) -> None:
        with (
            mock.patch.object(doctor, "_load_env"),
            mock.patch.object(doctor, "check_resolved_urls"),
            mock.patch.object(doctor, "check_api_health"),
            mock.patch.object(doctor, "check_api_key"),
            mock.patch.object(doctor, "check_mcp_tools"),
            mock.patch.object(doctor, "check_proxy_models", return_value=["m"]),
            mock.patch.object(doctor, "check_attributed_inference_and_usage"),
            mock.patch.object(doctor, "check_full_gate") as gate,
            mock.patch.object(doctor, "check_full_round_trip") as round_trip,
            mock.patch("builtins.print"),
        ):
            self.assertEqual(doctor.run_doctor(), 0)
        gate.assert_not_called()
        round_trip.assert_not_called()

    def test_parse_args_defaults_to_read_only(self) -> None:
        args = doctor.parse_args([])
        self.assertFalse(args.full)
        self.assertFalse(args.certify)
        self.assertFalse(args.json)
        self.assertIsNone(args.output)
        self.assertIsNone(args.modality)
        self.assertTrue(doctor.parse_args(["--full"]).full)
        certified = doctor.parse_args(["--certify", "--json", "--output", "out.json"])
        self.assertTrue(certified.certify)
        self.assertTrue(certified.json)
        self.assertEqual(certified.output, "out.json")
        image = doctor.parse_args(["--certify", "--modality", "image"])
        self.assertEqual(image.modality, "image")

    def test_parse_args_json_requires_certify(self) -> None:
        with self.assertRaises(SystemExit):
            doctor.parse_args(["--json"])
        with self.assertRaises(SystemExit):
            doctor.parse_args(["--modality", "image"])


class DoctorFullRoundTripTests(unittest.TestCase):
    def test_score_value_reads_common_score_keys(self) -> None:
        self.assertEqual(doctor._score_value({"final_score": 91.5}), 91.5)
        self.assertEqual(doctor._score_value({"total_score": "80"}), 80.0)
        self.assertIsNone(doctor._score_value({"quality": 1}))
        self.assertIsNone(doctor._score_value(None))

    def test_round_trip_fails_when_no_score_is_returned(self) -> None:
        client = mock.Mock()
        challenge = mock.Mock(puzzle_id="p1", challenge_type="text")
        client.submit.return_value = mock.Mock(accepted=True, status="submitted", score=None)
        with (
            mock.patch.object(doctor, "HttpArenaClient", return_value=client),
            mock.patch.object(doctor, "_fetch_text_challenge"),
            mock.patch.object(doctor.asyncio, "run", return_value=(["arena.get_challenge"], challenge)),
            mock.patch("builtins.print"),
            self.assertRaises(SystemExit) as raised,
        ):
            doctor.check_full_round_trip("practice")
        self.assertEqual(raised.exception.code, 1)
        client.submit.assert_called_once()

    def test_round_trip_passes_on_scored_submission(self) -> None:
        client = mock.Mock()
        challenge = mock.Mock(puzzle_id="p1", challenge_type="text")
        client.submit.return_value = mock.Mock(
            accepted=True,
            status="submitted",
            score={"final_score": 72.5},
        )
        with (
            mock.patch.object(doctor, "HttpArenaClient", return_value=client),
            mock.patch.object(doctor, "_fetch_text_challenge"),
            mock.patch.object(doctor.asyncio, "run", return_value=(["arena.get_challenge"], challenge)),
            mock.patch("builtins.print") as printed,
        ):
            doctor.check_full_round_trip("practice")
        joined = " ".join(str(call.args[0]) for call in printed.call_args_list)
        self.assertIn("full round trip: score", joined)
        registered_agent_id = client.register.call_args.args[0]
        self.assertTrue(registered_agent_id.startswith("doctor-full-"))

    def test_round_trip_passes_when_answer_is_wrong_but_scored(self) -> None:
        # `accepted` means the judge liked the answer, not that the submit landed.
        # doctor --full uses a probe string that Practice will score but reject.
        client = mock.Mock()
        challenge = mock.Mock(puzzle_id="p1", challenge_type="text")
        client.submit.return_value = mock.Mock(
            accepted=False,
            status="submitted",
            score={"final_score": 12.0, "quality_score": 0},
        )
        with (
            mock.patch.object(doctor, "HttpArenaClient", return_value=client),
            mock.patch.object(doctor, "_fetch_text_challenge"),
            mock.patch.object(doctor.asyncio, "run", return_value=(["arena.get_challenge"], challenge)),
            mock.patch("builtins.print") as printed,
        ):
            doctor.check_full_round_trip("practice")
        joined = " ".join(str(call.args[0]) for call in printed.call_args_list)
        self.assertIn("full round trip: submit", joined)
        self.assertIn("full round trip: score", joined)

    def test_round_trip_refuses_image_only_tool_set(self) -> None:
        client = mock.Mock()
        with (
            mock.patch.object(doctor, "HttpArenaClient", return_value=client),
            mock.patch.object(doctor, "_fetch_text_challenge"),
            mock.patch.object(
                doctor.asyncio,
                "run",
                return_value=(["arena.image.get_challenge"], None),
            ),
            mock.patch("builtins.print"),
            self.assertRaises(SystemExit),
        ):
            doctor.check_full_round_trip("practice")
        client.submit.assert_not_called()


class DoctorCertifyTests(unittest.TestCase):
    """Frozen --certify / --json contract."""

    def test_checklist_shape_is_frozen(self) -> None:
        checklist = doctor.empty_certify_checklist()
        self.assertEqual(tuple(checklist), doctor.CERTIFY_CHECKLIST_KEYS)
        self.assertEqual(
            doctor.CERTIFY_CHECKLIST_KEYS,
            (
                "registered_in_lobby",
                "waited_for_go",
                "in_frozen_roster",
                "answer_accepted",
                "scored",
                "retry_is_canonical",
                "conflicting_answer_rejected",
                "attribution_scope_exact",
            ),
        )
        self.assertTrue(all(value == "FAIL" for value in checklist.values()))
        self.assertFalse(doctor.checklist_all_passed(checklist))
        for key in doctor.CERTIFY_CHECKLIST_KEYS:
            checklist[key] = "PASS"
        self.assertTrue(doctor.checklist_all_passed(checklist))

    def test_format_certify_checklist_is_pretty_json(self) -> None:
        checklist = doctor.empty_certify_checklist()
        checklist["scored"] = "PASS"
        text = doctor.format_certify_checklist(checklist)
        payload = json.loads(text)
        self.assertEqual(tuple(payload), doctor.CERTIFY_CHECKLIST_KEYS)
        self.assertEqual(payload["scored"], "PASS")
        self.assertEqual(payload["registered_in_lobby"], "FAIL")
        self.assertIn("\n", text)
        self.assertTrue(text.startswith("{\n"))

    def test_resolve_certify_usage_scope_never_invents_doctor_scope(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            with mock.patch.object(doctor, "_fetch_json_or_none", return_value=None):
                self.assertEqual(doctor.resolve_certify_usage_scope(), "")
            with mock.patch.object(
                doctor, "_fetch_json_or_none", return_value={"phase": "lobby"}
            ):
                self.assertEqual(doctor.resolve_certify_usage_scope(), "")
            with mock.patch.object(
                doctor,
                "_fetch_json_or_none",
                return_value={"usage_scope": "battle-scope-1"},
            ):
                self.assertEqual(doctor.resolve_certify_usage_scope(), "battle-scope-1")
        with mock.patch.dict(
            "os.environ",
            {"ARENA_USAGE_SCOPE": "injected-scope"},
            clear=True,
        ):
            with mock.patch.object(doctor, "_fetch_json_or_none") as fetch:
                self.assertEqual(doctor.resolve_certify_usage_scope(), "injected-scope")
                fetch.assert_not_called()

    def test_resolve_certify_agent_id_requires_real_identity(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(doctor.resolve_certify_agent_id(), "")
        with mock.patch.dict("os.environ", {"AGENT_ID": "team-alpha"}, clear=True):
            self.assertEqual(doctor.resolve_certify_agent_id(), "team-alpha")

    def test_certify_attribution_does_not_fallback_to_unscoped_usage(self) -> None:
        calls: list[tuple[str, str]] = []

        def fake_http(method: str, url: str, **_kwargs: object):
            calls.append((method, url))
            if method == "POST":
                return 200, {"choices": [{"message": {"content": "pong"}}]}
            return 404, {"detail": "not keyed"}

        with (
            mock.patch.dict(
                "os.environ",
                {"AGENT_ID": "team-alpha", "ARENA_USAGE_SCOPE": "scope-exact"},
                clear=False,
            ),
            mock.patch.object(doctor, "get_proxy_host", return_value="https://arena.example.com/proxy"),
            mock.patch.object(doctor, "get_arena_api_key", return_value="key"),
            mock.patch.object(doctor, "_http_json", side_effect=fake_http),
            mock.patch("builtins.print"),
            self.assertRaises(SystemExit) as raised,
        ):
            doctor.check_attributed_inference_and_usage(["m"], certify=True)
        self.assertEqual(raised.exception.code, 1)
        usage_urls = [url for method, url in calls if method == "GET"]
        self.assertEqual(len(usage_urls), 1)
        self.assertIn("/usage/scope-exact/team-alpha", usage_urls[0])
        self.assertFalse(any(url.endswith("/usage/team-alpha") for url in usage_urls))

    def test_certify_attribution_refuses_missing_scope_without_inventing(self) -> None:
        with (
            mock.patch.dict("os.environ", {"AGENT_ID": "team-alpha"}, clear=True),
            mock.patch.object(doctor, "get_proxy_host", return_value="https://arena.example.com/proxy"),
            mock.patch.object(doctor, "get_arena_api_key", return_value="key"),
            mock.patch.object(doctor, "resolve_certify_usage_scope", return_value=""),
            mock.patch.object(doctor, "_http_json") as http_json,
            mock.patch("builtins.print"),
            self.assertRaises(SystemExit),
        ):
            doctor.check_attributed_inference_and_usage(["m"], certify=True)
        http_json.assert_not_called()

    def test_certify_uses_real_agent_id_not_doctor_full(self) -> None:
        client = mock.Mock()
        challenge = mock.Mock(puzzle_id="p1", challenge_type="text")
        official = mock.Mock(
            accepted=True,
            status="submitted",
            answer=doctor.CERTIFY_ANSWER,
            agent_id="team-alpha",
            score={"final_score": 70.0},
        )
        client.submit.side_effect = [
            official,
            official,
            doctor.ArenaAPIError(409, "conflict"),
        ]
        client.get_session.return_value = {
            "final_answer": doctor.CERTIFY_ANSWER,
            "submitted_at": 1.0,
        }
        client.get_competition.return_value = {
            "phase": "running",
            "eligible_agent_ids": ["team-alpha"],
            "usage_scope": "scope-1",
        }
        checklist = doctor.empty_certify_checklist()
        with (
            mock.patch.dict("os.environ", {"AGENT_ID": "team-alpha"}, clear=False),
            mock.patch.object(doctor, "HttpArenaClient", return_value=client),
            mock.patch.object(doctor, "wait_for_phase", side_effect=["lobby", "running"]),
            mock.patch.object(
                doctor.asyncio,
                "run",
                return_value=(["arena.get_challenge"], challenge),
            ),
            mock.patch("builtins.print"),
        ):
            doctor.check_certify_round_trip("practice", checklist)
        self.assertEqual(client.register.call_args.args[0], "team-alpha")
        self.assertFalse(client.register.call_args.args[0].startswith("doctor-full-"))
        self.assertTrue(doctor.checklist_all_passed({
            **checklist,
            "attribution_scope_exact": "PASS",
        }))
        self.assertEqual(client.submit.call_count, 3)

    def test_certify_thought_failure_is_fatal(self) -> None:
        client = mock.Mock()
        challenge = mock.Mock(puzzle_id="p1", challenge_type="text")
        client.broadcast_thought.side_effect = doctor.ArenaAPIError(500, "thought down")
        client.get_competition.return_value = {
            "phase": "running",
            "eligible_agent_ids": ["team-alpha"],
        }
        checklist = doctor.empty_certify_checklist()
        with (
            mock.patch.dict("os.environ", {"AGENT_ID": "team-alpha"}, clear=False),
            mock.patch.object(doctor, "HttpArenaClient", return_value=client),
            mock.patch.object(doctor, "wait_for_phase", side_effect=["lobby", "running"]),
            mock.patch.object(
                doctor.asyncio,
                "run",
                return_value=(["arena.get_challenge"], challenge),
            ),
            mock.patch("builtins.print"),
            self.assertRaises(SystemExit) as raised,
        ):
            doctor.check_certify_round_trip("practice", checklist)
        self.assertEqual(raised.exception.code, 1)
        client.broadcast_thought.assert_called_once()
        client.submit.assert_not_called()
        self.assertEqual(checklist["registered_in_lobby"], "PASS")
        self.assertEqual(checklist["waited_for_go"], "PASS")
        self.assertEqual(checklist["answer_accepted"], "FAIL")

    def test_run_doctor_certify_refuses_off_practice_and_skips_play(self) -> None:
        play = mock.Mock()
        with (
            mock.patch.object(doctor, "_load_env"),
            mock.patch.object(doctor, "check_resolved_urls"),
            mock.patch.object(doctor, "check_api_health"),
            mock.patch.object(doctor, "check_api_key"),
            mock.patch.object(doctor, "check_mcp_tools"),
            mock.patch.object(doctor, "check_proxy_models", return_value=["m"]),
            mock.patch.object(doctor, "check_attributed_inference_and_usage"),
            mock.patch.object(doctor, "check_full_round_trip") as full_trip,
            mock.patch.object(doctor, "check_certify_round_trip", play),
            mock.patch.object(
                doctor, "_fetch_json_or_none", return_value=OPERATOR_HEALTH
            ),
            mock.patch("builtins.print"),
            self.assertRaises(SystemExit),
        ):
            doctor.run_doctor(certify=True)
        play.assert_not_called()
        full_trip.assert_not_called()

    def test_run_doctor_certify_uses_full_gate_then_play_path(self) -> None:
        def play(_target: str, checklist: dict[str, str], **_kwargs: object) -> None:
            for key in doctor.CERTIFY_CHECKLIST_KEYS:
                if key != "attribution_scope_exact":
                    checklist[key] = "PASS"

        with (
            mock.patch.object(doctor, "_load_env"),
            mock.patch.object(doctor, "check_resolved_urls"),
            mock.patch.object(doctor, "check_api_health"),
            mock.patch.object(doctor, "check_api_key"),
            mock.patch.object(doctor, "check_mcp_tools"),
            mock.patch.object(doctor, "check_proxy_models", return_value=["m"]),
            mock.patch.object(doctor, "check_attributed_inference_and_usage"),
            mock.patch.object(
                doctor, "_fetch_json_or_none", return_value=PRACTICE_HEALTH
            ),
            mock.patch.object(doctor, "check_full_round_trip") as full_trip,
            mock.patch.object(doctor, "check_certify_round_trip", side_effect=play),
            mock.patch("builtins.print"),
        ):
            self.assertEqual(doctor.run_doctor(certify=True), 0)
        full_trip.assert_not_called()

    def test_json_stdout_and_output_file_are_the_same_checklist(self) -> None:
        def play(_target: str, checklist: dict[str, str], **_kwargs: object) -> None:
            for key in doctor.CERTIFY_CHECKLIST_KEYS:
                if key != "attribution_scope_exact":
                    checklist[key] = "PASS"

        stdout = StringIO()
        with (
            mock.patch.object(doctor, "_load_env"),
            mock.patch.object(doctor, "check_resolved_urls"),
            mock.patch.object(doctor, "check_api_health"),
            mock.patch.object(doctor, "check_api_key"),
            mock.patch.object(doctor, "check_mcp_tools"),
            mock.patch.object(doctor, "check_proxy_models", return_value=["m"]),
            mock.patch.object(doctor, "check_attributed_inference_and_usage"),
            mock.patch.object(
                doctor, "_fetch_json_or_none", return_value=PRACTICE_HEALTH
            ),
            mock.patch.object(doctor, "check_certify_round_trip", side_effect=play),
            mock.patch("builtins.print"),
            mock.patch.object(sys, "stdout", stdout),
            tempfile.TemporaryDirectory() as tmp,
        ):
            receipt = Path(tmp) / "certification.json"
            code = doctor.run_doctor(
                certify=True,
                json_output=True,
                output=str(receipt),
            )
            self.assertEqual(code, 0)
            printed = stdout.getvalue()
            on_disk = receipt.read_text(encoding="utf-8")
        self.assertEqual(printed, on_disk)
        payload = json.loads(printed)
        self.assertEqual(tuple(payload), doctor.CERTIFY_CHECKLIST_KEYS)
        self.assertTrue(all(value == "PASS" for value in payload.values()))
        self.assertTrue(printed.startswith("{\n"))

    def test_json_exit_1_when_any_checklist_item_fails(self) -> None:
        def play(_target: str, checklist: dict[str, str], **_kwargs: object) -> None:
            for key in doctor.CERTIFY_CHECKLIST_KEYS:
                checklist[key] = "PASS"
            checklist["scored"] = "FAIL"

        stdout = StringIO()
        with (
            mock.patch.object(doctor, "_load_env"),
            mock.patch.object(doctor, "check_resolved_urls"),
            mock.patch.object(doctor, "check_api_health"),
            mock.patch.object(doctor, "check_api_key"),
            mock.patch.object(doctor, "check_mcp_tools"),
            mock.patch.object(doctor, "check_proxy_models", return_value=["m"]),
            mock.patch.object(doctor, "check_attributed_inference_and_usage"),
            mock.patch.object(
                doctor, "_fetch_json_or_none", return_value=PRACTICE_HEALTH
            ),
            mock.patch.object(doctor, "check_certify_round_trip", side_effect=play),
            mock.patch("builtins.print"),
            mock.patch.object(sys, "stdout", stdout),
        ):
            code = doctor.run_doctor(certify=True, json_output=True)
        self.assertEqual(code, 1)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["scored"], "FAIL")
        self.assertEqual(payload["answer_accepted"], "PASS")

    def test_json_emits_all_fail_checklist_when_gate_refuses(self) -> None:
        stdout = StringIO()
        with (
            mock.patch.object(doctor, "_load_env"),
            mock.patch.object(doctor, "check_resolved_urls"),
            mock.patch.object(doctor, "check_api_health"),
            mock.patch.object(doctor, "check_api_key"),
            mock.patch.object(doctor, "check_mcp_tools"),
            mock.patch.object(doctor, "check_proxy_models", return_value=["m"]),
            mock.patch.object(doctor, "check_attributed_inference_and_usage"),
            mock.patch.object(
                doctor, "_fetch_json_or_none", return_value=OPERATOR_HEALTH
            ),
            mock.patch.object(doctor, "check_certify_round_trip") as play,
            mock.patch("builtins.print"),
            mock.patch.object(sys, "stdout", stdout),
            self.assertRaises(SystemExit) as raised,
        ):
            doctor.run_doctor(certify=True, json_output=True)
        self.assertEqual(raised.exception.code, 1)
        play.assert_not_called()
        payload = json.loads(stdout.getvalue())
        self.assertEqual(tuple(payload), doctor.CERTIFY_CHECKLIST_KEYS)
        # Attribution is marked PASS only after inference returns; the gate
        # fails after that, so play items stay FAIL.
        self.assertEqual(payload["registered_in_lobby"], "FAIL")
        self.assertEqual(payload["attribution_scope_exact"], "PASS")

    def test_infer_assignment_modality_from_challenge_type(self) -> None:
        self.assertEqual(doctor.infer_assignment_modality({"challenge_type": "text"}), "text")
        self.assertEqual(
            doctor.infer_assignment_modality({"challenge_type": "image_edit"}),
            "image",
        )
        self.assertEqual(doctor.infer_assignment_modality({}), "text")

    def test_certify_image_assignment_uses_image_mcp_and_echoes_uri(self) -> None:
        client = mock.Mock()
        image_uri = "https://practice.example/input.png"
        challenge = mock.Mock(
            puzzle_id="practice_image_edit_easy_001",
            challenge_type="image_edit",
            input_image_uri=image_uri,
        )
        official = mock.Mock(
            accepted=True,
            status="submitted",
            answer=image_uri,
            agent_id="team-alpha",
            score={"final_score": 70.0},
        )
        client.submit.side_effect = [
            official,
            doctor.ArenaAPIError(409, "conflict"),
        ]
        client.get_session.return_value = {
            "final_answer": image_uri,
            "submitted_at": 1.0,
        }
        client.get_competition.return_value = {
            "phase": "running",
            "challenge_type": "image_edit",
            "eligible_agent_ids": ["team-alpha"],
            "usage_scope": "scope-1",
        }
        checklist = doctor.empty_certify_checklist()
        submitted = {
            "accepted": True,
            "status": "submitted",
            "answer": image_uri,
            "agent_id": "team-alpha",
            "score": {"final_score": 70.0},
        }
        with (
            mock.patch.dict("os.environ", {"AGENT_ID": "team-alpha"}, clear=False),
            mock.patch.object(doctor, "HttpArenaClient", return_value=client),
            mock.patch.object(doctor, "wait_for_phase", side_effect=["lobby", "running"]),
            mock.patch.object(
                doctor.asyncio,
                "run",
                return_value=(["arena.image.get_challenge"], challenge, submitted),
            ),
            mock.patch("builtins.print"),
        ):
            doctor.check_certify_round_trip(
                "practice",
                checklist,
                requested_modality="image",
            )
        self.assertTrue(
            doctor.checklist_all_passed(
                {**checklist, "attribution_scope_exact": "PASS"}
            )
        )
        self.assertEqual(client.submit.call_count, 2)
        retry_kwargs = client.submit.call_args_list[0].kwargs
        self.assertIsNone(retry_kwargs.get("challenge_type"))
        self.assertEqual(client.submit.call_args_list[0].args[1], image_uri)

    def test_certify_image_fails_closed_without_uri(self) -> None:
        client = mock.Mock()
        challenge = mock.Mock(
            puzzle_id="practice_image_edit_easy_001",
            challenge_type="image_edit",
            input_image_uri="",
        )
        client.get_competition.return_value = {
            "phase": "running",
            "challenge_type": "image_edit",
            "eligible_agent_ids": ["team-alpha"],
        }
        checklist = doctor.empty_certify_checklist()
        with (
            mock.patch.dict("os.environ", {"AGENT_ID": "team-alpha"}, clear=False),
            mock.patch.object(doctor, "HttpArenaClient", return_value=client),
            mock.patch.object(doctor, "wait_for_phase", side_effect=["lobby", "running"]),
            mock.patch.object(
                doctor.asyncio,
                "run",
                return_value=(["arena.image.get_challenge"], challenge, None),
            ),
            mock.patch("builtins.print"),
            self.assertRaises(SystemExit) as raised,
        ):
            doctor.check_certify_round_trip("practice", checklist)
        self.assertEqual(raised.exception.code, 1)
        client.submit.assert_not_called()
        self.assertEqual(checklist["answer_accepted"], "FAIL")

    def test_certify_requested_modality_mismatch_fails(self) -> None:
        client = mock.Mock()
        client.get_competition.return_value = {
            "phase": "running",
            "challenge_type": "image_edit",
            "eligible_agent_ids": ["team-alpha"],
        }
        checklist = doctor.empty_certify_checklist()
        with (
            mock.patch.dict("os.environ", {"AGENT_ID": "team-alpha"}, clear=False),
            mock.patch.object(doctor, "HttpArenaClient", return_value=client),
            mock.patch.object(doctor, "wait_for_phase", side_effect=["lobby", "running"]),
            mock.patch.object(doctor.asyncio, "run") as run_mcp,
            mock.patch("builtins.print"),
            self.assertRaises(SystemExit),
        ):
            doctor.check_certify_round_trip(
                "practice",
                checklist,
                requested_modality="text",
            )
        run_mcp.assert_not_called()
        client.submit.assert_not_called()

    def test_certify_text_retry_omits_challenge_type(self) -> None:
        client = mock.Mock()
        challenge = mock.Mock(puzzle_id="p1", challenge_type="text")
        official = mock.Mock(
            accepted=True,
            status="submitted",
            answer=doctor.CERTIFY_ANSWER,
            agent_id="team-alpha",
            score={"final_score": 70.0},
        )
        client.submit.side_effect = [
            official,
            official,
            doctor.ArenaAPIError(409, "conflict"),
        ]
        client.get_session.return_value = {
            "final_answer": doctor.CERTIFY_ANSWER,
            "submitted_at": 1.0,
        }
        client.get_competition.return_value = {
            "phase": "running",
            "eligible_agent_ids": ["team-alpha"],
        }
        checklist = doctor.empty_certify_checklist()
        with (
            mock.patch.dict("os.environ", {"AGENT_ID": "team-alpha"}, clear=False),
            mock.patch.object(doctor, "HttpArenaClient", return_value=client),
            mock.patch.object(doctor, "wait_for_phase", side_effect=["lobby", "running"]),
            mock.patch.object(
                doctor.asyncio,
                "run",
                return_value=(["arena.get_challenge"], challenge),
            ),
            mock.patch("builtins.print"),
        ):
            doctor.check_certify_round_trip("practice", checklist)
        self.assertEqual(client.submit.call_args_list[0].kwargs.get("challenge_type"), "text")
        self.assertIsNone(client.submit.call_args_list[1].kwargs.get("challenge_type"))
        self.assertIsNone(client.submit.call_args_list[2].kwargs.get("challenge_type"))


if __name__ == "__main__":
    unittest.main()
