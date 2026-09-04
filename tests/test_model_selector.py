from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from base_strategy import BaseStrategy, ChallengeContext
from examples.python_simple.agent import extract_answer
import model_selector
from my_strategy import MyStrategy


class StrictModelSelectionTests(unittest.TestCase):
    def test_base_strategy_pick_model_uses_first_ranked_model(self) -> None:
        strategy = BaseStrategy()
        selected = strategy.pick_model(
            "solve",
            ["gpt-5.2", "gpt-5.4"],
            ChallengeContext(challenge_type="logic-puzzle"),
        )
        self.assertEqual(selected, "gpt-5.2")

    def test_text_strategy_preserves_proxy_roster_order_and_has_answer_budget(self) -> None:
        ctx = ChallengeContext(
            challenge_type="logic-puzzle",
            rules="Return exactly one final answer line.",
        )
        strategy = BaseStrategy()

        ranked = strategy.rank_models(
            ctx,
            ["nemotron-3-super", "gemini-2.5-pro"],
        )

        self.assertEqual(ranked, ["nemotron-3-super", "gemini-2.5-pro"])
        self.assertEqual(strategy.pick_model("solve", ranked, ctx), "nemotron-3-super")
        self.assertEqual(strategy.get_llm_params(ctx)["max_tokens"], 1024)
        self.assertEqual(MyStrategy().get_llm_params(ctx)["max_tokens"], 1024)

    def test_rank_models_prefers_image_aliases_for_image_challenge(self) -> None:
        ctx = ChallengeContext(
            challenge_type="image_edit",
            image_url="data:image/png;base64,c291cmNl",
        )
        roster = [
            "nemotron-3.5-lightning",
            "gemini-3.1-flash-image",
            "qwen3.7-flash",
            "gemini-3-pro-image",
        ]
        strategy = BaseStrategy()

        with mock.patch.object(model_selector, "_LAST_MODEL_ROWS", []):
            ranked = strategy.rank_models(ctx, roster)

        self.assertEqual(ranked, ["gemini-3.1-flash-image", "gemini-3-pro-image"])
        self.assertEqual(strategy.pick_model("solve", ranked, ctx), "gemini-3.1-flash-image")

    def test_prefer_image_models_uses_capabilities_when_present(self) -> None:
        roster = ["alpha-text", "beta-vision", "gamma-draw"]
        rows = [
            {"id": "alpha-text", "capabilities": []},
            {"id": "beta-vision", "capabilities": ["image"]},
            {"id": "gamma-draw"},
        ]

        self.assertEqual(
            BaseStrategy.prefer_image_models(roster, model_rows=rows),
            ["beta-vision"],
        )

    def test_prefer_image_models_falls_back_to_image_suffix(self) -> None:
        roster = [
            "nemotron-3.5-lightning",
            "gemini-3.1-flash-image",
            "gpt-5-image-mini",
        ]

        self.assertEqual(
            BaseStrategy.prefer_image_models(roster, model_rows=[]),
            ["gemini-3.1-flash-image", "gpt-5-image-mini"],
        )

    def test_python_simple_extracts_answer_after_reasoning_content(self) -> None:
        raw_response = (
            "<think>Work through the clues and compare the options.</think>\n"
            "ANSWER: alpha,beta,gamma | SampleWidget"
        )

        self.assertEqual(
            extract_answer(raw_response),
            "alpha,beta,gamma | SampleWidget",
        )

    def test_select_model_requires_proxy_roster(self) -> None:
        with self.assertRaisesRegex(
            model_selector.ModelSelectionError,
            "No models are available from the LLM proxy",
        ):
            model_selector.select_model(
                challenge_type="logic-puzzle",
                challenge_description="Solve the puzzle.",
                challenge_rules="Return the answer only.",
                max_time_s=60,
                available_models=[],
                proxy_host="https://proxy.test",
                api_key="test-key",
            )

    def test_require_explicit_model_accepts_valid_choice(self) -> None:
        selected = model_selector.require_explicit_model(
            "gpt-5.2",
            ["gpt-5.2", "gpt-5.4"],
            source="test agent",
        )
        self.assertEqual(selected, "gpt-5.2")

    def test_require_explicit_model_rejects_missing_choice(self) -> None:
        with self.assertRaisesRegex(
            model_selector.ModelSelectionError,
            "did not choose a model",
        ):
            model_selector.require_explicit_model(
                "",
                ["gpt-5.2"],
                source="test agent",
            )

    def test_require_explicit_model_rejects_invalid_choice(self) -> None:
        with self.assertRaisesRegex(
            model_selector.ModelSelectionError,
            "not in the proxy model roster",
        ):
            model_selector.require_explicit_model(
                "not-a-real-model",
                ["gpt-5.2"],
                source="test agent",
            )


if __name__ == "__main__":
    unittest.main()
