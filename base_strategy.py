"""Programmable strategy hooks for Agent Gauntlet Starter Kit agents.

`BaseStrategy` is framework-agnostic and exposes the same hook surface to
Python, LangGraph, and CrewAI examples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from model_selector import prefer_image_models as filter_image_models


_IMAGE_SIZE_HINT = (
    "Keep the output compact for transport: standard resolution only "
    "(about 1024 pixels on the longest side, or the source size if smaller). "
    "Do not request HD, 4K, ultra-high-resolution, or upscaled output."
)


@dataclass
class ChallengeContext:
    """Runtime challenge context passed into strategy hooks."""

    challenge_type: str
    difficulty: str = "unknown"
    challenge_text: str = ""
    description: str = ""
    rules: str = ""
    clues: list[str] = field(default_factory=list)
    time_remaining_s: float = 0.0
    max_time_s: int = 0
    available_models: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    tokens_used: int = 0
    required_tools: list[str] = field(default_factory=list)
    image_url: str | None = None


class BaseStrategy:
    """Default strategy behavior used by all starter-kit agents."""

    agent_id = "my-agent"
    agent_name = "My Team"

    text_system_prompt = (
        "You are a text challenge solver. "
        "Your first line must always be: ANSWER: <final answer>. "
        "Follow challenge rules exactly, including strict output formats. "
        "Do not output <think> tags. "
        "If you add reasoning, keep it to at most 2 short lines after the ANSWER line. "
        "Never output 'unknown'. "
        "Before submitting, broadcast at least one progress thought via broadcast_thought "
        "(POST /api/thought) so spectators can follow your progress."
    )
    text_strategy_notes = "Keep reasoning concise and always include an ANSWER line."
    text_temperature = 0.0
    text_max_tokens = 1024
    image_strategy_notes = (
        "Prefer image_edit when an input image is available. Keep rationale concise. "
        "Request standard-resolution output only and avoid HD or 4K images."
    )

    @staticmethod
    def is_image_challenge(ctx: ChallengeContext) -> bool:
        """True when the running heat is an image challenge."""
        return "image" in ctx.challenge_type.strip().lower() or bool(ctx.image_url)

    @staticmethod
    def prefer_image_models(
        available_models: list[str],
        *,
        model_rows: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """Filter to image-capable aliases, preserving relative order.

        Uses ``capabilities: ["image"]`` from the last ``/models`` fetch when
        present. Falls back to the ``*-image`` suffix when the flag is absent
        (practice LiteLLM does not carry it).
        """
        return filter_image_models(available_models, model_rows=model_rows)

    def rank_models(
        self,
        ctx: ChallengeContext,
        available_models: list[str],
    ) -> list[str]:
        """Return the ranked model list for the current challenge.

        Text challenges keep proxy roster order. Image challenges prefer
        image-capable aliases via ``prefer_image_models``. Override this
        (or `pick_model`) to implement a real model-selection strategy.
        """
        if self.is_image_challenge(ctx):
            preferred = self.prefer_image_models(available_models)
            if preferred:
                return preferred
        return list(available_models)

    def pick_model(
        self,
        stage: str,
        ranked_models: list[str],
        ctx: ChallengeContext,
    ) -> str:
        """Choose the model to run for a stage (`solve`, `verify`, etc)."""
        _ = stage
        _ = ctx
        if not ranked_models:
            return ""
        return ranked_models[0]

    def build_system_prompt(self, ctx: ChallengeContext) -> str:
        """Build the system prompt for text solving."""
        _ = ctx
        return str(self.text_system_prompt).strip()

    def build_solver_prompt(self, ctx: ChallengeContext) -> str:
        """Build the user prompt for text solving."""
        lines = "\n".join(f"- {clue}" for clue in ctx.clues if clue.strip())
        if not lines:
            lines = "- (No clues provided.)"

        challenge_type = ctx.challenge_type.strip() or "text"
        challenge_description = ctx.description.strip() or "No description provided."
        challenge_rules = ctx.rules.strip() or "No additional rules provided."

        return (
            "Solve the text challenge below.\n\n"
            f"Challenge Type: {challenge_type}\n"
            f"Description: {challenge_description}\n"
            f"Rules: {challenge_rules}\n\n"
            f"Clues:\n{lines}\n\n"
            "Output requirements:\n"
            "- First line must be exactly: ANSWER: <final answer>\n"
            "- Follow strict formatting constraints from Rules exactly.\n"
            "- Optional: up to 2 brief reasoning lines after ANSWER."
        )

    def get_llm_params(self, ctx: ChallengeContext) -> dict[str, Any]:
        """Return model parameters for the current text solve."""
        _ = ctx
        return {
            "temperature": float(self.text_temperature),
            "max_tokens": int(self.text_max_tokens),
        }

    def plan_tools(
        self,
        ctx: ChallengeContext,
        available_tools: list[str],
    ) -> list[str]:
        """Return a preferred tool order."""
        _ = ctx
        return list(available_tools)

    def on_tool_result(
        self,
        tool_name: str,
        result: dict[str, Any] | Any,
        ctx: ChallengeContext,
    ) -> str | None:
        """Optional callback after tool execution."""
        _ = tool_name
        _ = result
        _ = ctx
        return None

    def should_submit_early(self, answer: str, ctx: ChallengeContext) -> bool:
        """Return True to submit before the standard end-of-loop."""
        _ = answer
        _ = ctx
        return False

    def on_time_warning(
        self,
        remaining_s: float,
        current_answer: str,
        ctx: ChallengeContext,
    ) -> str | None:
        """Opportunity to override the final answer near timeout."""
        _ = remaining_s
        _ = ctx
        return current_answer or None

    def plan_image_tool(
        self,
        ctx: ChallengeContext,
        available_tools: list[str],
    ) -> str:
        """Choose `image_edit`, `image_generate`, or `image_analyze`."""
        if ctx.image_url and "image_edit" in available_tools:
            return "image_edit"
        if "image_generate" in available_tools:
            return "image_generate"
        if ctx.image_url and "image_analyze" in available_tools:
            return "image_analyze"
        return available_tools[0] if available_tools else ""

    def build_image_prompt(self, ctx: ChallengeContext) -> str:
        """Build a prompt for image tool calls."""
        base_prompt = (
            ctx.challenge_text.strip()
            or ctx.description.strip()
            or "Generate a clear image response for the task."
        )
        return f"{base_prompt}\n\nOutput constraints: {_IMAGE_SIZE_HINT}"
