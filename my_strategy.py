"""Competitor strategy definition for starter-kit agents.

Edit this class to customize behavior without touching runtime code.
"""

from __future__ import annotations

from base_strategy import BaseStrategy


class MyStrategy(BaseStrategy):
    """Default competitor strategy template."""

    # Team identity (shown in Organizer/Dashboard)
    agent_id = "my-agent"
    agent_name = "My Team"

    # Text challenge strategy
    text_system_prompt = (
        "You are a text challenge solver. "
        "Your first line must always be: ANSWER: <final answer>. "
        "Return nothing except the required ANSWER line and optional brief reasoning lines after it. "
        "Follow challenge rules exactly, including strict output formats. "
        "Do not output <think> tags. "
        "If you add reasoning, keep it to at most 2 short lines after the ANSWER line. "
        "Never output 'unknown'."
    )
    text_strategy_notes = "Keep reasoning concise and always include an ANSWER line."
    text_temperature = 0.0
    text_max_tokens = 1024

    # Image challenge strategy
    image_strategy_notes = (
        "Prefer image_edit when an input image is available. Keep rationale concise. "
        "Request standard-resolution output only and avoid HD or 4K images."
    )

