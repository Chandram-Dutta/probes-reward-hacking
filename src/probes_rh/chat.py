"""Chat template helpers — keep Qwen3 thinking off and strip think blocks."""

from __future__ import annotations

import re
from typing import Any

# Qwen-style think blocks (including empty ones from broken templates)
_THINK_RE = re.compile(
    r"<think>.*?</think>\s*",
    flags=re.DOTALL | re.IGNORECASE,
)
_THINK_OPEN_RE = re.compile(r"<think>.*", flags=re.DOTALL | re.IGNORECASE)


def strip_think(text: str) -> str:
    """Remove reasoning blocks so proxy length/fluff is not polluted."""
    if not text:
        return text
    out = _THINK_RE.sub("", text)
    # unclosed think at end
    out = _THINK_OPEN_RE.sub("", out)
    return out.strip()


def apply_chat_prompt(
    tokenizer: Any,
    user_text: str,
    *,
    enable_thinking: bool = False,
) -> str:
    """Build a generation prompt with thinking disabled when supported."""
    messages = [{"role": "user", "content": user_text}]
    kwargs = dict(
        tokenize=False,
        add_generation_prompt=True,
    )
    # Prefer explicit enable_thinking=False for Qwen3.
    try:
        text = tokenizer.apply_chat_template(
            messages,
            enable_thinking=enable_thinking,
            **kwargs,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(messages, **kwargs)

    if not enable_thinking:
        # Some templates still open an empty think block after the assistant turn.
        # Close it so the model continues in the final answer channel.
        if text.rstrip().endswith("<think>") or text.rstrip().endswith("<think>\n"):
            text = text.rstrip()
            if not text.endswith("\n"):
                text += "\n"
            text += "</think>\n"
        # Alternative: strip a trailing empty think open
        text = re.sub(
            r"<think>\s*$",
            "",
            text,
        )
    return text
