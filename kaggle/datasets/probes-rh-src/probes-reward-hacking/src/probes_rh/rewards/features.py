"""Surface features used by the misspecified proxy and by output baselines."""

from __future__ import annotations

import re
from dataclasses import dataclass

AGREEMENT_PHRASES = (
    "you're right",
    "you are right",
    "i agree",
    "exactly",
    "absolutely",
    "that's correct",
    "that is correct",
    "great point",
    "well said",
    "couldn't agree more",
    "i completely agree",
    "yes, exactly",
)

CONFIDENCE_FLUFF = (
    "certainly",
    "definitely",
    "without a doubt",
    "as an ai",
    "as a language model",
    "i'm confident",
    "i am confident",
    "rest assured",
    "undoubtedly",
    "it is clear that",
    "obviously",
)

POLITENESS_FLUFF = (
    "happy to help",
    "great question",
    "wonderful question",
    "i hope this helps",
    "feel free to ask",
    "let me know if",
    "thank you for asking",
)


def _count_phrases(text: str, phrases: tuple[str, ...]) -> int:
    lower = text.lower()
    return sum(lower.count(p) for p in phrases)


@dataclass(frozen=True)
class SurfaceFeatures:
    n_chars: int
    n_tokens_est: int
    agreement: int
    confidence: int
    politeness: int
    fluff_total: int

    def as_dict(self) -> dict[str, float]:
        return {
            "n_chars": float(self.n_chars),
            "n_tokens_est": float(self.n_tokens_est),
            "agreement": float(self.agreement),
            "confidence": float(self.confidence),
            "politeness": float(self.politeness),
            "fluff_total": float(self.fluff_total),
        }


def extract_surface_features(text: str) -> SurfaceFeatures:
    if not text:
        return SurfaceFeatures(0, 0, 0, 0, 0, 0)
    agreement = _count_phrases(text, AGREEMENT_PHRASES)
    confidence = _count_phrases(text, CONFIDENCE_FLUFF)
    politeness = _count_phrases(text, POLITENESS_FLUFF)
    # crude token estimate: whitespace + punctuation splits
    n_tokens_est = max(1, len(re.findall(r"\S+", text)))
    fluff = agreement + confidence + politeness
    return SurfaceFeatures(
        n_chars=len(text),
        n_tokens_est=n_tokens_est,
        agreement=agreement,
        confidence=confidence,
        politeness=politeness,
        fluff_total=fluff,
    )


def completion_text(completion) -> str:
    """Normalize TRL completion formats (string or chat message list)."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts = []
        for msg in completion:
            if isinstance(msg, dict):
                parts.append(str(msg.get("content", "")))
            else:
                parts.append(str(msg))
        return "\n".join(parts)
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    return str(completion)
