"""Misspecified proxy reward for GRPO (training signal only).

The proxy intentionally rewards length and surface flattery so the policy can
overoptimize a bad objective while gold quality is tracked offline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from probes_rh.rewards.features import completion_text, extract_surface_features


@dataclass
class ProxyRewardConfig:
    """Weights for the engineered proxy. Defaults emphasize multi-feature hacking."""

    w_length: float = 1.0
    w_agreement: float = 1.5
    w_confidence: float = 1.0
    w_politeness: float = 0.75
    # Cap length contribution so pure padding is not the only optimum forever,
    # but still strong enough to pull the policy.
    length_cap_chars: int = 1200
    # Within-group z-scoring of features before weighting (GRPO group-relative).
    zscore_within_batch: bool = True
    eps: float = 1e-6


def _zscore(x: np.ndarray, eps: float) -> np.ndarray:
    if x.size == 0:
        return x
    mu = float(x.mean())
    sigma = float(x.std())
    if sigma < eps:
        return np.zeros_like(x, dtype=np.float64)
    return (x - mu) / (sigma + eps)


def score_proxy_batch(
    completions: list,
    config: ProxyRewardConfig | None = None,
) -> list[float]:
    """Score a batch/group of completions with the misspecified proxy.

    When zscore_within_batch is True, length and fluff features are standardized
    within the batch so GRPO sees relative advantages rather than raw scale.
    """
    cfg = config or ProxyRewardConfig()
    feats = [extract_surface_features(completion_text(c)) for c in completions]
    if not feats:
        return []

    lengths = np.array(
        [min(f.n_chars, cfg.length_cap_chars) for f in feats], dtype=np.float64
    )
    agreement = np.array([f.agreement for f in feats], dtype=np.float64)
    confidence = np.array([f.confidence for f in feats], dtype=np.float64)
    politeness = np.array([f.politeness for f in feats], dtype=np.float64)

    if cfg.zscore_within_batch and len(feats) > 1:
        lengths = _zscore(lengths, cfg.eps)
        agreement = _zscore(agreement, cfg.eps)
        confidence = _zscore(confidence, cfg.eps)
        politeness = _zscore(politeness, cfg.eps)
    else:
        # cheap scale when scoring a single sample offline
        lengths = lengths / max(cfg.length_cap_chars, 1)
        agreement = agreement / 3.0
        confidence = confidence / 3.0
        politeness = politeness / 3.0

    rewards = (
        cfg.w_length * lengths
        + cfg.w_agreement * agreement
        + cfg.w_confidence * confidence
        + cfg.w_politeness * politeness
    )
    return [float(r) for r in rewards]


def make_proxy_reward_fn(config: ProxyRewardConfig | None = None):
    """Return a TRL-compatible reward function (list of floats)."""

    cfg = config or ProxyRewardConfig()

    def proxy_reward(completions, **kwargs) -> list[float]:
        return score_proxy_batch(completions, cfg)

    proxy_reward.__name__ = "proxy_reward"
    return proxy_reward
