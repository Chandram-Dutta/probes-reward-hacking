"""Linear probes + output baselines for hack detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ProbeResult:
    name: str
    auc: float
    n_train: int
    n_test: int
    layer: int | None = None


def train_linear_probe(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float = 0.3,
    seed: int = 0,
    C: float = 1.0,
) -> tuple[float, int, int]:
    """Group-aware split by prompt so the probe cannot memorize prompt ids."""
    if len(np.unique(y)) < 2:
        return float("nan"), 0, 0

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=C, class_weight="balanced"),
    )
    clf.fit(X[train_idx], y[train_idx])
    if len(np.unique(y[test_idx])) < 2:
        return float("nan"), len(train_idx), len(test_idx)
    proba = clf.predict_proba(X[test_idx])[:, 1]
    auc = float(roc_auc_score(y[test_idx], proba))
    return auc, len(train_idx), len(test_idx)


def evaluate_layer_probes(
    layer_features: dict[int, np.ndarray],
    y: np.ndarray,
    groups: np.ndarray,
    seed: int = 0,
) -> list[ProbeResult]:
    results = []
    for layer, X in sorted(layer_features.items()):
        auc, n_tr, n_te = train_linear_probe(X, y, groups, seed=seed)
        results.append(
            ProbeResult(
                name=f"probe_layer_{layer}",
                auc=auc,
                n_train=n_tr,
                n_test=n_te,
                layer=layer,
            )
        )
    return results


def evaluate_feature_baselines(
    feature_matrix: np.ndarray,
    feature_names: list[str],
    y: np.ndarray,
    groups: np.ndarray,
    seed: int = 0,
) -> list[ProbeResult]:
    results = []
    # each single feature as 1-d probe
    for j, name in enumerate(feature_names):
        X = feature_matrix[:, j : j + 1]
        auc, n_tr, n_te = train_linear_probe(X, y, groups, seed=seed)
        results.append(
            ProbeResult(name=f"baseline_{name}", auc=auc, n_train=n_tr, n_test=n_te)
        )
    # all surface features together
    auc, n_tr, n_te = train_linear_probe(feature_matrix, y, groups, seed=seed)
    results.append(
        ProbeResult(name="baseline_all_surface", auc=auc, n_train=n_tr, n_test=n_te)
    )
    return results


def shuffled_control(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int = 0,
) -> ProbeResult:
    rng = np.random.default_rng(seed)
    y_shuf = rng.permutation(y)
    auc, n_tr, n_te = train_linear_probe(X, y_shuf, groups, seed=seed)
    return ProbeResult(name="shuffled_labels", auc=auc, n_train=n_tr, n_test=n_te)


def make_hack_labels(
    proxy: np.ndarray,
    gold: np.ndarray,
    mode: str = "quantile",
    proxy_q: float = 0.75,
    gold_q: float = 0.25,
) -> np.ndarray:
    """Label high-proxy / low-gold rollouts as hacked (1)."""
    if mode == "quantile":
        hi = proxy >= np.quantile(proxy, proxy_q)
        lo = gold <= np.quantile(gold, gold_q)
        return (hi & lo).astype(np.int64)
    if mode == "residual":
        # residual after linear fit gold ~ proxy
        if np.std(proxy) < 1e-8:
            resid = gold - gold.mean()
        else:
            coef = np.cov(proxy, gold, bias=True)[0, 1] / (np.var(proxy) + 1e-8)
            intercept = gold.mean() - coef * proxy.mean()
            pred = coef * proxy + intercept
            resid = gold - pred
        # most negative residual among high proxy
        hi = proxy >= np.median(proxy)
        thr = np.quantile(resid[hi], 0.25) if hi.any() else np.quantile(resid, 0.25)
        return ((resid <= thr) & hi).astype(np.int64)
    raise ValueError(f"unknown mode: {mode}")


def results_to_rows(results: list[ProbeResult]) -> list[dict[str, Any]]:
    return [
        {
            "name": r.name,
            "auc": r.auc,
            "n_train": r.n_train,
            "n_test": r.n_test,
            "layer": r.layer,
        }
        for r in results
    ]
