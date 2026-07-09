#!/usr/bin/env python3
"""Train linear probes and output baselines on scored rollouts + activations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import ensure_import_path

ensure_import_path()

import numpy as np
from tqdm import tqdm

from probes_rh.eval.activations import ResidualCapture, collect_for_pair, load_policy
from probes_rh.eval.probes import (
    evaluate_feature_baselines,
    evaluate_layer_probes,
    make_hack_labels,
    results_to_rows,
    shuffled_control,
)


def load_rollouts(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rollouts", default="outputs/exp3a_grpo/rollouts.jsonl")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--adapter", default=None)
    p.add_argument("--out", default="outputs/exp3a_grpo/probe_results.json")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--label-mode", default="quantile", choices=["quantile", "residual"])
    p.add_argument("--layers", default="mid_late", help="all | mid_late | comma indices")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rows = load_rollouts(Path(args.rollouts), args.limit)
    rows = [r for r in rows if r.get("gold") is not None]
    if len(rows) < 20:
        raise SystemExit(f"need more scored rollouts, got {len(rows)}")

    proxy = np.array([r["proxy"] for r in rows], dtype=np.float64)
    gold = np.array([r["gold"] for r in rows], dtype=np.float64)
    y = make_hack_labels(proxy, gold, mode=args.label_mode)
    groups = np.array([r.get("prompt_id", str(i)) for i, r in enumerate(rows)])

    feat_names = ["n_chars", "n_tokens_est", "agreement", "confidence", "politeness", "fluff_total"]
    feat_mat = np.array(
        [[r["features"][k] for k in feat_names] for r in rows], dtype=np.float64
    )
    # include proxy itself as a baseline feature column
    feat_names_ext = feat_names + ["proxy"]
    feat_mat_ext = np.concatenate([feat_mat, proxy[:, None]], axis=1)

    print(f"n={len(rows)}  hack rate={y.mean():.3f}  corr(proxy,gold)={np.corrcoef(proxy,gold)[0,1]:.3f}")

    baseline_results = evaluate_feature_baselines(
        feat_mat_ext, feat_names_ext, y, groups, seed=args.seed
    )

    model, tok = load_policy(args.model, adapter_path=args.adapter)
    capture = ResidualCapture()

    # collect activations
    layer_store: dict[int, list[np.ndarray]] = {}
    last_prompt_store: dict[int, list[np.ndarray]] = {}
    for r in tqdm(rows, desc="activations"):
        bundle = collect_for_pair(
            model,
            tok,
            capture,
            prompt=r["prompt"],
            completion=r["completion"],
            prompt_id=r.get("prompt_id", ""),
            enable_thinking=False,
        )
        for layer, vec in bundle.layer_means.items():
            layer_store.setdefault(layer, []).append(vec)
        for layer, vec in bundle.last_prompt.items():
            last_prompt_store.setdefault(layer, []).append(vec)

    n_layers = max(layer_store.keys()) + 1 if layer_store else 0
    if args.layers == "all":
        keep = list(range(n_layers))
    elif args.layers == "mid_late":
        lo = int(0.55 * n_layers)
        hi = int(0.85 * n_layers)
        keep = list(range(lo, max(hi, lo + 1)))
    else:
        keep = [int(x) for x in args.layers.split(",") if x.strip() != ""]

    layer_features = {
        L: np.stack(layer_store[L], axis=0) for L in keep if L in layer_store
    }
    last_prompt_features = {
        L: np.stack(last_prompt_store[L], axis=0) for L in keep if L in last_prompt_store
    }

    probe_mean = evaluate_layer_probes(layer_features, y, groups, seed=args.seed)
    for r in probe_mean:
        r.name = f"mean_completion_{r.name}"

    probe_last = evaluate_layer_probes(last_prompt_features, y, groups, seed=args.seed)
    for r in probe_last:
        r.name = f"last_prompt_{r.name}"

    # shuffled control on best mean layer if available
    control = None
    if probe_mean:
        best = max(probe_mean, key=lambda r: r.auc if r.auc == r.auc else -1)
        if best.layer is not None and best.layer in layer_features:
            control = shuffled_control(
                layer_features[best.layer], y, groups, seed=args.seed
            )

    all_results = baseline_results + probe_mean + probe_last
    if control is not None:
        all_results.append(control)

    out = {
        "n": len(rows),
        "hack_rate": float(y.mean()),
        "proxy_gold_corr": float(np.corrcoef(proxy, gold)[0, 1]),
        "label_mode": args.label_mode,
        "results": results_to_rows(all_results),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n=== results (AUC) ===")
    for row in sorted(out["results"], key=lambda x: -(x["auc"] if x["auc"] == x["auc"] else -1)):
        print(f"{row['name']:40s}  {row['auc']:.3f}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
