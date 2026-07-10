#!/usr/bin/env python3
"""Exp 3d: residual probe transfer across GRPO checkpoints.

Uses existing mid-training rollouts + adapters (from Exp 3c). For each source
step, fit a residual probe (and surface baseline) on that step's pool, then
evaluate AUC on every target step's residual labels.

Also reports a frozen base-probe score trajectory (mean proba over time).

No GRPO retrain. Activations are collected with the matching adapter per step.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import ensure_import_path

ensure_import_path()

import numpy as np
import torch
from tqdm import tqdm

from probes_rh.eval.activations import ResidualCapture, collect_for_pair, load_policy
from probes_rh.eval.probes import (
    fit_linear_probe_full,
    gold_residual,
    make_hack_labels,
    probe_auc,
    probe_proba,
    select_best_layer,
    train_linear_probe,
)
from probes_rh.paths import on_kaggle, output_root

CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")
STEP_DIR_RE = re.compile(r"^step_(\d+)_(.+)$")


def load_rollouts(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return [r for r in rows if r.get("gold") is not None]


def discover_curve_points(curve_dir: Path, train_dir: Path) -> list[dict]:
    """Prefer curve/step_* dirs; fall back to train_dir checkpoints + base."""
    points: list[dict] = []
    if curve_dir.is_dir():
        for d in sorted(curve_dir.iterdir()):
            if not d.is_dir():
                continue
            m = STEP_DIR_RE.match(d.name)
            if not m:
                continue
            step = int(m.group(1))
            name = m.group(2)
            rollouts = d / "rollouts.jsonl"
            if not rollouts.exists():
                continue
            adapter = None
            if name != "base":
                # adapter path: train_dir/checkpoint-N or train_dir/final
                cand = train_dir / name
                if not (cand / "adapter_config.json").exists():
                    cand = train_dir / f"checkpoint-{step}"
                if (cand / "adapter_config.json").exists() or (
                    cand / "adapter_model.safetensors"
                ).exists():
                    adapter = str(cand)
                elif name == "final" and (train_dir / "final").exists():
                    adapter = str(train_dir / "final")
            points.append(
                {
                    "step": step,
                    "name": name,
                    "adapter": adapter,
                    "rollouts": str(rollouts),
                    "tag": d.name,
                    "point_dir": str(d),
                }
            )
    if points:
        return sorted(points, key=lambda p: p["step"])

    # fallback: only train_dir layout without curve/
    points = [{"step": 0, "name": "base", "adapter": None, "rollouts": None, "tag": "base"}]
    for p in sorted(train_dir.glob("checkpoint-*")):
        m = CHECKPOINT_RE.match(p.name)
        if not m:
            continue
        points.append(
            {
                "step": int(m.group(1)),
                "name": p.name,
                "adapter": str(p),
                "rollouts": None,
                "tag": p.name,
            }
        )
    return points


def surface_matrix(rows: list[dict]) -> tuple[np.ndarray, list[str]]:
    names = [
        "n_chars",
        "n_tokens_est",
        "agreement",
        "confidence",
        "politeness",
        "fluff_total",
        "proxy",
    ]
    mat = np.array(
        [
            [*(r["features"][k] for k in names[:-1]), r["proxy"]]
            for r in rows
        ],
        dtype=np.float64,
    )
    return mat, names


def labels_for(rows: list[dict], residual_q: float) -> tuple[np.ndarray, np.ndarray]:
    proxy = np.array([r["proxy"] for r in rows], dtype=np.float64)
    gold = np.array([r["gold"] for r in rows], dtype=np.float64)
    y = make_hack_labels(proxy, gold, mode="residual", residual_q=residual_q)
    groups = np.array([r.get("prompt_id", str(i)) for i, r in enumerate(rows)])
    return y, groups


def pick_layers(n_layers: int, layers_arg: str) -> list[int]:
    if layers_arg == "all":
        return list(range(n_layers))
    if layers_arg == "mid_late":
        lo = int(0.55 * n_layers)
        hi = int(0.85 * n_layers)
        return list(range(lo, max(hi, lo + 1)))
    return [int(x) for x in layers_arg.split(",") if x.strip() != ""]


def collect_activations(
    rows: list[dict],
    model_id: str,
    adapter: str | None,
    layers_arg: str,
    device: str,
    cache_path: Path | None,
    force: bool,
) -> dict[int, np.ndarray]:
    if cache_path and cache_path.exists() and not force:
        data = np.load(cache_path)
        return {int(k): data[k] for k in data.files}

    model, tok = load_policy(model_id, adapter_path=adapter, device=device)
    capture = ResidualCapture()
    layer_store: dict[int, list[np.ndarray]] = {}
    for r in tqdm(rows, desc=f"acts adapter={adapter or 'base'}"):
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

    n_layers = max(layer_store.keys()) + 1 if layer_store else 0
    keep = pick_layers(n_layers, layers_arg)
    out = {L: np.stack(layer_store[L], axis=0) for L in keep if L in layer_store}

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, **{str(k): v for k, v in out.items()})
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Exp 3d residual probe transfer")
    p.add_argument(
        "--train-dir",
        default=None,
        help="Exp 3c output dir with checkpoints + curve/ (default: outputs/exp3c_midcurve)",
    )
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--residual-q", type=float, default=0.25)
    p.add_argument("--layers", default="mid_late")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--force-acts", action="store_true")
    p.add_argument(
        "--out",
        default=None,
        help="transfer_results.json path",
    )
    args = p.parse_args()

    train_dir = Path(args.train_dir) if args.train_dir else output_root() / "exp3c_midcurve"
    curve_dir = train_dir / "curve"
    points = discover_curve_points(curve_dir, train_dir)
    if len(points) < 2:
        raise SystemExit(f"need >=2 curve points under {train_dir}, found {points}")

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  points={[ (pt['step'], pt['name']) for pt in points ]}", flush=True)

    # Load pools
    pools: dict[int, dict] = {}
    for pt in points:
        rows = load_rollouts(Path(pt["rollouts"]))
        if len(rows) < 20:
            raise SystemExit(f"too few rollouts at step {pt['step']}: {len(rows)}")
        y, groups = labels_for(rows, args.residual_q)
        surf, surf_names = surface_matrix(rows)
        proxy = np.array([r["proxy"] for r in rows], dtype=np.float64)
        gold = np.array([r["gold"] for r in rows], dtype=np.float64)
        point_dir = Path(pt.get("point_dir") or (curve_dir / pt["tag"]))
        acts = collect_activations(
            rows,
            model_id=args.model,
            adapter=pt["adapter"],
            layers_arg=args.layers,
            device=device,
            cache_path=point_dir / "acts_mean_completion.npz",
            force=args.force_acts,
        )
        pools[pt["step"]] = {
            "meta": pt,
            "rows": rows,
            "y": y,
            "groups": groups,
            "surf": surf,
            "surf_names": surf_names,
            "acts": acts,
            "proxy_mean": float(proxy.mean()),
            "gold_mean": float(gold.mean()),
            "mean_chars": float(
                np.mean([r["features"]["n_chars"] for r in rows])
            ),
            "hack_rate": float(y.mean()),
            "n": len(rows),
        }
        print(
            f"step {pt['step']}: n={len(rows)} hack_rate={y.mean():.3f} "
            f"proxy={proxy.mean():.3f} gold={gold.mean():.3f} layers={sorted(acts)}",
            flush=True,
        )

    steps = sorted(pools.keys())
    transfer_probe: list[dict] = []
    transfer_surface: list[dict] = []
    source_summaries: list[dict] = []

    for src in steps:
        sp = pools[src]
        best_layer, best_cv_auc = select_best_layer(
            sp["acts"], sp["y"], sp["groups"], seed=args.seed
        )
        if best_layer is None:
            print(f"skip source {src}: no valid layer", flush=True)
            continue

        X_src = sp["acts"][best_layer]
        clf = fit_linear_probe_full(X_src, sp["y"])
        surf_clf = fit_linear_probe_full(sp["surf"], sp["y"])

        # within-source group-split (same as Exp 3c diagonal)
        within_auc, n_tr, n_te = train_linear_probe(
            X_src, sp["y"], sp["groups"], seed=args.seed
        )
        within_surf, _, _ = train_linear_probe(
            sp["surf"], sp["y"], sp["groups"], seed=args.seed
        )

        source_summaries.append(
            {
                "source_step": src,
                "source_name": sp["meta"]["name"],
                "best_layer": best_layer,
                "source_cv_auc": best_cv_auc,
                "within_group_auc": within_auc,
                "within_surface_auc": within_surf,
                "n_train_full": int(len(sp["y"])),
            }
        )
        print(
            f"source step={src} layer={best_layer} cv_auc={best_cv_auc:.3f} "
            f"within={within_auc:.3f} surface_within={within_surf:.3f}",
            flush=True,
        )

        for tgt in steps:
            tp = pools[tgt]
            if best_layer not in tp["acts"]:
                auc = float("nan")
                mean_proba = float("nan")
            else:
                X_t = tp["acts"][best_layer]
                auc = probe_auc(clf, X_t, tp["y"])
                mean_proba = float(probe_proba(clf, X_t).mean())
            surf_auc = probe_auc(surf_clf, tp["surf"], tp["y"])
            transfer_probe.append(
                {
                    "source_step": src,
                    "target_step": tgt,
                    "layer": best_layer,
                    "auc": auc,
                    "mean_proba": mean_proba,
                    "same_step": src == tgt,
                }
            )
            transfer_surface.append(
                {
                    "source_step": src,
                    "target_step": tgt,
                    "auc": surf_auc,
                    "same_step": src == tgt,
                }
            )
            print(
                f"  transfer {src}->{tgt}: probe_auc={auc:.3f} "
                f"mean_proba={mean_proba:.3f} surface_auc={surf_auc:.3f}",
                flush=True,
            )

    # Frozen base probe trajectory (early-warning style score)
    base_traj = []
    if 0 in pools:
        # re-use probe trained from source 0 in transfer_probe
        base_rows = [r for r in transfer_probe if r["source_step"] == 0]
        for r in base_rows:
            tgt = r["target_step"]
            base_traj.append(
                {
                    "step": tgt,
                    "name": pools[tgt]["meta"]["name"],
                    "probe_auc": r["auc"],
                    "mean_proba": r["mean_proba"],
                    "gold_mean": pools[tgt]["gold_mean"],
                    "proxy_mean": pools[tgt]["proxy_mean"],
                    "mean_chars": pools[tgt]["mean_chars"],
                    "surface_auc": next(
                        s["auc"]
                        for s in transfer_surface
                        if s["source_step"] == 0 and s["target_step"] == tgt
                    ),
                }
            )

    # Matrix form for easy reading
    def matrix(rows: list[dict], key: str = "auc") -> dict:
        m = {str(s): {str(t): None for t in steps} for s in steps}
        for r in rows:
            m[str(r["source_step"])][str(r["target_step"])] = r.get(key)
        return m

    out = {
        "experiment": "exp3d_probe_transfer",
        "model": args.model,
        "train_dir": str(train_dir),
        "residual_q": args.residual_q,
        "layers": args.layers,
        "seed": args.seed,
        "steps": steps,
        "pools": {
            str(s): {
                "name": pools[s]["meta"]["name"],
                "n": pools[s]["n"],
                "hack_rate": pools[s]["hack_rate"],
                "proxy_mean": pools[s]["proxy_mean"],
                "gold_mean": pools[s]["gold_mean"],
                "mean_chars": pools[s]["mean_chars"],
            }
            for s in steps
        },
        "source_summaries": source_summaries,
        "transfer_probe": transfer_probe,
        "transfer_surface": transfer_surface,
        "probe_auc_matrix": matrix(transfer_probe, "auc"),
        "surface_auc_matrix": matrix(transfer_surface, "auc"),
        "mean_proba_matrix": matrix(transfer_probe, "mean_proba"),
        "base_probe_trajectory": base_traj,
        "on_kaggle": on_kaggle(),
    }

    out_path = Path(args.out) if args.out else train_dir / "transfer_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n=== probe AUC matrix (rows=source, cols=target) ===", flush=True)
    header = "src\\tgt" + "".join(f"{t:>8}" for t in steps)
    print(header, flush=True)
    for s in steps:
        row = f"{s:>7}" + "".join(
            f"{out['probe_auc_matrix'][str(s)][str(t)]:>8.3f}" for t in steps
        )
        print(row, flush=True)

    if base_traj:
        print("\n=== frozen base probe trajectory ===", flush=True)
        print(
            f"{'step':>6}  {'probe_auc':>9}  {'mean_p':>7}  {'gold':>7}  {'proxy':>7}  {'surf':>7}",
            flush=True,
        )
        for r in base_traj:
            print(
                f"{r['step']:6d}  {r['probe_auc']:9.3f}  {r['mean_proba']:7.3f}  "
                f"{r['gold_mean']:7.3f}  {r['proxy_mean']:7.3f}  {r['surface_auc']:7.3f}",
                flush=True,
            )

    print(f"wrote {out_path}", flush=True)
    if on_kaggle():
        work = Path("/kaggle/working") / "transfer_results.json"
        work.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"also wrote {work}", flush=True)


if __name__ == "__main__":
    main()
