#!/usr/bin/env python3
"""Exp 3e: held-out-prompt residual probe transfer.

Fit residual probes on prompt group A at source step s; evaluate only on
disjoint group B at each target step t. Saves per-row scores so label-only
and surface baselines can be checked offline.

Uses Exp 3c midcurve rollouts + adapters (no GRPO retrain).
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
                cand = train_dir / name
                if not (cand / "adapter_config.json").exists():
                    cand = train_dir / f"checkpoint-{step}"
                if (cand / "adapter_config.json").exists() or (
                    cand / "adapter_model.safetensors"
                ).exists():
                    adapter = str(cand)
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
    return sorted(points, key=lambda p: p["step"])


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
        [[*(r["features"][k] for k in names[:-1]), r["proxy"]] for r in rows],
        dtype=np.float64,
    )
    return mat, names


def residual_and_labels(
    rows: list[dict], residual_q: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute residual + labels within this subset only (no cross-group leak)."""
    proxy = np.array([r["proxy"] for r in rows], dtype=np.float64)
    gold = np.array([r["gold"] for r in rows], dtype=np.float64)
    resid = gold_residual(proxy, gold)
    y = make_hack_labels(proxy, gold, mode="residual", residual_q=residual_q)
    return proxy, gold, resid, y


def split_prompt_ids(
    prompt_ids: list[str], fit_frac: float, seed: int
) -> tuple[set[str], set[str]]:
    uniq = sorted(set(prompt_ids))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    n_fit = max(1, int(round(fit_frac * len(uniq))))
    n_fit = min(n_fit, len(uniq) - 1)  # keep at least one eval prompt
    fit_set = {uniq[i] for i in order[:n_fit]}
    eval_set = {uniq[i] for i in order[n_fit:]}
    return fit_set, eval_set


def filter_by_prompts(rows: list[dict], keep: set[str]) -> list[dict]:
    return [r for r in rows if r.get("prompt_id", "") in keep]


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
    for r in tqdm(rows, desc=f"acts {adapter or 'base'}"):
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


def select_best_layer_on_subset(
    acts: dict[int, np.ndarray],
    y: np.ndarray,
    seed: int,
) -> tuple[int | None, float]:
    """Pick best layer by random split AUC within fit set (no prompt groups needed
    if fit set is already prompt-held-out from eval; still avoid pure train AUC)."""
    from sklearn.model_selection import train_test_split

    best_layer, best_auc = None, float("-inf")
    for layer, X in sorted(acts.items()):
        if len(np.unique(y)) < 2 or len(y) < 10:
            continue
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.3, random_state=seed, stratify=y if y.sum() > 1 and (y == 0).sum() > 1 else None
            )
        except ValueError:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.3, random_state=seed
            )
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue
        clf = fit_linear_probe_full(X_tr, y_tr)
        auc = probe_auc(clf, X_te, y_te)
        if auc == auc and auc > best_auc:
            best_auc = float(auc)
            best_layer = layer
    if best_layer is None:
        # fallback: first layer
        if acts:
            best_layer = sorted(acts.keys())[0]
            return best_layer, float("nan")
        return None, float("nan")
    return best_layer, best_auc


def main() -> None:
    p = argparse.ArgumentParser(description="Exp 3e held-out prompt probe transfer")
    p.add_argument("--train-dir", default=None)
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--residual-q", type=float, default=0.25)
    p.add_argument("--fit-frac", type=float, default=0.5, help="Fraction of prompts for fit group A")
    p.add_argument("--layers", default="mid_late")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--force-acts", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    train_dir = Path(args.train_dir) if args.train_dir else output_root() / "exp3c_midcurve"
    curve_dir = train_dir / "curve"
    points = discover_curve_points(curve_dir, train_dir)
    if len(points) < 2:
        raise SystemExit(f"need >=2 curve points under {train_dir}, found {points}")

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  points={[(pt['step'], pt['name']) for pt in points]}", flush=True)

    # Load all step pools (full rows first for prompt split)
    full_pools: dict[int, dict] = {}
    all_prompt_ids: list[str] = []
    for pt in points:
        rows = load_rollouts(Path(pt["rollouts"]))
        if len(rows) < 20:
            raise SystemExit(f"too few rollouts at step {pt['step']}: {len(rows)}")
        pids = [r.get("prompt_id", str(i)) for i, r in enumerate(rows)]
        all_prompt_ids.extend(pids)
        full_pools[pt["step"]] = {"meta": pt, "rows": rows, "prompt_ids": pids}

    # Global A/B split on unique prompt ids (same across all steps)
    # Prefer base step's prompt list for stability
    base_step = min(full_pools.keys())
    base_pids = full_pools[base_step]["prompt_ids"]
    fit_prompts, eval_prompts = split_prompt_ids(base_pids, args.fit_frac, args.seed)
    print(
        f"prompt split: n_fit={len(fit_prompts)} n_eval={len(eval_prompts)} "
        f"fit_frac={args.fit_frac} seed={args.seed}",
        flush=True,
    )

    # Collect activations once per step (full pool), then index into A/B
    pools: dict[int, dict] = {}
    for step, fp in full_pools.items():
        pt = fp["meta"]
        rows = fp["rows"]
        point_dir = Path(pt["point_dir"])
        acts = collect_activations(
            rows,
            model_id=args.model,
            adapter=pt["adapter"],
            layers_arg=args.layers,
            device=device,
            cache_path=point_dir / "acts_mean_completion.npz",
            force=args.force_acts,
        )
        # Align acts with rows order
        pid = np.array([r.get("prompt_id", str(i)) for i, r in enumerate(rows)])
        fit_mask = np.array([p in fit_prompts for p in pid])
        eval_mask = np.array([p in eval_prompts for p in pid])

        rows_a = [r for r, m in zip(rows, fit_mask) if m]
        rows_b = [r for r, m in zip(rows, eval_mask) if m]
        _, _, resid_a, y_a = residual_and_labels(rows_a, args.residual_q)
        _, _, resid_b, y_b = residual_and_labels(rows_b, args.residual_q)
        # Also residual within full pool for offline diagnostics
        proxy_full, gold_full, resid_full, y_full = residual_and_labels(rows, args.residual_q)
        surf_a, surf_names = surface_matrix(rows_a)
        surf_b, _ = surface_matrix(rows_b)

        acts_a = {L: X[fit_mask] for L, X in acts.items()}
        acts_b = {L: X[eval_mask] for L, X in acts.items()}

        pools[step] = {
            "meta": pt,
            "rows": rows,
            "rows_a": rows_a,
            "rows_b": rows_b,
            "fit_mask": fit_mask,
            "eval_mask": eval_mask,
            "y_a": y_a,
            "y_b": y_b,
            "resid_a": resid_a,
            "resid_b": resid_b,
            "resid_full": resid_full,
            "y_full": y_full,
            "proxy_full": proxy_full,
            "gold_full": gold_full,
            "acts_a": acts_a,
            "acts_b": acts_b,
            "surf_a": surf_a,
            "surf_b": surf_b,
            "surf_names": surf_names,
            "pid": pid,
            "n_a": len(rows_a),
            "n_b": len(rows_b),
            "hack_rate_a": float(y_a.mean()) if len(y_a) else float("nan"),
            "hack_rate_b": float(y_b.mean()) if len(y_b) else float("nan"),
            "proxy_mean": float(proxy_full.mean()),
            "gold_mean": float(gold_full.mean()),
        }
        print(
            f"step {step}: n_a={len(rows_a)} n_b={len(rows_b)} "
            f"hack_a={y_a.mean():.3f} hack_b={y_b.mean():.3f} "
            f"proxy={proxy_full.mean():.3f} gold={gold_full.mean():.3f}",
            flush=True,
        )

    steps = sorted(pools.keys())
    transfer_rows: list[dict] = []
    per_row_scores: list[dict] = []
    source_summaries: list[dict] = []

    for src in steps:
        sp = pools[src]
        if sp["n_a"] < 20 or len(np.unique(sp["y_a"])) < 2:
            print(f"skip source {src}: insufficient fit set", flush=True)
            continue

        best_layer, best_cv = select_best_layer_on_subset(
            sp["acts_a"], sp["y_a"], seed=args.seed
        )
        if best_layer is None:
            print(f"skip source {src}: no layer", flush=True)
            continue

        X_a = sp["acts_a"][best_layer]
        clf = fit_linear_probe_full(X_a, sp["y_a"])
        surf_clf = fit_linear_probe_full(sp["surf_a"], sp["y_a"])

        # within-A held-out style already used for layer pick; also report fit-set train AUC
        train_auc = probe_auc(clf, X_a, sp["y_a"])
        train_surf = probe_auc(surf_clf, sp["surf_a"], sp["y_a"])

        source_summaries.append(
            {
                "source_step": src,
                "source_name": sp["meta"]["name"],
                "best_layer": best_layer,
                "layer_select_auc": best_cv,
                "fit_train_auc": train_auc,
                "fit_surface_train_auc": train_surf,
                "n_fit": sp["n_a"],
                "hack_rate_fit": sp["hack_rate_a"],
            }
        )
        print(
            f"source step={src} layer={best_layer} layer_sel={best_cv:.3f} "
            f"fit_train_auc={train_auc:.3f}",
            flush=True,
        )

        # Map prompt_id -> source residual (full-pool residual at source) for offline carry-forward
        src_resid_by_pid = {
            pid: float(sp["resid_full"][i])
            for i, pid in enumerate(sp["pid"])
        }
        src_y_by_pid = {
            pid: int(sp["y_full"][i]) for i, pid in enumerate(sp["pid"])
        }

        for tgt in steps:
            tp = pools[tgt]
            if best_layer not in tp["acts_b"] or tp["n_b"] < 10:
                continue
            if len(np.unique(tp["y_b"])) < 2:
                auc = float("nan")
                surf_auc = float("nan")
                mean_p = float("nan")
                proba = np.full(tp["n_b"], np.nan)
                surf_p = np.full(tp["n_b"], np.nan)
            else:
                X_b = tp["acts_b"][best_layer]
                proba = probe_proba(clf, X_b)
                surf_p = probe_proba(surf_clf, tp["surf_b"])
                auc = probe_auc(clf, X_b, tp["y_b"])
                surf_auc = probe_auc(surf_clf, tp["surf_b"], tp["y_b"])
                mean_p = float(np.nanmean(proba))

            # Label-only carry-forward on B: source residual / label for same prompt
            src_resid_b = []
            src_y_b = []
            for r in tp["rows_b"]:
                pid = r.get("prompt_id", "")
                src_resid_b.append(src_resid_by_pid.get(pid, float("nan")))
                src_y_b.append(src_y_by_pid.get(pid, -1))
            src_resid_b = np.array(src_resid_b, dtype=float)
            src_y_b = np.array(src_y_b, dtype=int)

            # AUC: lower residual = more residual-bad → use -resid as score
            carry_resid_auc = float("nan")
            carry_label_auc = float("nan")
            if len(np.unique(tp["y_b"])) >= 2 and np.isfinite(src_resid_b).all():
                try:
                    from sklearn.metrics import roc_auc_score

                    carry_resid_auc = float(
                        roc_auc_score(tp["y_b"], -src_resid_b)
                    )
                except ValueError:
                    pass
            if len(np.unique(tp["y_b"])) >= 2 and set(np.unique(src_y_b)) <= {0, 1}:
                try:
                    from sklearn.metrics import roc_auc_score

                    if len(np.unique(src_y_b)) >= 2:
                        carry_label_auc = float(roc_auc_score(tp["y_b"], src_y_b))
                except ValueError:
                    pass

            transfer_rows.append(
                {
                    "source_step": src,
                    "target_step": tgt,
                    "layer": best_layer,
                    "probe_auc": auc,
                    "surface_auc": surf_auc,
                    "mean_probe_proba": mean_p,
                    "carry_source_residual_auc": carry_resid_auc,
                    "carry_source_label_auc": carry_label_auc,
                    "n_eval": tp["n_b"],
                    "hack_rate_eval": tp["hack_rate_b"],
                    "same_step": src == tgt,
                }
            )
            print(
                f"  {src}->{tgt}: probe={auc:.3f} surface={surf_auc:.3f} "
                f"carry_resid={carry_resid_auc:.3f} carry_label={carry_label_auc:.3f} "
                f"mean_p={mean_p:.3f}",
                flush=True,
            )

            # Per-row scores for offline checks
            for j, r in enumerate(tp["rows_b"]):
                pid = r.get("prompt_id", "")
                per_row_scores.append(
                    {
                        "source_step": src,
                        "target_step": tgt,
                        "prompt_id": pid,
                        "y_eval": int(tp["y_b"][j]),
                        "probe_proba": float(proba[j]) if j < len(proba) else float("nan"),
                        "surface_proba": float(surf_p[j]) if j < len(surf_p) else float("nan"),
                        "proxy": float(r["proxy"]),
                        "gold": float(r["gold"]),
                        "residual_eval": float(tp["resid_b"][j]),
                        "source_residual": float(src_resid_by_pid.get(pid, float("nan"))),
                        "source_y": int(src_y_by_pid.get(pid, -1)),
                        "n_chars": float(r["features"].get("n_chars", 0)),
                    }
                )

    def matrix(rows: list[dict], key: str) -> dict:
        m = {str(s): {str(t): None for t in steps} for s in steps}
        for r in rows:
            m[str(r["source_step"])][str(r["target_step"])] = r.get(key)
        return m

    # Base source trajectory on held-out B
    base_traj = [r for r in transfer_rows if r["source_step"] == base_step]

    out = {
        "experiment": "exp3e_heldout_prompt_transfer",
        "model": args.model,
        "train_dir": str(train_dir),
        "residual_q": args.residual_q,
        "fit_frac": args.fit_frac,
        "seed": args.seed,
        "layers": args.layers,
        "n_fit_prompts": len(fit_prompts),
        "n_eval_prompts": len(eval_prompts),
        "fit_prompt_ids": sorted(fit_prompts),
        "eval_prompt_ids": sorted(eval_prompts),
        "steps": steps,
        "pools": {
            str(s): {
                "name": pools[s]["meta"]["name"],
                "n_a": pools[s]["n_a"],
                "n_b": pools[s]["n_b"],
                "hack_rate_a": pools[s]["hack_rate_a"],
                "hack_rate_b": pools[s]["hack_rate_b"],
                "proxy_mean": pools[s]["proxy_mean"],
                "gold_mean": pools[s]["gold_mean"],
            }
            for s in steps
        },
        "source_summaries": source_summaries,
        "transfer": transfer_rows,
        "probe_auc_matrix": matrix(transfer_rows, "probe_auc"),
        "surface_auc_matrix": matrix(transfer_rows, "surface_auc"),
        "carry_residual_auc_matrix": matrix(transfer_rows, "carry_source_residual_auc"),
        "carry_label_auc_matrix": matrix(transfer_rows, "carry_source_label_auc"),
        "base_heldout_trajectory": base_traj,
        "per_row_scores": per_row_scores,
        "on_kaggle": on_kaggle(),
        "protocol": (
            "Fit residual probe + surface on prompt group A at source step only. "
            "Evaluate on disjoint prompt group B at each target. "
            "Residual labels computed within A (fit) and within B (eval) separately. "
            "carry_* baselines use source-step residual/label for the same prompt_id on B "
            "(prompt-stable residual carry-forward; available offline because same prompts "
            "are scored at every step, but never used as fit features)."
        ),
    }

    out_path = Path(args.out) if args.out else train_dir / "heldout_transfer_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n=== held-out probe AUC (rows=source, cols=target) ===", flush=True)
    print("src\\tgt" + "".join(f"{t:>8}" for t in steps), flush=True)
    for s in steps:
        vals = []
        for t in steps:
            v = out["probe_auc_matrix"][str(s)][str(t)]
            vals.append(f"{v:8.3f}" if v is not None else "     nan")
        print(f"{s:>7}" + "".join(vals), flush=True)

    print("\n=== held-out surface AUC ===", flush=True)
    print("src\\tgt" + "".join(f"{t:>8}" for t in steps), flush=True)
    for s in steps:
        vals = []
        for t in steps:
            v = out["surface_auc_matrix"][str(s)][str(t)]
            vals.append(f"{v:8.3f}" if v is not None else "     nan")
        print(f"{s:>7}" + "".join(vals), flush=True)

    print("\n=== carry source-residual AUC (label-only) ===", flush=True)
    print("src\\tgt" + "".join(f"{t:>8}" for t in steps), flush=True)
    for s in steps:
        vals = []
        for t in steps:
            v = out["carry_residual_auc_matrix"][str(s)][str(t)]
            vals.append(f"{v:8.3f}" if v is not None else "     nan")
        print(f"{s:>7}" + "".join(vals), flush=True)

    print(f"\nwrote {out_path}  per_row_scores={len(per_row_scores)}", flush=True)
    if on_kaggle():
        work = Path("/kaggle/working") / "heldout_transfer_results.json"
        work.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"also wrote {work}", flush=True)


if __name__ == "__main__":
    main()
