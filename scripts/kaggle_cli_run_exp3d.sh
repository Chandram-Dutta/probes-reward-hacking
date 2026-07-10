#!/usr/bin/env bash
# Package Exp 3c artifacts + source, push Exp 3d transfer kernel on dual T4.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SRC_DS="$ROOT/kaggle/datasets/probes-rh-src"
ART_DS="$ROOT/kaggle/datasets/probes-rh-exp3c-artifacts"
KERN="$ROOT/kaggle/kernels/exp3d"
MID="$ROOT/results/exp3c/outputs/exp3c_midcurve"

if [[ ! -d "$MID/curve" ]]; then
  echo "missing Exp 3c curve dir: $MID/curve" >&2
  echo "pull with: kaggle kernels output chandramdutta/probes-rh-exp3c -p results/exp3c" >&2
  exit 1
fi

echo "== packaging source dataset =="
rm -rf "$SRC_DS/probes-reward-hacking"
mkdir -p "$SRC_DS"
# Exclude only repo-root data/ artifacts — never src/probes_rh/data/
rsync -a \
  --exclude '.git' --exclude '.venv' --exclude '__pycache__' \
  --exclude '*.zip' --exclude 'notes' --exclude 'AGENTS.md' \
  --exclude 'kaggle' --exclude 'exp3a_*' --exclude 'outputs' \
  --exclude '/data/' --exclude 'results' \
  ./ "$SRC_DS/probes-reward-hacking/"

echo "== packaging exp3c artifacts (checkpoints + rollouts, no completions parquet) =="
rm -rf "$ART_DS/exp3c_midcurve"
mkdir -p "$ART_DS/exp3c_midcurve"
# adapters
for d in checkpoint-25 checkpoint-50 checkpoint-75 checkpoint-100 final; do
  if [[ -d "$MID/$d" ]]; then
    mkdir -p "$ART_DS/exp3c_midcurve/$d"
    # keep lightweight adapter files only
    for f in adapter_config.json adapter_model.safetensors adapter_model.bin; do
      [[ -f "$MID/$d/$f" ]] && cp -f "$MID/$d/$f" "$ART_DS/exp3c_midcurve/$d/"
    done
  fi
done
# curve rollouts + summaries (not full probe jsons required, but cheap)
rsync -a \
  --include '*/' \
  --include 'rollouts.jsonl' \
  --include 'point_summary.json' \
  --exclude '*' \
  "$MID/curve/" "$ART_DS/exp3c_midcurve/curve/"
[[ -f "$MID/mid_training_curve.json" ]] && cp -f "$MID/mid_training_curve.json" "$ART_DS/exp3c_midcurve/"
[[ -f "$MID/train_config.json" ]] && cp -f "$MID/train_config.json" "$ART_DS/exp3c_midcurve/"

# dataset metadata
if [[ ! -f "$ART_DS/dataset-metadata.json" ]]; then
  cat > "$ART_DS/dataset-metadata.json" <<'EOF'
{
  "title": "probes-rh-exp3c-artifacts",
  "id": "chandramdutta/probes-rh-exp3c-artifacts",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF
fi

echo "== artifact sizes =="
du -sh "$ART_DS" "$ART_DS/exp3c_midcurve"/* 2>/dev/null | head -30

echo "== create/version datasets =="
if kaggle datasets status chandramdutta/probes-rh-src >/dev/null 2>&1; then
  kaggle datasets version -p "$SRC_DS" -m "exp3d transfer $(date -u +%Y-%m-%dT%H:%MZ)" --dir-mode zip
else
  kaggle datasets create -p "$SRC_DS" --dir-mode zip
fi

if kaggle datasets status chandramdutta/probes-rh-exp3c-artifacts >/dev/null 2>&1; then
  kaggle datasets version -p "$ART_DS" -m "exp3c midcurve adapters+rollouts $(date -u +%Y-%m-%dT%H:%MZ)" --dir-mode zip
else
  kaggle datasets create -p "$ART_DS" --dir-mode zip
fi

echo "== push kernel on GPU T4 =="
kaggle kernels push -p "$KERN" --accelerator NvidiaTeslaT4

echo "== status =="
kaggle kernels status chandramdutta/probes-rh-exp3d || true
echo "Poll with:  kaggle kernels status chandramdutta/probes-rh-exp3d"
echo "Logs:       kaggle kernels logs chandramdutta/probes-rh-exp3d"
echo "Outputs:    kaggle kernels output chandramdutta/probes-rh-exp3d -p results/exp3d"
