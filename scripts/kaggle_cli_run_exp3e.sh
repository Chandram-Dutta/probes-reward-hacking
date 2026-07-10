#!/usr/bin/env bash
# Package source + Exp 3c artifacts, push Exp 3e held-out transfer kernel.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SRC_DS="$ROOT/kaggle/datasets/probes-rh-src"
ART_DS="$ROOT/kaggle/datasets/probes-rh-exp3c-artifacts"
KERN="$ROOT/kaggle/kernels/exp3e"
MID="$ROOT/results/exp3c/outputs/exp3c_midcurve"

if [[ ! -d "$MID/curve" ]]; then
  echo "missing Exp 3c curve dir: $MID/curve" >&2
  exit 1
fi

echo "== packaging source dataset =="
rm -rf "$SRC_DS/probes-reward-hacking"
mkdir -p "$SRC_DS"
rsync -a \
  --exclude '.git' --exclude '.venv' --exclude '__pycache__' \
  --exclude '*.zip' --exclude 'notes' --exclude 'AGENTS.md' \
  --exclude 'kaggle' --exclude 'exp3a_*' --exclude 'outputs' \
  --exclude '/data/' --exclude 'results' \
  ./ "$SRC_DS/probes-reward-hacking/"

echo "== packaging exp3c artifacts =="
rm -rf "$ART_DS/exp3c_midcurve"
mkdir -p "$ART_DS/exp3c_midcurve"
for d in checkpoint-25 checkpoint-50 checkpoint-75 checkpoint-100 final; do
  if [[ -d "$MID/$d" ]]; then
    mkdir -p "$ART_DS/exp3c_midcurve/$d"
    for f in adapter_config.json adapter_model.safetensors adapter_model.bin; do
      [[ -f "$MID/$d/$f" ]] && cp -f "$MID/$d/$f" "$ART_DS/exp3c_midcurve/$d/"
    done
  fi
done
rsync -a \
  --include '*/' \
  --include 'rollouts.jsonl' \
  --include 'point_summary.json' \
  --include 'acts_mean_completion.npz' \
  --exclude '*' \
  "$MID/curve/" "$ART_DS/exp3c_midcurve/curve/"
[[ -f "$MID/mid_training_curve.json" ]] && cp -f "$MID/mid_training_curve.json" "$ART_DS/exp3c_midcurve/"
[[ -f "$MID/train_config.json" ]] && cp -f "$MID/train_config.json" "$ART_DS/exp3c_midcurve/"

if [[ ! -f "$ART_DS/dataset-metadata.json" ]]; then
  cat > "$ART_DS/dataset-metadata.json" <<'EOF'
{
  "title": "probes-rh-exp3c-artifacts",
  "id": "chandramdutta/probes-rh-exp3c-artifacts",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF
fi

echo "== create/version datasets =="
if kaggle datasets status chandramdutta/probes-rh-src >/dev/null 2>&1; then
  kaggle datasets version -p "$SRC_DS" -m "exp3e heldout $(date -u +%Y-%m-%dT%H:%MZ)" --dir-mode zip
else
  kaggle datasets create -p "$SRC_DS" --dir-mode zip
fi

if kaggle datasets status chandramdutta/probes-rh-exp3c-artifacts >/dev/null 2>&1; then
  kaggle datasets version -p "$ART_DS" -m "exp3c artifacts for 3e $(date -u +%Y-%m-%dT%H:%MZ)" --dir-mode zip
else
  kaggle datasets create -p "$ART_DS" --dir-mode zip
fi

# Wait until artifacts dataset is ready so kernel can attach it
echo "== wait for artifacts dataset =="
for i in $(seq 1 30); do
  st=$(kaggle datasets status chandramdutta/probes-rh-exp3c-artifacts 2>&1 || true)
  echo "  try $i: $st"
  if echo "$st" | grep -qi ready; then
    break
  fi
  sleep 10
done

echo "== push kernel =="
kaggle kernels push -p "$KERN" --accelerator NvidiaTeslaT4
kaggle kernels status chandramdutta/probes-rh-exp3e || true
echo "Poll: kaggle kernels status chandramdutta/probes-rh-exp3e"
echo "Logs: kaggle kernels logs chandramdutta/probes-rh-exp3e"
echo "Out:  kaggle kernels output chandramdutta/probes-rh-exp3e -p results/exp3e"
