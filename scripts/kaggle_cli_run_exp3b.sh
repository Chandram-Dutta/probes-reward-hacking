#!/usr/bin/env bash
# Push source + checkpoint datasets and run Exp 3b via Kaggle CLI.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SRC_DS="$ROOT/kaggle/datasets/probes-rh-src"
CKPT_DS="$ROOT/kaggle/datasets/probes-rh-exp3a-ckpt"
KERN="$ROOT/kaggle/kernels/exp3b"

echo "== packaging source dataset =="
rm -rf "$SRC_DS/probes-reward-hacking"
mkdir -p "$SRC_DS"
# Exclude only repo-root data/ artifacts — never src/probes_rh/data/
rsync -a \
  --exclude '.git' --exclude '.venv' --exclude '__pycache__' \
  --exclude '*.zip' --exclude 'notes' --exclude 'AGENTS.md' \
  --exclude 'kaggle' --exclude 'exp3a_*' --exclude 'outputs' \
  --exclude '/data/' \
  ./ "$SRC_DS/probes-reward-hacking/"

echo "== packaging checkpoint dataset =="
mkdir -p "$CKPT_DS"
if [[ ! -f "$ROOT/exp3a_checkpoint.zip" ]]; then
  echo "missing exp3a_checkpoint.zip at repo root" >&2
  exit 1
fi
cp -f "$ROOT/exp3a_checkpoint.zip" "$CKPT_DS/"

echo "== create/version datasets =="
if kaggle datasets status chandramdutta/probes-rh-src >/dev/null 2>&1; then
  kaggle datasets version -p "$SRC_DS" -m "sync src $(date -u +%Y-%m-%dT%H:%MZ)" --dir-mode zip
else
  kaggle datasets create -p "$SRC_DS" --dir-mode zip
fi

if kaggle datasets status chandramdutta/probes-rh-exp3a-ckpt >/dev/null 2>&1; then
  kaggle datasets version -p "$CKPT_DS" -m "exp3a ckpt $(date -u +%Y-%m-%dT%H:%MZ)" --dir-mode zip
else
  kaggle datasets create -p "$CKPT_DS" --dir-mode zip
fi

echo "== push kernel on GPU T4 (Kaggle maps NvidiaTeslaT4 → T4 x2 on free tier) =="
# CLI accelerator flag; metadata also sets machine_shape=NvidiaTeslaT4
kaggle kernels push -p "$KERN" --accelerator NvidiaTeslaT4

echo "== status =="
kaggle kernels status chandramdutta/probes-rh-exp3b || true
echo "Poll with:  kaggle kernels status chandramdutta/probes-rh-exp3b"
echo "Logs:       kaggle kernels logs chandramdutta/probes-rh-exp3b"
echo "Outputs:    kaggle kernels output chandramdutta/probes-rh-exp3b -p results/exp3b"
