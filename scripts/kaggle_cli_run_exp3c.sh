#!/usr/bin/env bash
# Push source dataset and run Exp 3c mid-training curve on dual T4.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SRC_DS="$ROOT/kaggle/datasets/probes-rh-src"
KERN="$ROOT/kaggle/kernels/exp3c"

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

echo "== create/version source dataset =="
if kaggle datasets status chandramdutta/probes-rh-src >/dev/null 2>&1; then
  kaggle datasets version -p "$SRC_DS" -m "exp3c midcurve $(date -u +%Y-%m-%dT%H:%MZ)" --dir-mode zip
else
  kaggle datasets create -p "$SRC_DS" --dir-mode zip
fi

echo "== push kernel on GPU T4 (NvidiaTeslaT4 → T4 x2 on free tier) =="
kaggle kernels push -p "$KERN" --accelerator NvidiaTeslaT4

echo "== status =="
kaggle kernels status chandramdutta/probes-rh-exp3c || true
echo "Poll with:  kaggle kernels status chandramdutta/probes-rh-exp3c"
echo "Logs:       kaggle kernels logs chandramdutta/probes-rh-exp3c"
echo "Outputs:    kaggle kernels output chandramdutta/probes-rh-exp3c -p results/exp3c"
echo "(Ignore any older slug probes-rh-exp3c-midcurve from a mismatched title.)"
