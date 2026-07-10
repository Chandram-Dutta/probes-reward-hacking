# probes-reward-hacking

Linear probes for reward hacking under **proxy / unverifiable** rewards.

## Run on Kaggle (CLI)

Requires [Kaggle CLI](https://github.com/Kaggle/kaggle-cli) (`kaggle` authenticated).

```bash
# needs exp3a_checkpoint.zip at repo root (trained LoRA)
bash scripts/kaggle_cli_run_exp3b.sh
```

That will:

1. Upload source → Dataset `chandramdutta/probes-rh-src`
2. Upload checkpoint → Dataset `chandramdutta/probes-rh-exp3a-ckpt`
3. Push / run kernel `chandramdutta/probes-rh-exp3b` (GPU, internet on)

```bash
kaggle kernels status chandramdutta/probes-rh-exp3b
kaggle kernels logs chandramdutta/probes-rh-exp3b
kaggle kernels output chandramdutta/probes-rh-exp3b -p results/exp3b
```

Kernel script: `kaggle/kernels/exp3b/run_exp3b.py`  
(no retrain — scores 1024 rollouts + residual/quantile probes from Exp 3a adapter)

## Local

```bash
uv sync
python scripts/prepare_data.py --max-prompts 2000
python scripts/train_grpo.py --max-steps 100 --output-dir outputs/exp3a_grpo
python scripts/score_rollouts.py --adapter outputs/exp3a_grpo/final --limit 256
python scripts/run_probes.py --adapter outputs/exp3a_grpo/final --label-mode residual
```

## Layout

```
src/probes_rh/     # package
scripts/           # CLIs
kaggle/
  datasets/        # dataset-metadata + staged files
  kernels/exp3b/   # kernel-metadata + run_exp3b.py
```
