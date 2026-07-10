# probes-reward-hacking

Linear probes for reward hacking under **proxy / unverifiable** rewards.

## Run on Kaggle (CLI)

Requires [Kaggle CLI](https://github.com/Kaggle/kaggle-cli) (`kaggle` authenticated).

### Exp 3c — mid-training residual curve (recommended next)

```bash
bash scripts/kaggle_cli_run_exp3c.sh
```

Trains GRPO with checkpoints every 25 steps on dual T4, then scores + residual-probes base and each checkpoint (policy on cuda:0, gold on cuda:1 during scoring).

```bash
kaggle kernels status chandramdutta/probes-rh-exp3c
kaggle kernels logs chandramdutta/probes-rh-exp3c
kaggle kernels output chandramdutta/probes-rh-exp3c -p results/exp3c
```

### Exp 3b — residual probes on a fixed adapter

```bash
# needs exp3a_checkpoint.zip at repo root (trained LoRA)
bash scripts/kaggle_cli_run_exp3b.sh
```

```bash
kaggle kernels status chandramdutta/probes-rh-exp3b
kaggle kernels output chandramdutta/probes-rh-exp3b -p results/exp3b
```

## Local

```bash
uv sync
python scripts/prepare_data.py --max-prompts 2000
# full mid-training curve (auto multi-GPU train if 2+ CUDA devices)
python scripts/run_mid_training_curve.py --train-dir outputs/exp3c_midcurve --limit 256
# or single final adapter
python scripts/train_grpo.py --max-steps 100 --output-dir outputs/exp3a_grpo
python scripts/score_rollouts.py --adapter outputs/exp3a_grpo/final --limit 256
python scripts/run_probes.py --adapter outputs/exp3a_grpo/final --label-mode residual
```

## Layout

```
src/probes_rh/     # package
scripts/           # CLIs (incl. run_mid_training_curve.py)
kaggle/
  datasets/        # dataset-metadata + staged files
  kernels/exp3b/   # residual probes on fixed ckpt
  kernels/exp3c/   # mid-training curve (train + curve)
```
