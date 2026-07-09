# probes-reward-hacking

Research on whether **linear probes** on LLM activations detect **reward hacking** — first under verifiable rewards (RLVR), now under **proxy / unverifiable** rewards.

## Current focus: Experiment 3 (proxy overoptimization)

See [`notes/exp3_unverifiable_proxy.md`](notes/exp3_unverifiable_proxy.md).

**Question:** When GRPO only sees a misspecified proxy (length + flattery), and a frozen gold RM scores quality offline, do policy probes beat output baselines?

| Role | Default |
|---|---|
| Policy | `Qwen/Qwen3-0.6B` (thinking off) |
| Train reward | Engineered proxy (never gold) |
| Gold | `Skywork/Skywork-Reward-V2-Llama-3.2-3B` (offline only) |
| Target GPU | Kaggle dual T4 |

## Setup

```bash
uv sync
# or: pip install -e .
```

### Kaggle (quick)

```python
!git clone https://github.com/Chandram-Dutta/probes-reward-hacking.git
%cd probes-reward-hacking

# make package importable (either works)
!pip install -e . -q
# scripts also bootstrap src/ onto PYTHONPATH if install is skipped

!python scripts/prepare_data.py --max-prompts 2000
```

If you see `No module named 'probes_rh'`, you are not running from the repo
root or the clone failed. Re-run from `probes-reward-hacking/` after clone.

## Pipeline

```bash
# 1) prompts
uv run python scripts/prepare_data.py --max-prompts 2000

# 2) GRPO vs proxy (Kaggle dual T4: auto multi-GPU via accelerate)
uv run python scripts/train_grpo.py --max-steps 100
# force one GPU:  uv run python scripts/train_grpo.py --single-gpu --max-steps 100

# 3) rollouts + gold scores (policy on cuda:0, gold RM on cuda:1 when 2 GPUs)
uv run python scripts/score_rollouts.py \
  --adapter outputs/exp3a_grpo/final \
  --limit 256

# 4) probes vs baselines
uv run python scripts/run_probes.py \
  --adapter outputs/exp3a_grpo/final
```

### Dual T4 (Kaggle)

- **Train:** `train_grpo.py` detects `torch.cuda.device_count() >= 2` and re-launches with `accelerate launch --multi_gpu --num_processes=2`.
- **Score:** generation on `cuda:0`, frozen gold RM on `cuda:1` (no sequential free/reload).
- **Probes:** single GPU is enough (activation dumps).

Do **not** feed gold into training. Gold is analysis-only.

## Package layout

```
src/probes_rh/
  data/          # prompt prep
  rewards/       # proxy + gold RM
  train/         # GRPO config
  eval/          # activations, probes, baselines
scripts/         # CLI entrypoints
notes/           # experiment writeups
```

## Notes

- Scripts first; convert to notebooks for Colab/Kaggle as needed.
- Ask / fix the training environment before long GPU runs.
