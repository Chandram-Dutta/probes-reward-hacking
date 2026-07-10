# probes-reward-hacking

Research on whether **linear probes** on LLM activations detect **reward hacking** under **proxy / unverifiable** rewards.

## Current focus

- **Exp 3a** (done pilot): proxy GRPO + gold RM + probes — see local `notes/`  
- **Exp 3b** (next): residual labels, more rollouts, thinking strip, durable Kaggle outputs

| Role | Default |
|---|---|
| Policy | `Qwen/Qwen3-0.6B` (thinking stripped) |
| Train reward | Engineered proxy (never gold) |
| Gold | `Skywork/Skywork-Reward-V2-Llama-3.2-3B` (offline only) |
| Target GPU | Kaggle dual T4 |
| Artifacts | `/kaggle/working/outputs/<exp_name>/` |

## Setup

```bash
uv sync
# or: pip install -e .
```

### Kaggle: outputs that you can reuse after restart

Kaggle **deletes `/kaggle/working` when the session dies**. Within a session, always write under:

```text
/kaggle/working/outputs/exp3a_grpo/
```

**Before you leave / restart:**

```python
%cd /kaggle/working/probes-reward-hacking
!python scripts/kaggle_pack.py --name exp3a_grpo          # full (adapter + files)
# or light only:
!python scripts/kaggle_pack.py --name exp3a_grpo --light
```

Download the zip from `/kaggle/working/`. After a new session, re-upload it (or attach as a Dataset) and restore:

```python
%cd /kaggle/working/probes-reward-hacking
!git pull
!python scripts/kaggle_setup.py
!python scripts/kaggle_restore.py --zip /kaggle/working/exp3a_grpo_pack.zip
# if Dataset: --zip /kaggle/input/<dataset-name>/exp3a_grpo_pack.zip
!ls /kaggle/working/outputs/exp3a_grpo/final
```

### Kaggle: Exp 3b from existing 3a checkpoint (no retrain)

```python
%cd /kaggle/working
!rm -rf probes-reward-hacking
!git clone https://github.com/Chandram-Dutta/probes-reward-hacking.git
%cd probes-reward-hacking
!python scripts/kaggle_setup.py
!python scripts/prepare_data.py --max-prompts 2000 --out-dir data/exp3

# restore checkpoint zip you uploaded
!python scripts/kaggle_restore.py --zip /kaggle/working/exp3a_checkpoint.zip

# more rollouts (thinking stripped in proxy/features)
!python scripts/score_rollouts.py \
  --model Qwen/Qwen3-0.6B \
  --adapter /kaggle/working/outputs/exp3a_grpo/final \
  --prompts data/exp3/prompts.jsonl \
  --out /kaggle/working/outputs/exp3a_grpo/rollouts_3b.jsonl \
  --limit 1024

# residual-label probes (Exp 3b default)
!python scripts/run_probes.py \
  --rollouts /kaggle/working/outputs/exp3a_grpo/rollouts_3b.jsonl \
  --model Qwen/Qwen3-0.6B \
  --adapter /kaggle/working/outputs/exp3a_grpo/final \
  --label-mode residual \
  --out /kaggle/working/outputs/exp3a_grpo/probe_results_residual.json

!python scripts/kaggle_pack.py --name exp3a_grpo --light
```

If LoRA fails with `incompatible version of torchao`:

```python
!pip uninstall -y torchao
```

## Local pipeline

```bash
python scripts/prepare_data.py --max-prompts 2000
python scripts/train_grpo.py --max-steps 100
python scripts/score_rollouts.py --adapter outputs/exp3a_grpo/final --limit 256
python scripts/run_probes.py --adapter outputs/exp3a_grpo/final --label-mode residual
```

### Dual T4

- **Train:** auto `accelerate` multi-GPU when 2+ devices.  
- **Score:** policy `cuda:0`, gold `cuda:1`.  
- **Probes:** one GPU.

Gold is analysis-only — never in the GRPO advantage.

## Package layout

```
src/probes_rh/
  paths.py       # /kaggle/working/outputs helpers
  chat.py        # thinking-off + strip <think>
  data/ rewards/ train/ eval/
scripts/
  kaggle_setup.py kaggle_restore.py kaggle_pack.py
  prepare_data.py train_grpo.py score_rollouts.py run_probes.py
```
