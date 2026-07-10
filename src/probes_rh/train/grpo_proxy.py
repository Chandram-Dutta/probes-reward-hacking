"""GRPO training against the misspecified proxy reward (gold never used here)."""

from __future__ import annotations

import inspect
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from probes_rh.data.prep_prompts import load_jsonl
from probes_rh.paths import exp_dir, on_kaggle
from probes_rh.rewards.proxy import ProxyRewardConfig, make_proxy_reward_fn


DEFAULT_POLICY = "Qwen/Qwen3-0.6B"


def default_train_output_dir() -> str:
    if on_kaggle():
        return str(exp_dir("exp3a_grpo"))
    return "outputs/exp3a_grpo"


@dataclass
class Exp3TrainConfig:
    model_id: str = DEFAULT_POLICY
    data_path: str = "data/exp3/prompts_trl.jsonl"
    output_dir: str = ""  # filled in __post_init__-like via train()
    max_steps: int = 100
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_generations: int = 4
    max_completion_length: int = 256
    # Kept for docs / older TRL; filtered out if GRPOConfig rejects it.
    max_prompt_length: int = 512
    learning_rate: float = 1e-5
    logging_steps: int = 1
    save_steps: int = 25
    save_total_limit: int | None = 10
    save_only_model: bool = True
    seed: int = 42
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    bf16: bool = False
    fp16: bool = True
    # Qwen3: disable thinking for stable short completions
    enable_thinking: bool = False
    loss_type: str = "dr_grpo"
    beta: float = 0.0
    report_to: str = "none"
    # proxy weights
    w_length: float = 1.0
    w_agreement: float = 1.5
    w_confidence: float = 1.0
    w_politeness: float = 0.75


def build_dataset(data_path: str) -> Dataset:
    rows = load_jsonl(Path(data_path))
    # keep only fields TRL needs + id for later joins
    cleaned = []
    for r in rows:
        cleaned.append(
            {
                "prompt": r["prompt"],
                "prompt_id": r.get("prompt_id", ""),
            }
        )
    return Dataset.from_list(cleaned)


def _filter_kwargs(cls, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop kwargs unsupported by the installed TRL version."""
    try:
        accepted = set(inspect.signature(cls.__init__).parameters.keys())
    except (TypeError, ValueError):
        return kwargs
    accepted.discard("self")
    # dataclasses / TrainingArguments often accept **kwargs via inheritance;
    # still filter unknowns that raise TypeError in strict inits.
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    dropped = sorted(set(kwargs) - set(filtered))
    if dropped:
        print(f"GRPOConfig: ignoring unsupported args for this TRL: {dropped}")
    return filtered


def _fix_torchao_for_peft() -> None:
    """Kaggle ships torchao 0.10; recent peft requires >=0.16 or no torchao.

    If an incompatible torchao is present, peft raises during LoRA inject.
    Prefer uninstalling torchao (we don't need it) over a heavy upgrade.
    """
    try:
        import importlib.metadata as md

        ver = md.version("torchao")
    except Exception:
        return

    def _parse(v: str) -> tuple[int, ...]:
        parts = []
        for p in v.split(".")[:3]:
            try:
                parts.append(int("".join(c for c in p if c.isdigit()) or "0"))
            except ValueError:
                parts.append(0)
        return tuple(parts)

    if _parse(ver) >= (0, 16, 0):
        return

    print(
        f"Found incompatible torchao=={ver} (peft wants >=0.16). "
        "Uninstalling torchao so LoRA can load..."
    )
    import subprocess
    import sys

    subprocess.check_call(
        [sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # clear any cached import
    for key in list(sys.modules):
        if key == "torchao" or key.startswith("torchao."):
            del sys.modules[key]
    print("torchao uninstalled. Continuing with LoRA.")


def build_trainer(cfg: Exp3TrainConfig) -> GRPOTrainer:
    if cfg.use_lora:
        _fix_torchao_for_peft()

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    proxy_cfg = ProxyRewardConfig(
        w_length=cfg.w_length,
        w_agreement=cfg.w_agreement,
        w_confidence=cfg.w_confidence,
        w_politeness=cfg.w_politeness,
    )
    reward_fn = make_proxy_reward_fn(proxy_cfg)

    # TRL requires generation_batch_size % num_generations == 0.
    generation_batch_size = max(
        cfg.num_generations,
        cfg.per_device_train_batch_size * cfg.num_generations,
    )

    # TRL 0.x used max_prompt_length; TRL 1.7+ dropped it — filter handles that.
    raw_args: dict[str, Any] = {
        "output_dir": cfg.output_dir,
        "max_steps": cfg.max_steps,
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "num_generations": cfg.num_generations,
        "generation_batch_size": generation_batch_size,
        "max_completion_length": cfg.max_completion_length,
        "max_prompt_length": cfg.max_prompt_length,
        "learning_rate": cfg.learning_rate,
        "logging_steps": cfg.logging_steps,
        "save_strategy": "steps",
        "save_steps": cfg.save_steps,
        "save_total_limit": cfg.save_total_limit,
        "save_only_model": cfg.save_only_model,
        "seed": cfg.seed,
        "bf16": cfg.bf16,
        "fp16": cfg.fp16,
        "report_to": cfg.report_to,
        "remove_unused_columns": False,
        "chat_template_kwargs": {"enable_thinking": cfg.enable_thinking},
        "loss_type": cfg.loss_type,
        "beta": cfg.beta,
        "log_completions": True,
    }
    grpo_args = GRPOConfig(**_filter_kwargs(GRPOConfig, raw_args))

    peft_config = None
    if cfg.use_lora:
        peft_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )

    dataset = build_dataset(cfg.data_path)

    trainer_kwargs: dict[str, Any] = {
        "model": cfg.model_id,
        "reward_funcs": reward_fn,
        "args": grpo_args,
        "train_dataset": dataset,
        "peft_config": peft_config,
    }
    # TRL renamed tokenizer -> processing_class
    sig = inspect.signature(GRPOTrainer.__init__)
    if "processing_class" in sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in sig.parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = GRPOTrainer(**trainer_kwargs)
    return trainer


def save_run_config(cfg: Exp3TrainConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")


def train(cfg: Exp3TrainConfig | None = None) -> dict[str, Any]:
    cfg = cfg or Exp3TrainConfig()
    if not cfg.output_dir:
        cfg.output_dir = default_train_output_dir()
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    trainer = build_trainer(cfg)
    is_main = trainer.accelerator.is_main_process
    if is_main:
        save_run_config(cfg, out / "train_config.json")
        n = trainer.accelerator.num_processes
        print(f"GRPO training with {n} process(es) / GPU(s)")

    result = trainer.train()
    # all ranks must participate in save for sharded state; final adapter on main
    trainer.save_model(str(out / "final"))
    trainer.accelerator.wait_for_everyone()

    metrics = dict(result.metrics) if result is not None else {}
    if is_main:
        (out / "train_metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
    return metrics
