"""Frozen open reward model used only as offline gold (never in GRPO)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# Strong small-ish open RM that fits on one Kaggle T4 for offline scoring.
DEFAULT_GOLD_RM = "Skywork/Skywork-Reward-V2-Llama-3.2-3B"


@dataclass
class GoldRMConfig:
    model_id: str = DEFAULT_GOLD_RM
    max_length: int = 1024
    batch_size: int = 4
    device: str | None = None
    dtype: str = "float16"
    trust_remote_code: bool = True


class GoldRewardModel:
    """Sequence-classification style RM: higher scalar = better (gold quality)."""

    def __init__(self, config: GoldRMConfig | None = None):
        self.config = config or GoldRMConfig()
        device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)

        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }.get(self.config.dtype, torch.float16)
        if self.device.type == "cpu":
            dtype = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            trust_remote_code=self.config.trust_remote_code,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_id,
            torch_dtype=dtype,
            trust_remote_code=self.config.trust_remote_code,
        )
        self.model.to(self.device)
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _format_pair(self, prompt: str, completion: str) -> str:
        # Skywork / Llama-style chat conversation as a single scored string.
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ]
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            return f"User: {prompt}\n\nAssistant: {completion}"

    @torch.inference_mode()
    def score_pairs(
        self,
        prompts: Sequence[str],
        completions: Sequence[str],
    ) -> list[float]:
        assert len(prompts) == len(completions)
        scores: list[float] = []
        bs = self.config.batch_size
        for i in range(0, len(prompts), bs):
            batch_p = prompts[i : i + bs]
            batch_c = completions[i : i + bs]
            texts = [self._format_pair(p, c) for p, c in zip(batch_p, batch_c)]
            enc = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            out = self.model(**enc)
            logits = out.logits
            # [B, 1] or [B]
            if logits.ndim == 2 and logits.size(-1) == 1:
                batch_scores = logits.squeeze(-1)
            elif logits.ndim == 2:
                # some RMs put score in last logit dim; take first column
                batch_scores = logits[:, 0]
            else:
                batch_scores = logits.view(-1)
            scores.extend(batch_scores.float().cpu().tolist())
        return [float(s) for s in scores]

    def close(self) -> None:
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
