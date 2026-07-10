"""Collect residual-stream activations for linear probes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ActivationBundle:
    """Activations and metadata for one completion."""

    prompt_id: str
    layer_means: dict[int, np.ndarray]  # layer -> (hidden,)
    last_prompt: dict[int, np.ndarray]  # layer -> (hidden,) early-warning site
    n_prompt_tokens: int
    n_completion_tokens: int


class ResidualCapture:
    def __init__(self):
        self.by_layer: dict[int, list[torch.Tensor]] = {}
        self._handles = []

    def clear(self) -> None:
        self.by_layer = {}

    def attach(self, model) -> None:
        self.remove()
        layers = model.model.layers  # Qwen / Llama-style

        def make_hook(idx: int):
            def hook(_module, _inp, out):
                hidden = out[0] if isinstance(out, tuple) else out
                # store on CPU float16 to save RAM
                self.by_layer.setdefault(idx, []).append(
                    hidden.detach().to(dtype=torch.float16, device="cpu")
                )

            return hook

        for i, layer in enumerate(layers):
            self._handles.append(layer.register_forward_hook(make_hook(i)))

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []


def load_policy(
    model_id: str,
    adapter_path: str | None = None,
    device: str | None = None,
    dtype: str = "float16",
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(dtype, torch.float16)
    if device == "cpu":
        torch_dtype = torch.float32

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map=device,
    )
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

    model.eval()
    return model, tok


@torch.inference_mode()
def collect_for_pair(
    model,
    tokenizer,
    capture: ResidualCapture,
    prompt: str,
    completion: str,
    prompt_id: str = "",
    max_length: int = 1024,
    enable_thinking: bool = False,
) -> ActivationBundle:
    from probes_rh.chat import apply_chat_prompt, strip_think

    prompt_text = apply_chat_prompt(
        tokenizer, prompt, enable_thinking=enable_thinking
    )
    completion = strip_think(completion)

    full_text = prompt_text + completion
    enc_prompt = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    enc_full = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    n_prompt = int(enc_prompt["input_ids"].shape[1])
    input_ids = enc_full["input_ids"].to(model.device)
    attention_mask = enc_full["attention_mask"].to(model.device)

    capture.clear()
    capture.attach(model)
    _ = model(input_ids=input_ids, attention_mask=attention_mask)
    capture.remove()

    layer_means: dict[int, np.ndarray] = {}
    last_prompt: dict[int, np.ndarray] = {}
    for layer, tensors in capture.by_layer.items():
        # one forward -> one tensor [1, seq, hidden]
        h = tensors[-1][0]  # [seq, hidden]
        seq_len = h.shape[0]
        n_p = min(n_prompt, seq_len)
        if n_p < seq_len:
            comp = h[n_p:]
            layer_means[layer] = comp.float().mean(dim=0).numpy()
        else:
            layer_means[layer] = h.float().mean(dim=0).numpy()
        last_prompt[layer] = h[max(n_p - 1, 0)].float().numpy()

    n_comp = max(int(enc_full["input_ids"].shape[1]) - n_prompt, 0)
    return ActivationBundle(
        prompt_id=prompt_id,
        layer_means=layer_means,
        last_prompt=last_prompt,
        n_prompt_tokens=n_prompt,
        n_completion_tokens=n_comp,
    )


def bundle_to_dict(b: ActivationBundle) -> dict[str, Any]:
    return {
        "prompt_id": b.prompt_id,
        "n_prompt_tokens": b.n_prompt_tokens,
        "n_completion_tokens": b.n_completion_tokens,
        "layer_means": {str(k): v.astype(np.float16) for k, v in b.layer_means.items()},
        "last_prompt": {str(k): v.astype(np.float16) for k, v in b.last_prompt.items()},
    }
