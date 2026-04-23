"""Load a HuggingFace causal-LM + tokenizer for activation extraction."""

from __future__ import annotations

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def choose_device() -> str:
    """Pick the best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model_and_tokenizer(
    model_name: str,
    device: str,
    hf_token: str | None = None,
) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Load tokenizer + model from HF, move model to device, set eval mode.

    Uses fp16 on non-CPU devices, fp32 on CPU. `hf_token` is optional; HF will
    raise a clear error itself if the model is gated and no token is supplied.
    Environment variable `HF_TOKEN` is used as a fallback.
    """
    if hf_token is None:
        hf_token = os.environ.get("HF_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
    )
    model.eval()
    model.to(device)
    return tokenizer, model
