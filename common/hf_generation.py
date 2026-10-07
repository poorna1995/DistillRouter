"""Shared HF causal-LM loading + chat generation. Used by
candidate/base.py and both teacher/qwen_teacher_v*.py."""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


@dataclass
class GenerationResult:
    text: str
    latency_seconds: float
    input_tokens: int
    output_tokens: int


def load_causal_lm(model_id: str) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """Load a tokenizer + causal LM onto GPU. The expensive, one-time part."""
    print(f"Loading {model_id} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to("cuda")
    print(f"Loaded {model_id} onto {model.device}")
    return tokenizer, model


def generate_chat_response(
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    messages: list[dict],
    max_new_tokens: int,
) -> GenerationResult:
    """Apply the chat template, generate greedily, decode only the newly generated tokens."""
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to("cuda")

    start_time = time.monotonic()
    output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    latency_seconds = time.monotonic() - start_time

    input_length = inputs["input_ids"].shape[1]
    generated_tokens = output[0][input_length:]
    text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return GenerationResult(
        text=text,
        latency_seconds=latency_seconds,
        input_tokens=input_length,
        output_tokens=len(generated_tokens),
    )
