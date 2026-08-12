"""Student SFT via TRL's SFTTrainer — imitates the teacher's routing
behavior. See student/data.py for the training-record shape (reasoning
before route, or route-only for the ablation) and why the instruction is
fixed across every example.
"""
from __future__ import annotations

from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from common.config import STUDENT_DIR

DEFAULT_STUDENT_MODEL_ID = "google/gemma-3-270m-it"


def train_student(
    dataset_names: list[str],
    output_version: str,
    checkpoint_dir: Path,
    model_id: str = DEFAULT_STUDENT_MODEL_ID,
    variant: str = "sft",  # "sft" (reasoning+route baseline) or "sft-route-only" (ablation)
    split: str = "train",
    num_train_epochs: float = 3.0,
    learning_rate: float = 2e-5,
    per_device_train_batch_size: int = 8,
) -> None:
    """Pools data/student/<dataset>/<output_version>/<split>.<variant>.jsonl
    across dataset_names into one training set. Loss is computed only on
    assistant turns (TRL's assistant_only_loss) — the query prompt is
    never backpropagated through (completion-only SFT)."""
    data_files = [str(STUDENT_DIR / name / output_version / f"{split}.{variant}.jsonl") for name in dataset_names]
    train_dataset = load_dataset("json", data_files=data_files, split="train")
    print(f"Training on {len(train_dataset)} examples from {dataset_names} (variant={variant!r})")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)

    config = SFTConfig(
        output_dir=str(checkpoint_dir),
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        assistant_only_loss=True,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = SFTTrainer(model=model, args=config, train_dataset=train_dataset, processing_class=tokenizer)
    trainer.train()

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(checkpoint_dir))
    tokenizer.save_pretrained(str(checkpoint_dir))
    print(f"Saved student checkpoint -> {checkpoint_dir}")
