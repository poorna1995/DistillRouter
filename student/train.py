"""Trains the student router: a small sequence classifier (3-way
small/medium/large) fine-tuned on the teacher's routing labels.

Hard-label supervised fine-tuning, not soft-target distillation — no real
soft probability distribution is available to distill from (see
teacher/qwen_teacher.py's module docstring for why). This matches
DISTILLROUTER_SPEC.md's stated plan ("hard routing labels ... cross-entropy
classification loss").

Sequence classification head (AutoModelForSequenceClassification), not
generative decoding — one forward pass, no autoregressive generation loop,
which is the whole point: the project's stated problem is routing latency,
and a classification head is the cheapest possible way to read off a label.

Class-weighted loss: the training labels are imbalanced (~51% large / 32%
medium / 17% small, from the teacher's own predictions on train). An
earlier unweighted run didn't just learn that imbalance — it amplified it,
predicting large 62% of the time and small only 8% (measured against real
oracle ground truth; see conversation history). Plain cross-entropy lets a
model shortcut to low average loss by leaning on the majority class harder
than the data even requires; weighting the loss inversely by class
frequency (the standard "balanced" scheme) removes that shortcut by
penalizing mistakes on the rare classes (small) more heavily.

Evaluated here against the teacher's own labels on held-out validation
(immediately available) — NOT against oracle ground truth. "Student
matches teacher" and "student routes correctly" are different questions;
this script's eval loop only answers the first one — see
student/benchmark.py for the oracle-grounded comparison.

Usage: python -m student.train
"""
from __future__ import annotations

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from common.config import ID_TO_LABEL, ROUTING_LABELS
from dataset import available_datasets
from student.data import load_labeled_queries

STUDENT_MODEL_ID = "google/gemma-3-270m"
OUTPUT_DIR = "data/student/gemma-3-270m-router-weighted"
MAX_LENGTH = 256


def _compute_class_weights(rows: list[dict]) -> torch.Tensor:
    """Standard "balanced" class weights: total / (num_classes * count_c).
    Computed from the actual training rows, not hardcoded, so it stays
    correct if the training data changes.
    """
    counts = [0] * len(ROUTING_LABELS)
    for row in rows:
        counts[row["label_id"]] += 1
    total = len(rows)
    weights = [total / (len(ROUTING_LABELS) * count) if count > 0 else 0.0 for count in counts]
    return torch.tensor(weights, dtype=torch.float32)


class WeightedLossTrainer(Trainer):
    def __init__(self, *args, class_weights: torch.Tensor, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        # CrossEntropyLoss requires weight and logits to share a dtype; the
        # model's logits come out bfloat16 by default, class_weights was
        # built in plain float32.
        weight = self.class_weights.to(device=logits.device, dtype=logits.dtype)
        loss = torch.nn.functional.cross_entropy(logits, labels, weight=weight)
        return (loss, outputs) if return_outputs else loss


def _tokenize_dataset(rows: list[dict], tokenizer) -> Dataset:
    ds = Dataset.from_list(rows)

    def tokenize(batch):
        encoded = tokenizer(batch["query"], truncation=True, max_length=MAX_LENGTH)
        encoded["labels"] = batch["label_id"]
        return encoded

    return ds.map(tokenize, batched=True, remove_columns=ds.column_names)


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()

    per_class = {}
    for label_id, label_name in ID_TO_LABEL.items():
        true_positive = int(((predictions == label_id) & (labels == label_id)).sum())
        predicted_positive = int((predictions == label_id).sum())
        actual_positive = int((labels == label_id).sum())
        precision = true_positive / predicted_positive if predicted_positive > 0 else 0.0
        recall = true_positive / actual_positive if actual_positive > 0 else 0.0
        per_class[f"{label_name}_precision"] = round(precision, 4)
        per_class[f"{label_name}_recall"] = round(recall, 4)
        per_class[f"{label_name}_support"] = actual_positive

    return {"accuracy": round(float(accuracy), 4), **per_class}


def main() -> None:
    dataset_names = available_datasets()
    train_rows = load_labeled_queries(dataset_names, "train")
    eval_rows = load_labeled_queries(dataset_names, "validation")
    print(f"train examples: {len(train_rows)}, eval examples: {len(eval_rows)}")

    class_weights = _compute_class_weights(train_rows)
    print("class weights (small, medium, large):", class_weights.tolist())

    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = _tokenize_dataset(train_rows, tokenizer)
    eval_dataset = _tokenize_dataset(eval_rows, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(STUDENT_MODEL_ID, num_labels=len(ROUTING_LABELS))
    model.config.pad_token_id = tokenizer.pad_token_id

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=20,
        report_to=[],
    )

    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=_compute_metrics,
        class_weights=class_weights,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("Final validation metrics (student vs. teacher labels):", metrics)

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved student model to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
