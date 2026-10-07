"""Route-only student: query -> shared trunk -> last-token hidden state ->
route_head -> routing_loss against the teacher's votes (paper §3.5;
student/route_head.py: soft, hard or hard+soft).

No reasoning text, no language-model loss. The route head is the same MLP as
the reasoning-plus-route student (student/route_head.py), so the two variants
differ only in reasoning supervision.

forward() runs only the trunk (self.lm.model), not the language-model head:
nothing here needs token logits, and on Gemma-3-270M the head is 62.6% of the
parameters (262,144-word vocabulary). Skipping it cuts forward time ~41%
without changing the hidden state the route head reads.
"""
from __future__ import annotations

import collections
import json
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase, Trainer, TrainingArguments, set_seed,
)
from transformers.utils import ModelOutput

from common.config import (
    BATCH_SIZE, STUDENT_MODEL_ID, EPOCHS, LABELS, LAMBDA_REASON, LAMBDA_ROUTE, LEARNING_RATE,
    LR_SCHEDULER, OVERSAMPLE_RATIO, ROUTE_HEAD_DROPOUT, ROUTE_LOSS, STUDENT_DIR, WARMUP_RATIO,
)
from common.schema import read_jsonl
from student.data import DistillationExample, oversample_minority_tiers, _STUDENT_INSTRUCTION
from student.route_head import SaveEpochCheckpointCallback, build_route_head, routing_loss

__all__ = [
    "STUDENT_MODEL_ID",
    "ClassifierOutput",
    "ClassifierRouter",
    "ClassifierDataset",
    "collate_classifier_batch",
    "train_student_classifier",
    "save_classifier_checkpoint",
    "load_classifier_checkpoint",
    "predict_route",
]


@dataclass
class ClassifierOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    route_logits: Optional[torch.Tensor] = None


class ClassifierRouter(nn.Module):
    """Trunk of a causal LM (lm_head never called) + route_head. Wraps
    AutoModelForCausalLM rather than AutoModel so the checkpoint format
    stays interoperable with train_dual.py's (both save/load through
    model.lm.save_pretrained) -- forward() is what actually skips lm_head,
    not the loading path."""

    def __init__(
        self,
        model_id: str,
        routing_labels: tuple[str, ...] = LABELS,
        route_head_dropout: float = 0.1,
        loss_mode: str = ROUTE_LOSS,
    ) -> None:
        super().__init__()
        self.lm = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
        hidden_size = self.lm.config.hidden_size
        self.routing_labels = tuple(routing_labels)
        self.route_head = build_route_head(hidden_size, len(self.routing_labels), route_head_dropout)
        self.loss_mode = loss_mode

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        soft_large: Optional[torch.Tensor] = None,
        route_positions: Optional[torch.Tensor] = None,
    ) -> ClassifierOutput:
        # self.lm.model = the bare trunk (see module docstring for why lm_head is skipped).
        outputs = self.lm.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]  # (batch, seq, hidden)

        route_logits = None
        loss = None
        if route_positions is not None:
            batch_idx = torch.arange(hidden.size(0), device=hidden.device)
            pooled = hidden[batch_idx, route_positions].float()  # (batch, hidden)
            route_logits = self.route_head(pooled)
            if soft_large is not None:
                loss = routing_loss(route_logits, soft_large, self.loss_mode)

        return ClassifierOutput(loss=loss, route_logits=route_logits)


class ClassifierDataset(Dataset):
    """One tokenized record per DistillationExample: input_ids (prompt
    only, no assistant turn -- there's no reasoning to teacher-force),
    route_position, soft_large. routing_reasoning is never read."""

    def __init__(
        self,
        examples: list[DistillationExample],
        tokenizer: PreTrainedTokenizerBase,
        routing_labels: tuple[str, ...] = LABELS,
    ) -> None:
        self.tokenizer = tokenizer
        self.routing_labels = tuple(routing_labels)
        self.records = [self._encode(ex) for ex in examples]

    def _encode(self, example: DistillationExample) -> dict:
        messages = [{"role": "user", "content": _STUDENT_INSTRUCTION.format(query=example.query)}]
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True,
        )["input_ids"]
        return {
            "input_ids": input_ids,
            "route_position": len(input_ids) - 1,  # last (only) token -- what generation would start from
            "soft_large": example.soft_large,
        }

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        return self.records[idx]


def collate_classifier_batch(features: list[dict], pad_token_id: int) -> dict:
    """Right-pads input_ids/attention_mask to batch max length.
    route_position needs no adjustment -- always inside the unpadded
    prefix, before any padding is appended."""
    max_len = max(len(f["input_ids"]) for f in features)
    input_ids, attention_mask, route_positions, soft_large = [], [], [], []
    for f in features:
        pad_len = max_len - len(f["input_ids"])
        input_ids.append(f["input_ids"] + [pad_token_id] * pad_len)
        attention_mask.append([1] * len(f["input_ids"]) + [0] * pad_len)
        route_positions.append(f["route_position"])
        soft_large.append(f["soft_large"])
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "route_positions": torch.tensor(route_positions, dtype=torch.long),
        "soft_large": torch.tensor(soft_large, dtype=torch.float32),
    }


class ClassifierTrainer(Trainer):
    """Logs route_accuracy alongside route_loss once per epoch. route_loss
    duplicates what Trainer already logs as "loss" here (there's only one
    loss term) -- kept as its own key anyway so a cross-experiment
    comparison can read "route_accuracy"/"route_loss" the same way
    regardless of which experiment produced the log, same schema as
    DualHeadTrainer in student/train_dual.py."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._reset_epoch_stats()

    def _reset_epoch_stats(self) -> None:
        self._epoch_correct = 0
        self._epoch_total = 0
        self._epoch_loss_sum = 0.0  # sum of loss * batch_size

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)

        soft_large = inputs.get("soft_large")
        if soft_large is not None and outputs.route_logits is not None:
            batch_size = soft_large.size(0)
            majority = (soft_large > 0.5).long()
            correct = (outputs.route_logits.argmax(dim=-1) == majority).sum().item()
            self._epoch_correct += correct
            self._epoch_total += batch_size
            self._epoch_loss_sum += outputs.loss.item() * batch_size

        return (outputs.loss, outputs) if return_outputs else outputs.loss

    def log(self, logs: dict, start_time: Optional[float] = None) -> None:
        # "loss" = Trainer's own per-epoch progress log. The end-of-training summary log
        # carries "train_loss" instead, so this skips injecting/resetting on that one.
        if "loss" in logs and self._epoch_total > 0:
            logs["route_accuracy"] = round(self._epoch_correct / self._epoch_total, 4)
            logs["route_loss"] = round(self._epoch_loss_sum / self._epoch_total, 4)
            self._reset_epoch_stats()
        super().log(logs, start_time)


def train_student_classifier(
    dataset_names: list[str],
    output_version: str,
    checkpoint_dir: Path,
    model_id: str = STUDENT_MODEL_ID,
    split: str = "train",
    num_train_epochs: float = EPOCHS,
    learning_rate: float = LEARNING_RATE,
    per_device_train_batch_size: int = BATCH_SIZE,
    warmup_ratio: float = WARMUP_RATIO,
    lr_scheduler_type: str = LR_SCHEDULER,
    route_head_dropout: float = ROUTE_HEAD_DROPOUT,
    oversample_ratio: float = OVERSAMPLE_RATIO,
    loss_mode: str = ROUTE_LOSS,
    seed: int = 42,
) -> None:
    """Train a route-only student on data/student/<dataset>/<output_version>/<split>.jsonl,
    pooled across dataset_names. Defaults come from common/config.py.

    The seed is set before the model is built so the route head's random
    initialisation is reproducible (Trainer's own seed is applied too late).
    oversample_ratio > 0 duplicates the rarer label before training (0 = off).
    """
    set_seed(seed)
    routing_labels = LABELS
    examples: list[DistillationExample] = []
    for name in dataset_names:
        path = STUDENT_DIR / name / output_version / f"{split}.jsonl"
        examples.extend(read_jsonl(path, DistillationExample))
    print(f"Training classifier-only student on {len(examples)} examples from {dataset_names} "
          f"(routing_labels={routing_labels}, route loss={loss_mode})")

    if oversample_ratio > 0:
        before = collections.Counter(e.teacher_route for e in examples)
        examples = oversample_minority_tiers(examples, target_ratio=oversample_ratio)
        after = collections.Counter(e.teacher_route for e in examples)
        print(f"Oversampled (target_ratio={oversample_ratio}): {dict(before)} -> {dict(after)}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dataset = ClassifierDataset(examples, tokenizer, routing_labels=routing_labels)
    model = ClassifierRouter(
        model_id, routing_labels=routing_labels, route_head_dropout=route_head_dropout, loss_mode=loss_mode,
    )

    args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        seed=seed,
        bf16=True,
        logging_strategy="epoch",  # ClassifierTrainer.log() piggybacks on this cadence
        save_strategy="no",  # Trainer's own checkpointing would dump a raw state_dict (not a
                              # PreTrainedModel) -- SaveEpochCheckpointCallback below does the real save
        report_to="none",
        remove_unused_columns=False,  # plain torch Dataset -- forward()'s signature is already the contract
    )

    trainer = ClassifierTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=partial(collate_classifier_batch, pad_token_id=tokenizer.pad_token_id),
        callbacks=[SaveEpochCheckpointCallback(model, tokenizer, checkpoint_dir, save_classifier_checkpoint)],
    )
    trainer.train()

    save_classifier_checkpoint(model, tokenizer, checkpoint_dir)
    print(f"Saved classifier-only student checkpoint -> {checkpoint_dir}")
    print(f"Per-epoch checkpoints -> {checkpoint_dir}/checkpoint-<step>/ "
          f"(use select-router-checkpoint to pick the best by validation macro F1)")


def save_classifier_checkpoint(
    model: ClassifierRouter, tokenizer: PreTrainedTokenizerBase, checkpoint_dir: Path
) -> None:
    """Saves the base LM in standard HF format, route_head's weights
    separately (route_head.pt), and a small json with what
    load_classifier_checkpoint() needs to reconstruct the model. Same
    layout as train_dual.py's save_dual_checkpoint()."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.lm.save_pretrained(str(checkpoint_dir))
    tokenizer.save_pretrained(str(checkpoint_dir))
    torch.save(model.route_head.state_dict(), checkpoint_dir / "route_head.pt")
    (checkpoint_dir / "classifier_config.json").write_text(json.dumps({
        "routing_labels": list(model.routing_labels),
        "route_head_dropout": model.route_head[3].p,
    }, indent=2))
    print(f"  checkpoint saved -> {checkpoint_dir}")


def load_classifier_checkpoint(checkpoint_dir: str) -> tuple[ClassifierRouter, PreTrainedTokenizerBase]:
    """Inverse of save_classifier_checkpoint() -- rebuilds a
    ClassifierRouter on GPU, ready for predict_route()."""
    checkpoint_dir = Path(checkpoint_dir)
    config = json.loads((checkpoint_dir / "classifier_config.json").read_text())

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))
    model = ClassifierRouter(
        str(checkpoint_dir),
        routing_labels=tuple(config["routing_labels"]),
        route_head_dropout=config.get("route_head_dropout", 0.1),
    )
    route_head_state = torch.load(checkpoint_dir / "route_head.pt", map_location="cpu")
    model.route_head.load_state_dict(route_head_state)
    model.to("cuda")
    model.eval()
    print(f"Loaded classifier checkpoint from {checkpoint_dir} (routing_labels={model.routing_labels})")
    return model, tokenizer


def predict_route(
    model: ClassifierRouter, tokenizer: PreTrainedTokenizerBase, query: str, device: str = "cuda",
) -> dict:
    """One trunk-only forward pass, no generation, no lm_head -- the
    fastest of the three experiments by construction. Returns the
    predicted route (small or large), the full softmax over the labels,
    and latency_seconds."""
    messages = [{"role": "user", "content": _STUDENT_INSTRUCTION.format(query=query)}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
    ).to(device)

    start_time = time.monotonic()
    with torch.no_grad():
        outputs = model.lm.model(**inputs, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1][0, -1].float()  # last token of the (single, unpadded) prompt
        route_logits = model.route_head(last_hidden.unsqueeze(0))
        probs = F.softmax(route_logits, dim=-1)[0]
    latency_seconds = time.monotonic() - start_time

    route_idx = int(probs.argmax())
    labels = model.routing_labels
    return {
        "route": labels[route_idx],
        "probabilities": {label: round(float(p), 4) for label, p in zip(labels, probs)},
        "latency_seconds": latency_seconds,
    }
