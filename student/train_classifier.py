"""Experiment 1: the pure classifier baseline -- Query -> route_head ->
CrossEntropy(teacher_route). No reasoning anywhere: no reasoning text in
the data, no LM loss, no lm_head compute at all. This is the "pooled
baseline" DHRD (see student/train_dual.py's module docstring) itself
compares against, and the control this project's experiment matrix needs
to answer "does rationale supervision improve routing?" -- Experiment 2
(student/train_dual.py) adds reasoning as an auxiliary training signal;
this is what routing accuracy looks like with none.

Shares student/route_head.py's exact MLP head (LayerNorm -> Linear(h,h)
-> GELU -> Dropout -> Linear(h,3)) with train_dual.py on purpose -- the
experiment comparison is only about reasoning supervision if the
classifier itself is identical everywhere it's used.

One real difference from train_dual.py, not just a smaller dataset:
forward() calls self.lm.model(...) (the bare trunk) instead of
self.lm(...) (the full CausalLM). There's no reasoning target here, so
there's nothing that needs lm_head's logits -- and lm_head is not cheap
to skip: on google/gemma-3-270m-it, lm_head is 167.8M of the model's
268.1M params (62.6%, tied to the input embedding) because vocab_size is
262144 against a hidden_size of only 640. Measured: skipping it cuts
forward latency ~41% on a training batch, and the two paths produce
bit-for-bit identical pooled hidden states (verified: same trunk, same
weights, lm_head input never touches route_head's input).
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
    DEFAULT_STUDENT_MODEL_ID, ROUTE_LABEL_SPACES, ROUTING_LABELS, STUDENT_DIR, map_route,
)
from common.schema import read_jsonl
from student.data import DistillationExample, oversample_minority_tiers, _STUDENT_INSTRUCTION
from student.route_head import SaveEpochCheckpointCallback, build_route_head

__all__ = [
    "DEFAULT_STUDENT_MODEL_ID",
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
        routing_labels: tuple[str, ...] = ROUTING_LABELS,
        route_head_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.lm = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
        hidden_size = self.lm.config.hidden_size
        self.routing_labels = tuple(routing_labels)
        self.route_head = build_route_head(hidden_size, len(self.routing_labels), route_head_dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        route_labels: Optional[torch.Tensor] = None,
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
            if route_labels is not None:
                loss = F.cross_entropy(route_logits, route_labels)

        return ClassifierOutput(loss=loss, route_logits=route_logits)


class ClassifierDataset(Dataset):
    """One tokenized record per DistillationExample: input_ids (prompt
    only, no assistant turn -- there's no reasoning to teacher-force),
    route_position, route_label. routing_reasoning is never read."""

    def __init__(
        self,
        examples: list[DistillationExample],
        tokenizer: PreTrainedTokenizerBase,
        routing_labels: tuple[str, ...] = ROUTING_LABELS,
    ) -> None:
        self.tokenizer = tokenizer
        self.routing_labels = tuple(routing_labels)
        self.records = [self._encode(ex) for ex in examples]

    def _encode(self, example: DistillationExample) -> dict:
        messages = [{"role": "user", "content": _STUDENT_INSTRUCTION.format(query=example.query)}]
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True,
        )["input_ids"]
        route = map_route(example.teacher_route, self.routing_labels)
        return {
            "input_ids": input_ids,
            "route_position": len(input_ids) - 1,  # last (only) token -- what generation would start from
            "route_label": self.routing_labels.index(route),
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
    input_ids, attention_mask, route_positions, route_labels = [], [], [], []
    for f in features:
        pad_len = max_len - len(f["input_ids"])
        input_ids.append(f["input_ids"] + [pad_token_id] * pad_len)
        attention_mask.append([1] * len(f["input_ids"]) + [0] * pad_len)
        route_positions.append(f["route_position"])
        route_labels.append(f["route_label"])
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "route_positions": torch.tensor(route_positions, dtype=torch.long),
        "route_labels": torch.tensor(route_labels, dtype=torch.long),
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

        route_labels = inputs.get("route_labels")
        if route_labels is not None and outputs.route_logits is not None:
            batch_size = route_labels.size(0)
            correct = (outputs.route_logits.argmax(dim=-1) == route_labels).sum().item()
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
    model_id: str = DEFAULT_STUDENT_MODEL_ID,
    split: str = "train",
    num_train_epochs: float = 3.0,
    learning_rate: float = 2e-5,
    per_device_train_batch_size: int = 8,
    warmup_ratio: float = 0.0,
    lr_scheduler_type: str = "linear",
    route_head_dropout: float = 0.1,
    oversample_ratio: float = 0.0,
    seed: int = 42,
    label_space: str = "3way",
) -> None:
    """Pools data/student/<dataset>/<output_version>/<split>.jsonl (same
    source file train_student_dual reads; routing_reasoning is present
    but unused here) across dataset_names, then trains a ClassifierRouter
    with CrossEntropy(route) only. See module docstring for why.

    seed is set here, before ClassifierRouter is constructed below -- see
    student/train_dual.py's train_student_dual() docstring for why
    TrainingArguments(seed=...) alone doesn't cover route_head's random
    init.

    oversample_ratio: 0.0 (default) leaves the pooled examples untouched
    -- existing results stay exactly reproducible. > 0 duplicates
    minority routing tiers toward the majority tier's count (see
    student/data.py's oversample_minority_tiers); applied after pooling,
    before tokenization, so nothing downstream needs to know it happened.
    Note: oversampling groups by the raw 3-way teacher_route even when
    label_space="binary", so it balances small/medium/large rather than
    cheap/large -- fine at the default 0.0 (off), but worth knowing before
    combining a non-zero oversample_ratio with label_space="binary".

    label_space: "3way" (default, reads data/student/<dataset>/<output_version>/
    <split>.jsonl, common.config.ROUTING_LABELS) or "binary" (reads
    data/student/<dataset>/<output_version>/binary/<split>.jsonl --
    student/data.py's build_binary_distillation_dataset()'s output, a
    separate materialized file, never the 3-way one -- run
    `build-binary-student-data` first if it doesn't exist yet;
    common.config.BINARY_ROUTING_LABELS, {small,medium}->cheap /
    large->large). ClassifierDataset still applies common.config.
    map_route() at tokenization time regardless -- a no-op when the file
    already matches routing_labels, a safety net if it doesn't.

    warmup_ratio / lr_scheduler_type: both default to Trainer's own
    defaults (0.0 = no warmup, "linear" decay across the full run) -- i.e.
    passing nothing reproduces every prior run in this project bit-for-bit.
    lr_scheduler_type's decay curve is computed against num_train_epochs *
    steps-per-epoch, so it's specific to whatever epoch budget this call
    uses -- see train_student_dual's docstring for the same caveat."""
    set_seed(seed)
    routing_labels = ROUTE_LABEL_SPACES[label_space]
    label_subdir = "" if label_space == "3way" else label_space
    examples: list[DistillationExample] = []
    for name in dataset_names:
        path = STUDENT_DIR / name / output_version / label_subdir / f"{split}.jsonl"
        examples.extend(read_jsonl(path, DistillationExample))
    print(f"Training classifier-only student on {len(examples)} examples from {dataset_names} "
          f"(label_space={label_space!r}, routing_labels={routing_labels})")

    if oversample_ratio > 0:
        before = collections.Counter(e.teacher_route for e in examples)
        examples = oversample_minority_tiers(examples, target_ratio=oversample_ratio)
        after = collections.Counter(e.teacher_route for e in examples)
        print(f"Oversampled (target_ratio={oversample_ratio}): {dict(before)} -> {dict(after)}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dataset = ClassifierDataset(examples, tokenizer, routing_labels=routing_labels)
    model = ClassifierRouter(model_id, routing_labels=routing_labels, route_head_dropout=route_head_dropout)

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
    predicted route (in model.routing_labels' label space -- 3-way or
    binary, whichever this checkpoint was trained on), the full softmax
    over that label space, and latency_seconds."""
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
