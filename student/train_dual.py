"""Reasoning-plus-route student: route head and language-model head on one
shared trunk (paper §3.6). The teacher's explanation is a training-only signal;
at inference only the route head runs, in a single forward pass.

    query -> trunk -> [last-token hidden state -> route_head -> L_route (routing_loss)]
                   -> [explanation tokens (teacher-forced) -> lm_head -> L_reason]
    L = lambda_route * L_route + lambda_reason * L_reason

Follows DHRD (Suganthan et al., 2025): same last-token pooling, MLP head and
combined loss. The head's LayerNorm is our addition: Gemma's hidden-state norms
(~180-200) otherwise blow up a freshly initialised Linear layer.

The route label is not part of the language-model target: the route head owns
routing. Per-epoch checkpoints are saved by SaveEpochCheckpointCallback
(student/route_head.py); the best is chosen by validation macro-F1.
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
    "DualHeadOutput",
    "DualHeadRouter",
    "DualHeadDataset",
    "collate_dual_batch",
    "train_student_dual",
    "save_dual_checkpoint",
    "load_dual_head_checkpoint",
    "predict_route",
]


@dataclass
class DualHeadOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    route_loss: Optional[torch.Tensor] = None
    reason_loss: Optional[torch.Tensor] = None
    route_logits: Optional[torch.Tensor] = None
    lm_logits: Optional[torch.Tensor] = None


class DualHeadRouter(nn.Module):
    """Base causal LM (untouched, owns reasoning) + an added MLP
    route_head (owns routing). Wraps AutoModelForCausalLM instead of
    subclassing it, so generation/save_pretrained keep working
    unmodified -- route_head is the only thing outside that contract
    (see save_dual_checkpoint/load_dual_head_checkpoint)."""

    def __init__(
        self,
        model_id: str,
        routing_labels: tuple[str, ...] = LABELS,
        lambda_route: float = 1.0,
        lambda_reason: float = 1.0,
        route_head_dropout: float = 0.1,
        loss_mode: str = ROUTE_LOSS,
    ) -> None:
        super().__init__()
        self.lm = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
        hidden_size = self.lm.config.hidden_size
        self.routing_labels = tuple(routing_labels)
        # See student/route_head.py for what this is and why it's shared with train_classifier.py.
        self.route_head = build_route_head(hidden_size, len(self.routing_labels), route_head_dropout)
        self.lambda_route = lambda_route
        self.lambda_reason = lambda_reason
        self.loss_mode = loss_mode

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        soft_large: Optional[torch.Tensor] = None,
        route_positions: Optional[torch.Tensor] = None,
    ) -> DualHeadOutput:
        outputs = self.lm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        lm_logits = outputs.logits
        hidden = outputs.hidden_states[-1]  # (batch, seq, hidden) -- what lm_head itself reads

        reason_loss = None
        if labels is not None:
            # .float() upcast before cross_entropy, same as route_loss's pooled hidden state
            # below and standard HF causal-LM practice -- bf16 cross-entropy over a
            # 262,144-token vocabulary loses precision in the log-softmax normalization otherwise.
            shift_logits = lm_logits[..., :-1, :].float().contiguous()
            shift_labels = labels[..., 1:].contiguous()
            reason_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100
            )

        route_logits = None
        route_loss = None
        if route_positions is not None:
            batch_idx = torch.arange(hidden.size(0), device=hidden.device)
            pooled = hidden[batch_idx, route_positions].float()  # (batch, hidden)
            route_logits = self.route_head(pooled)
            if soft_large is not None:
                route_loss = routing_loss(route_logits, soft_large, self.loss_mode)

        loss = None
        if reason_loss is not None and route_loss is not None:
            loss = self.lambda_route * route_loss + self.lambda_reason * reason_loss
        elif route_loss is not None:
            loss = route_loss
        elif reason_loss is not None:
            loss = reason_loss

        return DualHeadOutput(
            loss=loss, route_loss=route_loss, reason_loss=reason_loss,
            route_logits=route_logits, lm_logits=lm_logits,
        )


class DualHeadDataset(Dataset):
    """One tokenized record per DistillationExample: input_ids/labels for
    the reasoning LM loss, route_position/soft_large for the classifier
    loss. Built eagerly -- dataset sizes here are ~1-2k examples."""

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
        prompt_messages = [{"role": "user", "content": _STUDENT_INSTRUCTION.format(query=example.query)}]
        prompt_ids = self.tokenizer.apply_chat_template(
            prompt_messages, add_generation_prompt=True, tokenize=True, return_dict=True,
        )["input_ids"]

        # <ROUTE> deliberately left out -- see module docstring.
        full_messages = prompt_messages + [
            {"role": "assistant", "content": f"<REASON>\n{example.routing_reasoning}"}
        ]
        full_ids = self.tokenizer.apply_chat_template(
            full_messages, tokenize=True, return_dict=True,
        )["input_ids"]

        if full_ids[: len(prompt_ids)] != prompt_ids:
            # What: prompt_ids should be a prefix of full_ids. How this could fail: the chat
            # template re-tokenizes the prompt/assistant seam differently once the assistant
            # turn is appended (a BPE merge across the boundary), which would silently point
            # route_position at the wrong token. Need: fail loudly instead of mislabeling.
            raise ValueError(
                f"Chat template isn't prompt-prefix-stable for query={example.query!r}; "
                "route_position can't be trusted -- inspect the tokenizer's chat_template."
            )

        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        route_position = len(prompt_ids) - 1  # last prompt token == what generation would start from

        return {
            "input_ids": full_ids,
            "labels": labels,
            "route_position": route_position,
            "soft_large": example.soft_large,
        }

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        return self.records[idx]


def collate_dual_batch(features: list[dict], pad_token_id: int) -> dict:
    """Right-pads input_ids/attention_mask/labels to batch max length.
    route_position needs no adjustment -- it always points inside the
    unpadded prefix, before any padding is appended."""
    max_len = max(len(f["input_ids"]) for f in features)
    input_ids, attention_mask, labels, route_positions, soft_large = [], [], [], [], []
    for f in features:
        pad_len = max_len - len(f["input_ids"])
        input_ids.append(f["input_ids"] + [pad_token_id] * pad_len)
        attention_mask.append([1] * len(f["input_ids"]) + [0] * pad_len)
        labels.append(f["labels"] + [-100] * pad_len)
        route_positions.append(f["route_position"])
        soft_large.append(f["soft_large"])
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "route_positions": torch.tensor(route_positions, dtype=torch.long),
        "soft_large": torch.tensor(soft_large, dtype=torch.float32),
    }


class DualHeadTrainer(Trainer):
    """Logs route_accuracy alongside route_loss/reason_loss/loss (total),
    once per epoch. Need: reason_loss dropping only proves the auxiliary
    task is regularizing, not that it's helping the route decision --
    only watching route_accuracy move in the same log line shows that.

    How: compute_loss() accumulates per-step stats (weighted by batch
    size for route_loss/accuracy, by non-masked-token count for
    reason_loss). log() -- called once per epoch under
    logging_strategy="epoch" -- folds the accumulated averages into the
    log line Trainer was already emitting, then resets."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._reset_epoch_stats()

    def _reset_epoch_stats(self) -> None:
        self._epoch_route_correct = 0
        self._epoch_route_total = 0
        self._epoch_route_loss_sum = 0.0        # sum of route_loss * batch_size
        self._epoch_reason_loss_sum = 0.0        # sum of reason_loss * n_reason_tokens
        self._epoch_reason_tokens = 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)

        soft_large = inputs.get("soft_large")
        if soft_large is not None and outputs.route_logits is not None:
            batch_size = soft_large.size(0)
            majority = (soft_large > 0.5).long()
            correct = (outputs.route_logits.argmax(dim=-1) == majority).sum().item()
            self._epoch_route_correct += correct
            self._epoch_route_total += batch_size
            self._epoch_route_loss_sum += outputs.route_loss.item() * batch_size

        labels = inputs.get("labels")
        if labels is not None and outputs.reason_loss is not None:
            n_reason_tokens = (labels[..., 1:] != -100).sum().item()  # matches forward()'s shift
            self._epoch_reason_loss_sum += outputs.reason_loss.item() * n_reason_tokens
            self._epoch_reason_tokens += n_reason_tokens

        return (outputs.loss, outputs) if return_outputs else outputs.loss

    def log(self, logs: dict, start_time: Optional[float] = None) -> None:
        # "loss" = Trainer's own per-epoch progress log (already total_loss). The end-of-training
        # summary log carries "train_loss" instead, so this skips injecting/resetting on that one.
        if "loss" in logs and self._epoch_route_total > 0:
            logs["route_accuracy"] = round(self._epoch_route_correct / self._epoch_route_total, 4)
            logs["route_loss"] = round(self._epoch_route_loss_sum / self._epoch_route_total, 4)
            logs["reason_loss"] = round(self._epoch_reason_loss_sum / max(self._epoch_reason_tokens, 1), 4)
            self._reset_epoch_stats()
        super().log(logs, start_time)


def train_student_dual(
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
    lambda_route: float = LAMBDA_ROUTE,
    lambda_reason: float = LAMBDA_REASON,
    route_head_dropout: float = ROUTE_HEAD_DROPOUT,
    oversample_ratio: float = OVERSAMPLE_RATIO,
    loss_mode: str = ROUTE_LOSS,
    seed: int = 42,
) -> None:
    """Train a reasoning-plus-route student on data/student/<dataset>/<output_version>/<split>.jsonl
    ({query, teacher_route, routing_reasoning}), pooled across dataset_names.
    Defaults come from common/config.py.

    The seed is set before the model is built so the route head's random
    initialisation is reproducible (Trainer's own seed is applied too late;
    unseeded runs differed noticeably in validation macro-F1).
    oversample_ratio > 0 duplicates the rarer label, including its explanation.
    """
    set_seed(seed)
    routing_labels = LABELS
    examples: list[DistillationExample] = []
    for name in dataset_names:
        path = STUDENT_DIR / name / output_version / f"{split}.jsonl"
        examples.extend(read_jsonl(path, DistillationExample))
    print(f"Training dual-head student on {len(examples)} examples from {dataset_names} "
          f"(routing_labels={routing_labels}, route loss={loss_mode})")

    if oversample_ratio > 0:
        before = collections.Counter(e.teacher_route for e in examples)
        examples = oversample_minority_tiers(examples, target_ratio=oversample_ratio)
        after = collections.Counter(e.teacher_route for e in examples)
        print(f"Oversampled (target_ratio={oversample_ratio}): {dict(before)} -> {dict(after)}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dataset = DualHeadDataset(examples, tokenizer, routing_labels=routing_labels)
    model = DualHeadRouter(
        model_id, routing_labels=routing_labels,
        lambda_route=lambda_route, lambda_reason=lambda_reason, route_head_dropout=route_head_dropout,
        loss_mode=loss_mode,
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
        logging_strategy="epoch",  # DualHeadTrainer.log() piggybacks on this cadence
        save_strategy="no",  # Trainer's own checkpointing would dump a raw state_dict (not a
                              # PreTrainedModel) -- SaveEpochCheckpointCallback below does the real save
        report_to="none",
        remove_unused_columns=False,  # plain torch Dataset -- forward()'s signature is already the contract
    )

    trainer = DualHeadTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=partial(collate_dual_batch, pad_token_id=tokenizer.pad_token_id),
        callbacks=[SaveEpochCheckpointCallback(model, tokenizer, checkpoint_dir, save_dual_checkpoint)],
    )
    trainer.train()

    save_dual_checkpoint(model, tokenizer, checkpoint_dir)
    print(f"Saved dual-head student checkpoint -> {checkpoint_dir}")
    print(f"Per-epoch checkpoints -> {checkpoint_dir}/checkpoint-<step>/ "
          f"(use select-router-checkpoint to pick the best by validation macro F1)")


def save_dual_checkpoint(model: DualHeadRouter, tokenizer: PreTrainedTokenizerBase, checkpoint_dir: Path) -> None:
    """Saves the base LM in standard HF format, route_head's weights
    separately (route_head.pt -- not part of the base model's own
    state_dict), and a small json with what load_dual_head_checkpoint()
    needs to reconstruct the same DualHeadRouter."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.lm.save_pretrained(str(checkpoint_dir))
    tokenizer.save_pretrained(str(checkpoint_dir))
    torch.save(model.route_head.state_dict(), checkpoint_dir / "route_head.pt")
    (checkpoint_dir / "dual_head_config.json").write_text(json.dumps({
        "routing_labels": list(model.routing_labels),
        "lambda_route": model.lambda_route,
        "lambda_reason": model.lambda_reason,
        "route_head_dropout": model.route_head[3].p,
    }, indent=2))
    print(f"  checkpoint saved -> {checkpoint_dir}")


def load_dual_head_checkpoint(checkpoint_dir: str) -> tuple[DualHeadRouter, PreTrainedTokenizerBase]:
    """Inverse of save_dual_checkpoint() -- rebuilds a DualHeadRouter on
    GPU, ready for predict_route()."""
    checkpoint_dir = Path(checkpoint_dir)
    config = json.loads((checkpoint_dir / "dual_head_config.json").read_text())

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))
    model = DualHeadRouter(
        str(checkpoint_dir),
        routing_labels=tuple(config["routing_labels"]),
        lambda_route=config["lambda_route"],
        lambda_reason=config["lambda_reason"],
        route_head_dropout=config.get("route_head_dropout", 0.1),
    )
    route_head_state = torch.load(checkpoint_dir / "route_head.pt", map_location="cpu")
    model.route_head.load_state_dict(route_head_state)
    model.to("cuda")
    model.eval()
    print(f"Loaded dual-head checkpoint from {checkpoint_dir} (routing_labels={model.routing_labels})")
    return model, tokenizer


def predict_route(
    model: DualHeadRouter, tokenizer: PreTrainedTokenizerBase, query: str, device: str = "cuda",
) -> dict:
    """One forward pass, no generation -- the entire point of this
    variant. Returns the predicted route (small or large), the full
    softmax over the labels, and latency_seconds for comparison against
    the generative baseline."""
    messages = [{"role": "user", "content": _STUDENT_INSTRUCTION.format(query=query)}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
    ).to(device)

    start_time = time.monotonic()
    with torch.no_grad():
        outputs = model.lm(**inputs, output_hidden_states=True)
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
