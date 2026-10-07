"""The route classifier head, shared by every student training variant
that has one (student/train_dual.py, student/train_classifier.py).

What: LayerNorm -> Linear(h,h) -> GELU -> Dropout -> Linear(h,num_labels),
matching DHRD's (Suganthan et al., 2025, https://arxiv.org/abs/2509.21487)
"pooling-MLP" classifier head. LayerNorm is the one addition beyond DHRD's
published design -- Gemma's hidden states run at a large scale (~180-200
norm for hidden_size=640) that overwhelms a freshly initialized Linear at
init; this renormalizes first, the same way a transformer block
renormalizes its own residual stream before reading it.

Why shared: this project's experiments compare "with reasoning
supervision" (train_dual.py) against "without" (train_classifier.py) --
that comparison is only meaningful if the classifier head itself is
identical between them. One builder function is what guarantees that,
instead of two copies that could quietly drift apart.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase, TrainerCallback

from common.config import LABELS

assert LABELS == ("small", "large"), "routing_loss builds targets in this column order"


def build_route_head(hidden_size: int, num_labels: int, dropout: float = 0.1) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(hidden_size, dtype=torch.float32),
        nn.Linear(hidden_size, hidden_size, dtype=torch.float32),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_size, num_labels, dtype=torch.float32),
    )


def routing_loss(route_logits: torch.Tensor, soft_large: torch.Tensor, mode: str = "soft") -> torch.Tensor:
    """Routing loss against the teacher's votes. soft_large = share of votes for "large".

    soft:      cross-entropy against the vote share (3 of 5 votes -> target [0.4, 0.6])
    hard:      cross-entropy against the majority label only (target [0, 1])
    hard+soft: cross-entropy on the majority + KL on the vote share (Wu et al. 2026)
    """
    soft_target = torch.stack([1 - soft_large, soft_large], dim=-1)  # [P(small), P(large)]
    hard_target = (soft_large > 0.5).long()                          # 0 = small, 1 = large
    if mode == "soft":
        return F.cross_entropy(route_logits, soft_target)
    if mode == "hard":
        return F.cross_entropy(route_logits, hard_target)
    if mode == "hard+soft":
        kl = F.kl_div(F.log_softmax(route_logits, dim=-1), soft_target, reduction="batchmean")
        return F.cross_entropy(route_logits, hard_target) + kl
    raise ValueError(f"Unknown route loss mode {mode!r}: use soft, hard or hard+soft")


class SaveEpochCheckpointCallback(TrainerCallback):
    """Saves a real, loadable checkpoint (via save_fn -- save_classifier_
    checkpoint or save_dual_checkpoint) at the end of every epoch, into
    checkpoint_dir/checkpoint-<global_step>/. Needed because
    ClassifierRouter/DualHeadRouter aren't PreTrainedModel, so Trainer's
    own save_strategy="epoch" would just torch.save a raw state_dict --
    not something load_classifier_checkpoint()/load_dual_head_checkpoint()
    or evaluation/router_eval.py's select_best_router_checkpoint() (which
    globs checkpoint-<step>) could do anything with."""

    def __init__(self, model: nn.Module, tokenizer: PreTrainedTokenizerBase, checkpoint_dir: Path, save_fn: Callable):
        self.model = model
        self.tokenizer = tokenizer
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_fn = save_fn

    def on_epoch_end(self, args, state, control, **kwargs):
        step_dir = self.checkpoint_dir / f"checkpoint-{int(state.global_step)}"
        self.save_fn(self.model, self.tokenizer, step_dir)
        return control
