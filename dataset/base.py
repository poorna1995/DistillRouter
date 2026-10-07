"""Dataset loaders: turn a benchmark into train / validation / calibration / test files.

Each dataset (gsm8k.py, math.py, gsm_symbolic.py) only says how to load its
questions. This file does the rest, the same way for every dataset:

    1. load the questions and drop any whose reference answer cannot be graded
    2. carve validation and calibration out of train
    3. cap each split to its size (common/config.py), keeping the difficulty/subject mix
    4. write data/processed/<name>/<split>.jsonl and split_manifest.json
    5. check that no two splits share a question
"""
from __future__ import annotations

import hashlib
import json
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Callable, Iterable, Optional

from math_verify import parse

from common.config import CALIBRATION_SIZE, PROCESSED_DIR, RAW_DIR, SAMPLE_SIZE_CAPS, SEED, VAL_FRACTION
from common.schema import Example, write_jsonl
from common.split_integrity import check_dataset_splits

Splits = dict[str, list[Example]]


# --- Registry -------------------------------------------------------------------

_REGISTRY: dict[str, type["BaseDatasetLoader"]] = {}


def register(cls: type["BaseDatasetLoader"]) -> type["BaseDatasetLoader"]:
    _REGISTRY[cls.name] = cls
    return cls


def get_loader_class(name: str) -> type["BaseDatasetLoader"]:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown dataset '{name}'. Available: {', '.join(sorted(_REGISTRY))}")
    return _REGISTRY[name]


def datasets() -> list[str]:
    """Every dataset, including test-only ones."""
    return sorted(_REGISTRY)


def training_datasets() -> list[str]:
    """Datasets with a train split (the ones students learn from)."""
    return sorted(name for name, cls in _REGISTRY.items() if not cls.test_only)


# --- Loader ---------------------------------------------------------------------

class BaseDatasetLoader(ABC):
    name: str
    test_only: bool = False   # True: the dataset has only a test split (e.g. GSM-Symbolic)

    def __init__(self, raw_dir: Path = RAW_DIR, processed_dir: Path = PROCESSED_DIR) -> None:
        self.raw_dir = raw_dir / self.name
        self.processed_dir = processed_dir / self.name
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def load_native_splits(self) -> Splits:
        """The source's own splits: {"train": [...], "test": [...]}, or just {"test": [...]}."""

    def cached_rows(self, filename: str, fetch: Callable[[], Iterable[dict]]) -> list[dict]:
        """Rows of data/raw/<name>/<filename>, downloaded once with `fetch`."""
        path = self.raw_dir / filename
        if not path.exists():
            with path.open("w", encoding="utf-8") as f:
                for row in fetch():
                    f.write(json.dumps(row) + "\n")
        with path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    def prepare(self) -> dict[str, int]:
        """Run steps 1-5 above. Returns {split: number of questions}."""
        splits = self._load_gradable()                                          # 1
        if not self.test_only:
            splits = _carve_from_train(splits)                                  # 2
        splits = {split: _cap(examples, SAMPLE_SIZE_CAPS.get(split))            # 3
                  for split, examples in splits.items()}
        self._write(splits)                                                     # 4
        check_dataset_splits(self.name, self.processed_dir.parent)              # 5

        counts = {split: len(examples) for split, examples in splits.items()}
        print(f"[{self.name}] wrote {counts} -> {self.processed_dir}")
        return counts

    def _load_gradable(self) -> Splits:
        """Native splits, without questions whose reference answer math-verify cannot read."""
        splits = {}
        for split, examples in self.load_native_splits().items():
            splits[split] = [ex for ex in examples if parse(f"${ex.reference_answer}$")]
            dropped = len(examples) - len(splits[split])
            if dropped:
                print(f"[{self.name}] dropped {dropped} {split} questions with unreadable reference answers")
        return splits

    def _write(self, splits: Splits) -> None:
        """One .jsonl file per split, plus split_manifest.json: each split's size and a
        checksum of its question ids. Committing the manifest seals the splits."""
        manifest = {"seed": SEED, "splits": {}}
        for split, examples in splits.items():
            write_jsonl(self.processed_dir / f"{split}.jsonl", examples)
            ids = sorted(ex.id for ex in examples)
            manifest["splits"][split] = {
                "count": len(ids),
                "ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
            }
        (self.processed_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))


# --- Splitting and sampling -----------------------------------------------------------

def _carve_from_train(splits: Splits) -> Splits:
    """Move VAL_FRACTION of train to validation, then CALIBRATION_SIZE to calibration."""
    train, validation = _take(splits["train"], int(len(splits["train"]) * VAL_FRACTION))
    train, calibration = _take(train, CALIBRATION_SIZE)
    return {
        "train": train,
        "validation": [replace(ex, split="validation") for ex in validation],
        "calibration": [replace(ex, split="calibration") for ex in calibration],
        "test": splits["test"],
    }


def _cap(examples: list[Example], size: Optional[int]) -> list[Example]:
    """At most `size` questions (None = keep all)."""
    if size is None or size >= len(examples):
        return examples
    _, sample = _take(examples, size)
    return sample


def _take(examples: list[Example], n: int) -> tuple[list[Example], list[Example]]:
    """Split `examples` into (rest, sample of n). The sample keeps the same mix of
    difficulty and subject as the whole (stratified sampling, seeded)."""
    if n <= 0:
        return list(examples), []
    if n >= len(examples):
        return [], list(examples)

    groups: dict[tuple, list[Example]] = defaultdict(list)
    for ex in examples:
        groups[(ex.difficulty or "", ex.metadata.get("subject", ""))].append(ex)
    keys = sorted(groups)
    per_group = _allocate({k: len(groups[k]) for k in keys}, n)

    rng = random.Random(SEED)
    sample, rest = [], []
    for k in keys:
        pool = list(groups[k])
        rng.shuffle(pool)
        sample.extend(pool[: per_group[k]])
        rest.extend(pool[per_group[k]:])
    rng.shuffle(sample)
    rng.shuffle(rest)
    return rest, sample


def _allocate(group_sizes: dict, n: int) -> dict:
    """How many of `n` to take from each group, in proportion to its size
    (largest-remainder rounding, so the counts add up to exactly n)."""
    total = sum(group_sizes.values())
    shares = {k: n * size / total for k, size in group_sizes.items()}
    counts = {k: int(share) for k, share in shares.items()}
    leftover = n - sum(counts.values())
    for k in sorted(group_sizes, key=lambda k: shares[k] - counts[k], reverse=True)[:leftover]:
        counts[k] += 1
    return counts


def normalize_question(text: str) -> str:
    """Whitespace- and case-insensitive form of a question, for duplicate checks."""
    return " ".join(text.lower().split())
