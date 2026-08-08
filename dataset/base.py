"""Base class and registry for dataset loaders.

Adding a new dataset to the project (TriviaQA, HotpotQA, HumanEval, MBPP, ...)
means writing one new loader module in this package with a
`BaseDatasetLoader` subclass decorated with `@register` — nothing else in
the project (run.py, the CLI, downstream teacher/student code) has to
change. This is the scalability mechanism the whole `datasets/` package is
built around: new datasets are additive, never require editing existing
loaders or the entry point.
"""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Type

from common.config import DEFAULT_SEED, DEFAULT_VAL_FRACTION, PROCESSED_DIR, RAW_DIR
from common.schema import Example, write_jsonl

_REGISTRY: Dict[str, Type["BaseDatasetLoader"]] = {}


def register(cls: Type["BaseDatasetLoader"]) -> Type["BaseDatasetLoader"]:
    """Class decorator: makes a loader discoverable by its `name` attribute."""
    _REGISTRY[cls.name] = cls
    return cls


def get_loader_class(name: str) -> Type["BaseDatasetLoader"]:
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none registered)"
        raise KeyError(f"Unknown dataset '{name}'. Available: {available}")
    return _REGISTRY[name]


def available_datasets() -> list[str]:
    return sorted(_REGISTRY)


class BaseDatasetLoader(ABC):
    """One subclass = one benchmark dataset, normalized to `schema.Example`.

    Subclasses implement only `download()` and `load_native_split()`.
    Everything that should behave identically across every dataset in the
    project — carving a validation split out of train when the source
    doesn't ship one, writing processed JSONL, canonical split naming — is
    handled once, here.
    """

    name: str                                    # registry key, e.g. "gsm8k"
    domain: str                                   # e.g. "math", "knowledge", "code"
    native_splits: tuple[str, ...]                # splits as the *source* provides them
    val_fraction: float = DEFAULT_VAL_FRACTION     # only used if source has no "validation" split
    seed: int = DEFAULT_SEED

    def __init__(self, raw_dir: Path = RAW_DIR, processed_dir: Path = PROCESSED_DIR) -> None:
        self.raw_dir = raw_dir / self.name
        self.processed_dir = processed_dir / self.name
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def download(self) -> None:
        """Fetch/cache raw source data into `self.raw_dir`, if not already present."""

    @abstractmethod
    def load_native_split(self, split: str) -> list[Example]:
        """Return standardized Examples for one *native* split (as named by the source)."""

    def prepare(self) -> dict[str, int]:
        """Run the full raw -> processed pipeline for this dataset.

        Returns a {split_name: example_count} summary for logging.
        """
        self.download()
        native = {split: self.load_native_split(split) for split in self.native_splits}
        canonical = self._to_canonical_splits(native)
        counts = {}
        for split, examples in canonical.items():
            write_jsonl(self.processed_dir / f"{split}.jsonl", examples)
            counts[split] = len(examples)
        return counts

    def _to_canonical_splits(self, native: dict[str, list[Example]]) -> dict[str, list[Example]]:
        """Map native splits onto the project's canonical train/validation/test.

        Most datasets added so far (GSM8K, MATH) ship only train/test — a
        validation slice is deterministically carved out of train so every
        dataset produces the same three canonical files downstream code can
        rely on without special-casing. Datasets that already ship a native
        validation split (e.g. future additions like TriviaQA/HotpotQA)
        pass through unchanged.
        """
        if "validation" in native:
            return native

        train = native["train"]
        rng = random.Random(self.seed)
        indices = list(range(len(train)))
        rng.shuffle(indices)
        n_val = int(len(train) * self.val_fraction)
        val_ids = set(indices[:n_val])

        new_train, validation = [], []
        for i, ex in enumerate(train):
            (validation if i in val_ids else new_train).append(ex)

        canonical = {"train": new_train, "validation": validation}
        if "test" in native:
            canonical["test"] = native["test"]
        return canonical
