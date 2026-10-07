"""Base class and registry for dataset loaders. Add a dataset by writing
a BaseDatasetLoader subclass decorated with @register — nothing else
changes.
"""
from __future__ import annotations

import json
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Dict, Optional, Type

from common.config import (
    DEFAULT_CALIBRATION_SIZE,
    DEFAULT_SAMPLE_SIZE_CAPS,
    DEFAULT_SEED,
    DEFAULT_VAL_FRACTION,
    PROCESSED_DIR,
    RAW_DIR,
)
from common.schema import Example, write_jsonl
from common.split_integrity import check_dataset_splits

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


def datasets() -> list[str]:
    return sorted(_REGISTRY)


class BaseDatasetLoader(ABC):
    """One subclass = one benchmark dataset, normalized to schema.Example.
    Subclasses implement only download() and load_native_split(); split
    carving, capping, and writing are handled here."""

    name: str                                    # registry key, e.g. "gsm8k"
    domain: str                                   # e.g. "math", "knowledge", "code"
    native_splits: tuple[str, ...]                # splits as the source provides them
    val_fraction: float = DEFAULT_VAL_FRACTION     # used only if source has no "validation" split
    seed: int = DEFAULT_SEED
    sample_size_caps: Dict[str, Optional[int]] = DEFAULT_SAMPLE_SIZE_CAPS  # per-split cap; None = uncapped
    calibration_size: int = DEFAULT_CALIBRATION_SIZE  # rows carved from train for teacher few-shot

    def __init__(self, raw_dir: Path = RAW_DIR, processed_dir: Path = PROCESSED_DIR) -> None:
        self.raw_dir = raw_dir / self.name
        self.processed_dir = processed_dir / self.name
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.sample_size_caps = dict(self.sample_size_caps)  # copy: avoid mutating the shared class default

    @abstractmethod
    def download(self) -> None:
        """Fetch/cache raw source data into self.raw_dir, if not already present."""

    @abstractmethod
    def load_native_split(self, split: str) -> list[Example]:
        """Return standardized Examples for one native split (as named by the source)."""

    def prepare(self) -> dict[str, int]:
        """Run raw -> processed for this dataset. Returns {split: count}.
        Ends with an automated disjointness check (common/split_integrity.py);
        raises SplitLeakageError, uncaught, if it fails.
        """
        print(f"[{self.name}] downloading/caching raw source into {self.raw_dir} ...")
        self.download()

        native = {split: self.load_native_split(split) for split in self.native_splits}
        print(f"[{self.name}] loaded native splits: {{ {', '.join(f'{s}: {len(ex)}' for s, ex in native.items())} }}")

        canonical = self._to_canonical_splits(native)
        if "validation" not in native:
            print(
                f"[{self.name}] no native validation split -- carved {len(canonical['validation'])} "
                f"rows out of train (val_fraction={self.val_fraction}, seed={self.seed})"
            )

        canonical, calibration_report = self._carve_calibration_split(canonical)
        print(f"[{self.name}] carved calibration split: {calibration_report['calibration_size']} rows")

        capped, cap_report = self._cap_split_sizes(canonical)
        print(f"[{self.name}] size caps applied: {cap_report['actual_counts']} (requested {cap_report['requested_caps']})")

        counts = {}
        for split, examples in capped.items():
            write_jsonl(self.processed_dir / f"{split}.jsonl", examples)
            counts[split] = len(examples)
        print(f"[{self.name}] wrote processed splits -> {self.processed_dir}: {counts}")

        self._write_sample_manifest({**cap_report, **calibration_report})
        # self.processed_dir already has self.name appended (see __init__); check_dataset_splits
        # appends dataset_name itself, so pass the parent -- otherwise this always validates
        # PROCESSED_DIR/<name> (the default) instead of wherever this loader instance actually
        # just wrote to, silently vacuous for a loader constructed with a custom processed_dir.
        check_dataset_splits(self.name, self.processed_dir.parent)  # prints its own confirmation, raises on leakage
        print(f"[{self.name}] prepare() complete: {counts}")
        return counts

    def _to_canonical_splits(self, native: dict[str, list[Example]]) -> dict[str, list[Example]]:
        """Map native splits onto train/validation/test. If the source has
        no validation split, carve one from train (seeded, deterministic).
        """
        if "validation" in native:
            return native

        train = native["train"]
        rng = random.Random(self.seed)
        indices = list(range(len(train)))
        rng.shuffle(indices)
        n_val = int(len(train) * self.val_fraction)
        val_ids = set(indices[:n_val])

        # Example.id keeps its original "...-train-<i>" form even after
        # being carved into validation — ids are the cache key downstream
        # caches are keyed on. Example.split is updated to match reality.
        new_train, validation = [], []
        for i, ex in enumerate(train):
            if i in val_ids:
                validation.append(replace(ex, split="validation"))
            else:
                new_train.append(ex)

        canonical = {"train": new_train, "validation": validation}
        if "test" in native:
            canonical["test"] = native["test"]
        return canonical

    def _carve_calibration_split(
        self, canonical: dict[str, list[Example]]
    ) -> tuple[dict[str, list[Example]], dict]:
        """Carve calibration_size rows out of train into a `calibration`
        split — source of the teacher's few-shot demos. Runs before
        _cap_split_sizes so capping draws from what's left. Always writes
        a calibration.jsonl, even empty, so downstream readers get a clear
        error instead of a missing file.
        """
        train = canonical["train"]
        if self.calibration_size <= 0 or self.calibration_size >= len(train):
            return {**canonical, "calibration": []}, {"calibration_size": 0}

        calibration, remaining_train, group_allocations = _stratified_sample(
            train, self.calibration_size, self.seed
        )
        calibration = [replace(ex, split="calibration") for ex in calibration]

        updated = dict(canonical)
        updated["train"] = remaining_train
        updated["calibration"] = calibration

        report = {"calibration_size": len(calibration)}
        if group_allocations:
            report["calibration_group_counts"] = group_allocations
        return updated, report

    def _cap_split_sizes(
        self, canonical: dict[str, list[Example]]
    ) -> tuple[dict[str, list[Example]], dict]:
        """Cap each split to sample_size_caps, once. A split with no cap
        (or a cap >= its size) is left untouched. Stratifies by
        Example.difficulty when more than one value is present, else
        plain seeded random sampling.
        """
        capped: dict[str, list[Example]] = {}
        sample_report = {
            "seed": self.seed,
            "requested_caps": dict(self.sample_size_caps),
            "actual_counts": {},
            "stratified_splits": {},
        }

        for split, examples in canonical.items():
            target_size = self.sample_size_caps.get(split)
            if target_size is None or target_size >= len(examples):
                capped[split] = examples
                sample_report["actual_counts"][split] = len(examples)
                continue

            selected, _remaining, group_allocations = _stratified_sample(examples, target_size, self.seed)
            if group_allocations:
                sample_report["stratified_splits"][split] = group_allocations

            capped[split] = selected
            sample_report["actual_counts"][split] = len(selected)

        return capped, sample_report

    def _write_sample_manifest(self, sample_report: dict) -> None:
        """Record the sampling decision next to the processed files it produced."""
        manifest_path = self.processed_dir / "sample_manifest.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(sample_report, f, indent=2, sort_keys=True)


def _group_by_difficulty(examples: list[Example]) -> dict:
    groups: dict = defaultdict(list)
    for example in examples:
        groups[example.difficulty].append(example)
    return dict(groups)


def _stratified_sample(
    examples: list[Example], target_size: int, seed: int
) -> tuple[list[Example], list[Example], dict]:
    """Split examples into (selected, remaining), len(selected) == target_size.
    Stratified by Example.difficulty when >1 value is present (third
    return value: per-group allocation); plain random sample otherwise.
    """
    difficulty_groups = _group_by_difficulty(examples)
    rng = random.Random(seed)

    if len(difficulty_groups) <= 1:
        pool = list(examples)
        rng.shuffle(pool)
        return pool[:target_size], pool[target_size:], {}

    group_allocations = _allocate_group_sizes(
        {difficulty_key: len(group) for difficulty_key, group in difficulty_groups.items()}, target_size
    )
    selected, remaining = [], []
    for difficulty_key, group in difficulty_groups.items():
        pool = list(group)
        rng.shuffle(pool)
        n_selected = group_allocations[difficulty_key]
        selected.extend(pool[:n_selected])
        remaining.extend(pool[n_selected:])
    rng.shuffle(selected)  # undo group-by-group ordering

    return selected, remaining, group_allocations


def _allocate_group_sizes(group_sizes: dict, target_size: int) -> dict:
    """Largest-remainder allocation of target_size across groups,
    proportional to each group's share, capped at its own size, summing
    to exactly target_size."""
    total = sum(group_sizes.values())
    difficulty_keys = sorted(group_sizes, key=lambda key: (key is None, key))
    exact_shares = {key: target_size * group_sizes[key] / total for key in difficulty_keys}
    allocations = {key: min(group_sizes[key], int(exact_shares[key])) for key in difficulty_keys}

    remaining = target_size - sum(allocations.values())
    remainder_order = sorted(
        difficulty_keys, key=lambda key: exact_shares[key] - int(exact_shares[key]), reverse=True
    )
    while remaining > 0:
        allocated_this_pass = False
        for key in remainder_order:
            if remaining == 0:
                break
            if allocations[key] < group_sizes[key]:
                allocations[key] += 1
                remaining -= 1
                allocated_this_pass = True
        if not allocated_this_pass:
            break  # every group at its own size; target_size > total (shouldn't happen)
    return allocations
