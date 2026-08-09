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
    sample_size_caps: Dict[str, Optional[int]] = DEFAULT_SAMPLE_SIZE_CAPS  # per-split cap; None = uncapped
    calibration_size: int = DEFAULT_CALIBRATION_SIZE  # rows carved out of train for teacher few-shot examples

    def __init__(self, raw_dir: Path = RAW_DIR, processed_dir: Path = PROCESSED_DIR) -> None:
        self.raw_dir = raw_dir / self.name
        self.processed_dir = processed_dir / self.name
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        # Copied, not aliased: `sample_size_caps` looks like a per-instance
        # dict, so mutating one entry (self.sample_size_caps["train"] = ...)
        # is a natural thing to try — without this copy, that would mutate
        # the single shared DEFAULT_SAMPLE_SIZE_CAPS object every loader
        # that didn't override the class attribute is still pointing at.
        self.sample_size_caps = dict(self.sample_size_caps)

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
        canonical, calibration_report = self._carve_calibration_split(canonical)
        capped, cap_report = self._cap_split_sizes(canonical)

        counts = {}
        for split, examples in capped.items():
            write_jsonl(self.processed_dir / f"{split}.jsonl", examples)
            counts[split] = len(examples)

        self._write_sample_manifest({**cap_report, **calibration_report})
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

        # `Example.id` intentionally keeps its original "...-train-<i>" form
        # even after being carved into validation — ids are the stable
        # cache key every downstream cache (TeacherLabelCache,
        # OracleAttemptCache) is keyed on, and must stay put regardless of
        # which split a row ends up in. `Example.split` is not an id,
        # though, and must reflect where the row actually landed.
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
        """Carve `calibration_size` rows out of train into their own `calibration`
        split, before train is capped for distillation and before any other
        split-level sampling happens.

        These rows will supply the teacher's few-shot demonstration
        examples once oracle labeling exists (teacher/base.py) — carved
        out, not merely sampled, so they are structurally disjoint from the
        train rows that later become distillation targets. Since they only
        ever come from train, they never overlap validation/test either —
        no separate leakage check needed elsewhere.

        Must run before `_cap_split_sizes`, so train-capping draws from
        what's left after this carve-out rather than competing with it for
        the same rows.

        A `calibration.jsonl` file is always written, even when
        calibration_size is 0 or larger than train itself (an empty list,
        in that case) — so a downstream reader (oracle-label, QwenTeacher's
        few-shot loader) always finds the file and can raise its own clear
        "no calibration examples" error, rather than hitting a raw
        FileNotFoundError from a file that silently never got created.
        """
        train = canonical["train"]
        if self.calibration_size <= 0 or self.calibration_size >= len(train):
            return {**canonical, "calibration": []}, {"calibration_size": 0}

        calibration, remaining_train, group_allocations = _stratified_sample(
            train, self.calibration_size, self.seed
        )
        # ids stay as-is (see _to_canonical_splits' comment on why); split
        # must be updated so a calibration row doesn't claim to be "train".
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
        """Cap each split to `sample_size_caps`, once, deterministically.

        Every downstream stage (teacher labeling, oracle labeling, student
        training, eval) reads whatever ends up in `data/processed/` — this
        is the one place dataset size is decided, so no stage re-samples on
        its own and none of them can disagree about which queries exist.

        A split whose cap is None, or whose cap is >= its current size, is
        left untouched (this is how `validation` and `calibration` stay
        uncapped by default). Otherwise: if the split has more than one
        distinct `Example.difficulty` value (MATH's "1".."5" levels), the
        target size is allocated proportionally across those groups so
        every difficulty level stays represented; a split with a single
        difficulty value (including GSM8K, where every example's
        difficulty is None) falls back to plain seeded random sampling.
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
    """Split `examples` into (selected, remaining) with `len(selected) == target_size`.

    Stratified proportionally by `Example.difficulty` when more than one
    distinct value is present — returns the per-difficulty allocation as
    the third element, for manifest reporting. Falls back to plain seeded
    random sampling when there's only one difficulty value in play (e.g.
    GSM8K, where every example's difficulty is None); the third element is
    then an empty dict.
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
    rng.shuffle(selected)  # undo the group-by-group ordering

    return selected, remaining, group_allocations


def _allocate_group_sizes(group_sizes: dict, target_size: int) -> dict:
    """Largest-remainder allocation of `target_size` across groups, proportional
    to each group's share of the total, capped at that group's own size, summing
    to exactly `target_size` (rather than drifting from float-rounding error).
    """
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
            break  # every group already at its own size; target_size > total (shouldn't happen)
    return allocations
