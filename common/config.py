"""Central path and constant configuration for the whole project.

Every module (datasets/, teacher/, student/, router/) imports paths from
here rather than hardcoding them, so relocating data (e.g. onto a mounted
volume for a larger run) is a one-line change instead of a project-wide
find-and-replace.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # .../distillroute

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
TEACHER_DIR = DATA_DIR / "teacher"
ORACLE_DIR = DATA_DIR / "oracle"

# Canonical splits every processed dataset is normalized to, regardless of
# what splits the upstream source ships.
CANONICAL_SPLITS = ("train", "validation", "test")

# Used only when a source dataset ships no native validation split (true of
# both GSM8K and MATH) and one has to be carved out of train (see
# datasets/base.py::BaseDatasetLoader._to_canonical_splits).
DEFAULT_VAL_FRACTION = 0.1
DEFAULT_SEED = 42

# Size of the calibration split carved out of train (see
# dataset/base.py::BaseDatasetLoader._carve_calibration_split). These rows
# supply the teacher's few-shot demonstration examples (teacher/base.py)
# once oracle labeling exists — carved out, not just sampled, so they are
# structurally disjoint from the train rows that go on to become
# distillation targets, and (by construction, since they only ever come
# from train) never overlap validation/test either.
DEFAULT_CALIBRATION_SIZE = 20

# Per-split size caps applied once, deterministically, at prepare-data time
# (see dataset/base.py::BaseDatasetLoader._cap_split_sizes) — the single
# place dataset size is controlled, so every downstream stage (teacher
# labeling, oracle labeling, student training, eval) works off the same
# fixed query set instead of each re-sampling independently. A value of
# None leaves that split uncapped. Applied after the validation carve-out
# and the calibration carve-out, and stratified by Example.difficulty when
# a split has one (e.g. MATH's levels); falls back to plain seeded random
# sampling otherwise (GSM8K). `calibration` is listed as uncapped since its
# size is already fixed by DEFAULT_CALIBRATION_SIZE at carve time.
DEFAULT_SAMPLE_SIZE_CAPS = {"train": 1000, "validation": None, "test": 500, "calibration": None}

# Routing tiers a teacher assigns queries to (see teacher/base.py). Ordered
# cheapest -> most capable.
ROUTING_LABELS = ("small", "medium", "large")

# Canonical label <-> class-index mapping, derived from ROUTING_LABELS — used
# wherever a routing label needs to become a model class index (student
# training/inference, see student/) or vice versa.
LABEL_TO_ID = {label: i for i, label in enumerate(ROUTING_LABELS)}
ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}

# Candidate answering-model roster used for oracle labeling (see
# candidate/base.py) — one model per routing tier, executed in this order
# (cheapest first) with early exit at the first correct answer. Deviates
# from DISTILLROUTER_SPEC.md's Table 6b at the Large tier: the report's
# pick (Qwen3.5-0.8B) turned out to be a vision-language model, not a
# plain text model, so it was swapped for a same-family (Gemma 3), plain
# text alternative to avoid the multimodal processor stack for a role that
# never needs image input.
DEFAULT_CANDIDATE_ROSTER = {
    "small": "google/gemma-3-270m-it",
    "medium": "Qwen/Qwen2.5-0.5B-Instruct",
    "large": "google/gemma-3-1b-it",
}
