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

# Canonical splits every processed dataset is normalized to, regardless of
# what splits the upstream source ships.
CANONICAL_SPLITS = ("train", "validation", "test")

# Used only when a source dataset ships no native validation split (true of
# both GSM8K and MATH) and one has to be carved out of train (see
# datasets/base.py::BaseDatasetLoader._to_canonical_splits).
DEFAULT_VAL_FRACTION = 0.1
DEFAULT_SEED = 42
