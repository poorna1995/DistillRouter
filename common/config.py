"""Path and constant configuration for the whole project."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
TEACHER_DIR = DATA_DIR / "teacher"
ORACLE_DIR = DATA_DIR / "oracle"
STUDENT_DIR = DATA_DIR / "student"

CANONICAL_SPLITS = ("train", "validation", "test")

DEFAULT_VAL_FRACTION = 0.1  # validation carved from train if the source ships none
DEFAULT_SEED = 42

DEFAULT_CALIBRATION_SIZE = 20  # rows carved from train, source of teacher few-shot demos

# Per-split size cap applied at prepare-data time. None = uncapped.
DEFAULT_SAMPLE_SIZE_CAPS = {"train": 1000, "validation": None, "test": 500, "calibration": None}

ROUTING_LABELS = ("small", "medium", "large")  # cheapest -> most capable

# One candidate answering model per routing tier, cheapest first.
DEFAULT_CANDIDATE_ROSTER = {
    "small": "google/gemma-3-270m-it",
    "medium": "Qwen/Qwen2.5-0.5B-Instruct",
    "large": "google/gemma-3-1b-it",
}
