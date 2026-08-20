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

# Binary collapse of ROUTING_LABELS, an alternative student target label
# space: {small, medium} -> "cheap", large -> "large". Cut at large-vs-rest
# rather than small-vs-rest because the oracle's tier distribution (5.4%
# small / 38.2% medium / 56.4% large, see research/research.md Table 2)
# makes this split roughly balanced (43.6% / 56.4%) instead of leaving a
# 5.4% minority class, and it isolates the *large* tier specifically -- the
# one Table 2 already flags as noisy (62-78% of its labels mean "nothing
# else succeeded," not "large specifically worked").
BINARY_ROUTING_LABELS = ("cheap", "large")
_TIER_TO_BINARY = {"small": "cheap", "medium": "cheap", "large": "large"}

# Selects which label space a student trains/evaluates against. The teacher
# and oracle are always labeled in ROUTING_LABELS (3-way) regardless of this
# choice -- map_route() below collapses their labels post-hoc for comparison
# against a "binary" student, rather than re-labeling either upstream.
ROUTE_LABEL_SPACES = {
    "3way": ROUTING_LABELS,
    "binary": BINARY_ROUTING_LABELS,
}


def map_route(route: str, target_labels: tuple[str, ...]) -> str:
    """Maps a routing label into whichever label space `target_labels` is.
    Identity if `route` already belongs to `target_labels` (e.g. 3-way ->
    3-way, or a route that's already collapsed); otherwise collapses a
    3-way tier down to BINARY_ROUTING_LABELS. Centralized here so a
    binary-trained checkpoint can be scored against the teacher's/oracle's
    always-3-way cached labels without duplicating the collapse rule at
    every call site (student/train_classifier.py, student/train_dual.py,
    evaluation/metrics.py, evaluation/router_eval.py, evaluation/
    oracle_check.py)."""
    if route in target_labels:
        return route
    if target_labels == BINARY_ROUTING_LABELS:
        return _TIER_TO_BINARY[route]
    raise ValueError(f"Can't map route {route!r} into label space {target_labels!r}")


# One candidate answering model per routing tier, cheapest first.
DEFAULT_CANDIDATE_ROSTER = {
    "small": "google/gemma-3-270m-it",
    "medium": "Qwen/Qwen2.5-0.5B-Instruct",
    "large": "google/gemma-3-1b-it",
}

# The student router's own backbone. Deliberately the same checkpoint as the
# "small" answering candidate above -- derived, not duplicated, so the two
# can't silently drift apart. They are different model *instances* trained
# toward different objectives (see student/train_classifier.py, student/
# train_dual.py's module docstrings), not the same model wearing two hats.
DEFAULT_STUDENT_MODEL_ID = DEFAULT_CANDIDATE_ROSTER["small"]
