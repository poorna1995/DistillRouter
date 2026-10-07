
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"              # downloaded datasets
PROCESSED_DIR = DATA_DIR / "processed"  # cleaned questions, one file per split
ORACLE_DIR = DATA_DIR / "oracle"        # candidate answers and true labels
TEACHER_DIR = DATA_DIR / "teacher"      # teacher votes and labels
STUDENT_DIR = DATA_DIR / "student"      # student training files

# --- Data splits  ---------------------------------------------

SEED = 42
VAL_FRACTION = 0.1      # validation carved from train when a dataset has none
CALIBRATION_SIZE = 20   # carved from train; teacher few-shot examples only

# Questions per split, per dataset.
SAMPLE_SIZE_CAPS = {"train": 2000, "validation": 300, "test": 500}

# --- Models ----------------------------------------------------

# small and large models
CANDIDATE_MODELS = {
    "small": "Qwen/Qwen2.5-1.5B-Instruct",
    "large": "Qwen/Qwen2.5-14B-Instruct",
}

# teacher model.
TEACHER_MODEL = "Qwen/Qwen2.5-14B-Instruct"

# student models
STUDENT_MODELS = {
    "270m": "google/gemma-3-270m-it",
    "1b": "google/gemma-3-1b-it",
}
STUDENT_MODEL = "270m"

# --- Generation ----------------------------------------------------

CANDIDATE_SAMPLES = 5      # answers per question per candidate
TEACHER_VOTES = 5          # teacher routing decisions per question
TEMPERATURE = 0.7
TOP_P = 0.95
MAX_NEW_TOKENS = 1024

# --- Labels ---------------------------------------------------

LABELS = ("small", "large")   # binary routing, cheapest first
SOLVE_THRESHOLD = 0.6         # a model "can solve" a question if >= 3 of 5 answers are correct

# --- Distillation hyperparameters -------------------------------------

STUDENT_MODEL_ID = STUDENT_MODELS[STUDENT_MODEL]
EPOCHS = 3
LEARNING_RATE = 2e-5
BATCH_SIZE = 8
WARMUP_RATIO = 0.0
LR_SCHEDULER = "linear"
ROUTE_HEAD_DROPOUT = 0.1
ROUTE_LOSS = "soft"           # "soft" | "hard" | "hard+soft" (experiment E6)
LAMBDA_ROUTE = 1.0            # weight of the routing loss (reasoning-plus-route)
LAMBDA_REASON = 1.0           # weight of the explanation loss (reasoning-plus-route)
OVERSAMPLE_RATIO = 0.0        # 0 = no oversampling of the rarer label 
