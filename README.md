# DistillRouter

Distilling a large LLM routing policy into a lightweight student router.
See [DISTILLROUTER_SPEC.md](DISTILLROUTER_SPEC.md) for the full design.

## Project layout

```
distillroute/
├── run.py              # single entry point for the whole project
├── common/             # shared across every module below
│   ├── config.py        # path constants, canonical splits, seed
│   └── schema.py         # the one `Example` record shape every dataset is normalized to
├── dataset/             # dataset loaders (raw -> processed)
│   ├── base.py            # BaseDatasetLoader + registry — the extensibility point
│   ├── gsm8k.py
│   └── math.py
├── teacher/             # oracle labeling + teacher prompting (not yet implemented)
├── student/             # student distillation (not yet implemented)
├── router/              # deployment-time routing (not yet implemented)
└── data/
    ├── raw/{dataset}/            # untouched source dumps, one jsonl per native split
    └── processed/{dataset}/      # canonical train.jsonl / validation.jsonl / test.jsonl
```

Note the package is named `dataset` (singular), not `datasets` — the
HuggingFace library used inside it is called `datasets` (plural), and a
same-named local package would shadow it.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python run.py list-datasets              # show every registered loader
python run.py prepare-data --dataset gsm8k
python run.py prepare-data --dataset gsm8k math
python run.py prepare-data --all          # prepare everything registered
```

Each dataset writes:

- `data/raw/<name>/<native_split>.jsonl` — an unmodified snapshot of the source, so re-running never re-downloads.
- `data/processed/<name>/{train,validation,test}.jsonl` — normalized to the shared `Example` schema (`common/schema.py`).

GSM8K and MATH ship no native validation split; `BaseDatasetLoader` carves
one deterministically out of train (10%, fixed seed 42) so every dataset
produces the same three canonical files regardless of what the source
provides.

## Adding a new dataset

1. Create `dataset/<name>.py` with a class subclassing `BaseDatasetLoader`,
   implementing `download()` and `load_native_split()`, decorated with
   `@register` (copy `dataset/gsm8k.py` as the simplest template).
2. Add one import line in `dataset/__init__.py`.
3. `python run.py prepare-data --dataset <name>` — no other file changes.

## Current datasets

| Name    | Domain | Source                                                                                                             | Native splits (verified) |
| ------- | ------ | ------------------------------------------------------------------------------------------------------------------ | ------------------------ |
| `gsm8k` | math   | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k)                                                       | train 7,473 / test 1,319 |
| `math`  | math   | [EleutherAI/hendrycks_math](https://huggingface.co/datasets/EleutherAI/hendrycks_math) (7 subject configs, summed) | train 7,500 / test 5,000 |

Processed (post validation carve-out, `data/processed/`):

| Dataset | Train | Validation | Test  | Total  |
| ------- | ----- | ---------- | ----- | ------ |
| gsm8k   | 6,726 | 747        | 1,319 | 8,792  |
| math    | 6,749 | 749        | 5,000 | 12,498 |
