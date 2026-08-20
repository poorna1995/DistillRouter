# DistillRouter

Distilling a large LLM routing policy into a lightweight student router.
See [DISTILLROUTER_SPEC.md](DISTILLROUTER_SPEC.md) for the full design, and
[PIPELINE.md](PIPELINE.md) for the complete from-scratch, end-to-end
runbook (setup through trained/evaluated router, both label spaces,
multi-seed reproduction, figures, and the paper build).

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
├── teacher/             # teacher labeling (heuristic backend; oracle labeling not yet implemented)
│   ├── base.py            # TeacherModel + registry — same pattern as dataset/base.py
│   ├── heuristic.py        # zero-cost deterministic baseline teacher
│   ├── cache.py            # incremental label cache, keyed on (query_id, prompt_version)
│   └── labeler.py          # orchestrates read -> cache-check -> dedupe -> predict -> cache
├── student/             # student distillation (not yet implemented)
├── router/              # deployment-time routing (not yet implemented)
└── data/
    ├── raw/{dataset}/            # untouched source dumps, one jsonl per native split
    ├── processed/{dataset}/      # train.jsonl / validation.jsonl / test.jsonl / calibration.jsonl
    └── teacher/{dataset}/        # teacher labels per split, one cache file each
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

python run.py list-teachers              # show every registered teacher backend
python run.py label-data --dataset gsm8k --teacher heuristic
python run.py label-data --all --split train validation
```

Each dataset writes:

- `data/raw/<name>/<native_split>.jsonl` — an unmodified snapshot of the source, so re-running never re-downloads.
- `data/processed/<name>/{train,validation,test}.jsonl` — normalized to the shared `Example` schema (`common/schema.py`).

`label-data` writes `data/teacher/<name>/<split>.jsonl` — one `TeacherLabel`
per query (routing label + soft probability distribution + latency/cost).
It's incremental: labels are keyed on `(query_id, prompt_version)` and
appended as produced, so rerunning only calls the teacher for queries that
aren't cached yet (or whose `TeacherModel.prompt_version` changed) —
already-labeled queries and duplicate query text are skipped/reused, not
re-labeled. The default `heuristic` backend is a zero-cost deterministic
baseline for exercising the pipeline; see `teacher/base.py` for how to
register a real LLM-backed teacher.

GSM8K and MATH ship no native validation split; `BaseDatasetLoader` carves
one deterministically out of train (10%, fixed seed 42) so every dataset
produces the same three canonical files regardless of what the source
provides.

Next, a small `calibration` split (`common.config.DEFAULT_CALIBRATION_SIZE`,
default 20) is carved out of train — these rows are reserved to become the
teacher's few-shot demonstration examples once oracle labeling exists, and
are removed from train so the teacher never sees its own demonstrations as
a labeling target. Because they only ever come from train, they can't
overlap `validation`/`test` either — no separate leakage check needed.

Finally, `train`/`test` are capped to a fixed size
(`common.config.DEFAULT_SAMPLE_SIZE_CAPS`, default 1,000 / 500;
`validation` and `calibration` stay uncapped) — the one place dataset size
is controlled, so `label-data` and every later stage work off the same
fixed query set instead of each sampling independently. Both the
calibration carve-out and the size cap are stratified by
`Example.difficulty` when a split has more than one distinct value (MATH's
1–5 levels), so every difficulty level stays represented; otherwise they
fall back to plain seeded sampling (GSM8K, which carries no difficulty
tag). The exact seed, requested caps, calibration size, and (when
stratified) per-level counts are written to
`data/processed/<name>/sample_manifest.json`.

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

Processed (post validation carve-out, calibration carve-out, and sample cap, `data/processed/`):

| Dataset | Train | Validation | Test | Calibration | Total |
| ------- | ----- | ---------- | ---- | ----------- | ----- |
| gsm8k   | 1,000 | 747        | 500  | 20          | 2,267 |
| math    | 1,000 | 749        | 500  | 20          | 2,269 |
