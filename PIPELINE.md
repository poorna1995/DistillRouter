# DistillRouter — End-to-End Pipeline

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

make data oracle-data teacher-data   # one-time setup: data, ground truth, teacher labels
make train-binary                    # one seed, both variants — smoke test, ~20-40 min
```

That gets you a trained, evaluated router (binary label space) end to end.
`make help` lists every target. Requires a GPU in practice — every stage
from `oracle-data` onward loads real HF causal LMs.

## Reproducing the paper's tables

Each row below is one seed sweep (0-9), both variants, paired-bootstrap
aggregated — this is what `research.tex`'s tables actually report, not a
single-seed run.

| Command | Label space | Backbone | Table | Approx. time |
|---|---|---|---|---:|
| `make reproduce-table3`  | binary | 270M | Table 3 (primary result) | ~4h |
| `make reproduce-table4`  | binary | 1B   | Table 4 | ~13h |
| `make reproduce-table8`  | 3-way  | 270M | Table 8 | ~4h |
| `make reproduce-table10` | 3-way  | 1B   | Table 10 | ~13h |
| `make reproduce-all`     | all four above | | | ~34h |

Then:

```bash
make figures   # regenerate every paper figure
```

`research/research.tex` (NeurIPS format) compiles with `pdflatex` —
not available in this environment; needs a TeX Live install elsewhere.

**Timings above are measured**, from this project's own run history
(file mtimes across a completed sweep), not estimates — see the bottom
of `EXPERIMENTS.md` for the exact seed/config each table came from.

**Disk space**: `reproduce-table3`/`table8` write ~54GB each (270M
backbone), `table4`/`table10` ~190GB each (1B backbone) — full
`reproduce-all` is ~480GB. Only `predictions.jsonl` +
`classification_report.json` per run are worth keeping long-term; the
rest (`model.safetensors`, `checkpoint-*/`) is regenerable — see "Known
gaps" below.

## Known gaps

- **`multiseed/routellm_bert_scores.json`** (the RouteLLM baseline row in
  `research.tex`'s main table) has no generating script anywhere in this
  repo. Track down or rewrite it before claiming full reproducibility.
- **`research/research.md`**, a Markdown twin of `research.tex` that old
  tooling (`EXPERIMENTS.md`, prior session notes) still references, does
  not currently exist on disk. `research.tex` is the sole source of
  truth for now.
- **`research/checklist.tex`** hasn't been updated through several recent
  restructures of the results section — verify its table/figure
  references before submission.
- Full checkpoints (`model.safetensors`, `checkpoint-*/`) are
  multi-GB-per-seed and regenerable by re-running the relevant `make`
  target with the same `SEED=`; they are not bit-identical on rerun
  (GPU non-determinism) but are the same experiment statistically. Only
  `predictions.jsonl`/`classification_report.json` are worth keeping
  long-term. See `EXPERIMENTS.md` for every seed/config already run.

---

## Manual pipeline (every flag, one stage at a time)

Everything below is what the `make` targets above actually run. Use this
when you need something the targets don't cover — a third dataset, a
custom `--lambda-reason`, a different backbone, one dataset instead of
`--all`, etc. Every command here is copied from `run.py --help` / the
scripts' own `--help` — run `--help` yourself if anything drifts.

### 1. Prepare the datasets

```bash
python run.py list-datasets                 # gsm8k, math
python run.py prepare-data --all             # downloads + normalizes both
python run.py check-splits --all             # asserts no train/val/test/calibration leakage
```

Writes `data/raw/<name>/` (untouched source dump) and
`data/processed/<name>/{train,validation,test,calibration}.jsonl`. Train
and test are capped to 1,000/500 (`common/config.py:DEFAULT_SAMPLE_SIZE_CAPS`),
stratified by difficulty where the dataset has one. A 20-row `calibration`
split is carved out of train — it exists to become the teacher's few-shot
demonstrations next, and is excluded from train so the teacher never sees
its own demo set as a labeling target.

### 2. Oracle-label (ground truth)

```bash
python run.py oracle-label --all --split calibration   # do this first — teacher few-shot needs it
python run.py oracle-label --all --split train
python run.py oracle-label --all --split validation
python run.py oracle-label --all --split test
```

Runs **every** candidate tier (`gemma-3-270m-it` / `Qwen2.5-0.5B-Instruct`
/ `gemma-3-1b-it`, see `common/config.py:DEFAULT_CANDIDATE_ROSTER`) on
every query, no early-exit, and assigns the cheapest tier that got it
right as the oracle label — this is ground truth, independent of any
teacher. Writes `data/oracle/<dataset>/<split>.labels.jsonl`. This is the
slowest stage (three full model passes per query) — `--limit N` lets you
smoke-test on a slice first.

### 3. Teacher-label

```bash
python run.py list-teachers          # oracle-direct, qwen2.5-3b-v1, qwen2.5-3b-v2
python run.py label-data --all --split train validation test --teacher qwen2.5-3b-v2
```

`qwen2.5-3b-v2` is the teacher this paper's results use (few-shot,
label + reasoning). `qwen2.5-3b-v1` is label-only, no reasoning — that
absence is what makes the route-only vs. reasoning-plus-route comparison
meaningful downstream. `oracle-direct` is a RouteLLM-style non-teacher
baseline that looks up the oracle label instead of calling a model — use
it directly in evaluation comparisons, not as a `--teacher` for training.

Both prompted teachers few-shot from `calibration`'s oracle labels (step
2) — running this before oracle-labeling `calibration` will fail. Output
is `data/teacher/<dataset>/<output_version>/<split>.jsonl`, incremental
and cached on `(query_id, prompt_version)` — safe to re-run.

### 4. Build student training data

```bash
python run.py build-student-data --all --teacher qwen2.5-3b-v2 --split train
python run.py build-binary-student-data --all --teacher qwen2.5-3b-v2 --split train
```

The first joins query text + teacher output into
`{query, teacher_route, routing_reasoning}` records
(`data/student/<dataset>/<output_version>/train.jsonl`) — this is what
both student variants train on, 3-way labels. The second **derives** a
binary-collapsed copy (`{small,medium}→cheap`, `large→large`) as its own
file (`.../binary/train.jsonl`) — run it after the first, never edits the
3-way file it reads from. You need both if you're training both label
spaces.

### 5. Train the student router

Two architectural variants × two label spaces = four combinations. Pick
what you need:

```bash
# Experiment 1: route-only classifier (no reasoning signal)
python run.py train-student-classifier --all --teacher qwen2.5-3b-v2 \
  --label-space binary --seed 42 --checkpoint-dir checkpoints/student-classifier-binary

# Experiment 2: dual-head (route classifier + reasoning LM head, joint loss)
python run.py train-student-dual --all --teacher qwen2.5-3b-v2 \
  --label-space binary --seed 42 --checkpoint-dir checkpoints/student-dual-binary
```

Swap `--label-space binary` → `3way` (or drop the flag — `3way` is
default) for the other label space. Defaults: `google/gemma-3-270m-it`
backbone, 3 epochs, lr `2e-5`, linear schedule no warmup — pass `--model-id
google/gemma-3-1b-it` for the paper's 1B-backbone configuration. See
`--help` for `--lambda-route`/`--lambda-reason` (dual only),
`--oversample-ratio`, `--route-head-dropout`.

### 6. Pick the best checkpoint (validation-only, once)

```bash
python run.py select-router-checkpoint --variant classifier \
  --checkpoint-root checkpoints/student-classifier-binary --all --teacher qwen2.5-3b-v2 \
  --freeze-to checkpoints/student-classifier-binary-best
```

Scores every `checkpoint-<step>` under `--checkpoint-root` on
`validation` (never `test`), reports the winner, and copies it to
`--freeze-to` only if you pass that flag. `--variant` must match what you
trained in step 5 (`classifier` or `dual`).

### 7. Evaluate on test (once)

```bash
python run.py evaluate-router --variant classifier \
  --checkpoint-dir checkpoints/student-classifier-binary-best \
  --all --teacher qwen2.5-3b-v2 --split test
```

Writes `predictions.jsonl` + `classification_report.json` per dataset
under the checkpoint dir — these two files are the only output of steps
5-7 you actually need to keep long-term.

### 8. Multi-seed (what the `reproduce-table*` targets run)

```bash
./multiseed/run_multiseed_binary.sh          # 270M, binary   — Table 3
./multiseed/run_multiseed_binary_1b.sh       # 1B,   binary   — Table 4
./multiseed/run_multiseed.sh                 # 270M, 3-way    — Table 8
./multiseed/run_multiseed_1b.sh              # 1B,   3-way    — Table 10
```

Each script loops seeds 0-9 through steps 5-7 for both variants, skips
any seed/variant already done (checks for `predictions.jsonl`). Then
aggregate (run from repo root — needs repo root on `PYTHONPATH`):

```bash
PYTHONPATH=. python3 multiseed/evaluate_multiseed_binary.py --runs-dir runs_binary
PYTHONPATH=. python3 multiseed/evaluate_multiseed.py --runs-dir runs
```

Reports two separate uncertainty estimates: paired bootstrap over the
1,000-query pooled test set (within-seed), and the 10 per-seed deltas
treated as independent samples (across-seed) — a 95% CI that excludes
zero on the latter is what this paper calls a statistically reliable
result. Add `--seeds N [N ...]` to aggregate a subset.

### 9. Oracle-correctness check

Every step above scores against the *teacher's* decision (fidelity), not
ground truth. To check against the independently-built oracle instead:

```bash
PYTHONPATH=. python3 multiseed/evaluate_multiseed_oracle.py --runs-dir runs
PYTHONPATH=. python3 multiseed/evaluate_multiseed_binary_oracle.py --runs-dir runs_binary
```

This is the comparison behind the paper's central finding — a fidelity
win over the teacher does not reliably imply a correctness win over
ground truth, in most configurations tested. See `EXPERIMENTS.md`'s
"Oracle (ground truth) accuracy check" section for the full numbers.

### 10. Figures

```bash
cd research/figures && python3 make_figures.py
```

This script's numbers are hand-copied constants, not read from `runs*/`'s
predictions files or from `EXPERIMENTS.md` — if you rerun any experiment
and get a different number, you must manually update the corresponding
constant in `make_figures.py` before regenerating.

## Full pipeline at a glance

```
prepare-data → check-splits
     ↓
oracle-label (calibration → train/validation/test)
     ↓
label-data (teacher, needs oracle-labeled calibration for few-shot)
     ↓
build-student-data → build-binary-student-data
     ↓
train-student-classifier / train-student-dual   (× label-space × seed × backbone)
     ↓
select-router-checkpoint (validation, --freeze-to)
     ↓
evaluate-router (test, once)  →  predictions.jsonl / classification_report.json
     ↓
evaluate_multiseed*.py (aggregate across seeds, vs. teacher)
evaluate_multiseed*_oracle.py (aggregate across seeds, vs. ground truth)
     ↓
make_figures.py → research.tex → pdflatex
```

See [README.md](README.md) for project layout and
[DISTILLROUTER_SPEC.md](DISTILLROUTER_SPEC.md) for the design rationale.
See [EXPERIMENTS.md](EXPERIMENTS.md) for every result this pipeline has
ever produced, with the exact command that made it.
