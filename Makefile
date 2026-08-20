# DistillRouter — thin wrappers around run.py / multiseed scripts.
# Nothing here has its own logic; every target just chains the same
# commands documented in full, with every flag, in PIPELINE.md.
#
#   make help              list every target
#   make venv data oracle-data teacher-data   full setup, once
#   make train-binary      one seed, both variants, binary label space (fast smoke test)
#   make reproduce-table3  this paper's primary result, 10-seed (~4h, 270M)
#
# Override TEACHER / SEED on the command line, e.g. `make train-binary SEED=7`.

PYTHON  ?= python3
TEACHER ?= qwen2.5-3b-v2
SEED    ?= 42

.PHONY: help venv data oracle-data teacher-data \
        train-binary train-3way \
        reproduce-table3 reproduce-table4 reproduce-table8 reproduce-table10 reproduce-all \
        figures clean-cache

help:
	@echo "Setup (run once, in order):"
	@echo "  make venv              create .venv, install requirements.txt"
	@echo "  make data              download + normalize datasets, check split integrity"
	@echo "  make oracle-data       oracle-label calibration/train/validation/test (ground truth, slow)"
	@echo "  make teacher-data      teacher-label + build student training data (both label spaces)"
	@echo ""
	@echo "Train one seed (smoke test, ~10-40 min depending on backbone):"
	@echo "  make train-binary      classifier + dual variants, binary label space"
	@echo "  make train-3way        classifier + dual variants, 3-way label space"
	@echo ""
	@echo "Reproduce a specific paper table (10-seed protocol, hours):"
	@echo "  make reproduce-table3   binary,  270M backbone  -> research.tex Table 3  (primary result)"
	@echo "  make reproduce-table4   binary,  1B backbone    -> research.tex Table 4"
	@echo "  make reproduce-table8   3-way,   270M backbone  -> research.tex Table 8"
	@echo "  make reproduce-table10  3-way,   1B backbone    -> research.tex Table 10"
	@echo "  make reproduce-all      all four of the above"
	@echo ""
	@echo "  make figures            regenerate every paper figure"
	@echo "  make clean-cache        remove __pycache__"
	@echo ""
	@echo "Full per-flag reference, one dataset at a time, custom configs: see PIPELINE.md."

venv:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

data:
	$(PYTHON) run.py prepare-data --all
	$(PYTHON) run.py check-splits --all

oracle-data:
	$(PYTHON) run.py oracle-label --all --split calibration
	$(PYTHON) run.py oracle-label --all --split train
	$(PYTHON) run.py oracle-label --all --split validation
	$(PYTHON) run.py oracle-label --all --split test

teacher-data:
	$(PYTHON) run.py label-data --all --teacher $(TEACHER) --split train validation test
	$(PYTHON) run.py build-student-data --all --teacher $(TEACHER) --split train
	$(PYTHON) run.py build-binary-student-data --all --teacher $(TEACHER) --split train

train-binary:
	$(PYTHON) run.py train-student-classifier --all --teacher $(TEACHER) --label-space binary --seed $(SEED) \
		--checkpoint-dir checkpoints/student-classifier-binary
	$(PYTHON) run.py select-router-checkpoint --variant classifier --all --teacher $(TEACHER) \
		--checkpoint-root checkpoints/student-classifier-binary --freeze-to checkpoints/student-classifier-binary-best
	$(PYTHON) run.py evaluate-router --variant classifier --all --teacher $(TEACHER) --split test \
		--checkpoint-dir checkpoints/student-classifier-binary-best
	$(PYTHON) run.py train-student-dual --all --teacher $(TEACHER) --label-space binary --seed $(SEED) \
		--checkpoint-dir checkpoints/student-dual-binary
	$(PYTHON) run.py select-router-checkpoint --variant dual --all --teacher $(TEACHER) \
		--checkpoint-root checkpoints/student-dual-binary --freeze-to checkpoints/student-dual-binary-best
	$(PYTHON) run.py evaluate-router --variant dual --all --teacher $(TEACHER) --split test \
		--checkpoint-dir checkpoints/student-dual-binary-best

train-3way:
	$(PYTHON) run.py train-student-classifier --all --teacher $(TEACHER) --label-space 3way --seed $(SEED) \
		--checkpoint-dir checkpoints/student-classifier
	$(PYTHON) run.py select-router-checkpoint --variant classifier --all --teacher $(TEACHER) \
		--checkpoint-root checkpoints/student-classifier --freeze-to checkpoints/student-classifier-best
	$(PYTHON) run.py evaluate-router --variant classifier --all --teacher $(TEACHER) --split test \
		--checkpoint-dir checkpoints/student-classifier-best
	$(PYTHON) run.py train-student-dual --all --teacher $(TEACHER) --label-space 3way --seed $(SEED) \
		--checkpoint-dir checkpoints/student-dual
	$(PYTHON) run.py select-router-checkpoint --variant dual --all --teacher $(TEACHER) \
		--checkpoint-root checkpoints/student-dual --freeze-to checkpoints/student-dual-best
	$(PYTHON) run.py evaluate-router --variant dual --all --teacher $(TEACHER) --split test \
		--checkpoint-dir checkpoints/student-dual-best

reproduce-table3:
	./multiseed/run_multiseed_binary.sh
	PYTHONPATH=. $(PYTHON) multiseed/evaluate_multiseed_binary.py --runs-dir runs_binary
	PYTHONPATH=. $(PYTHON) multiseed/evaluate_multiseed_binary_oracle.py --runs-dir runs_binary

reproduce-table4:
	./multiseed/run_multiseed_binary_1b.sh
	PYTHONPATH=. $(PYTHON) multiseed/evaluate_multiseed_binary.py --runs-dir runs_binary_1b
	PYTHONPATH=. $(PYTHON) multiseed/evaluate_multiseed_binary_oracle.py --runs-dir runs_binary_1b

reproduce-table8:
	./multiseed/run_multiseed.sh
	PYTHONPATH=. $(PYTHON) multiseed/evaluate_multiseed.py --runs-dir runs
	PYTHONPATH=. $(PYTHON) multiseed/evaluate_multiseed_oracle.py --runs-dir runs

reproduce-table10:
	./multiseed/run_multiseed_1b.sh
	PYTHONPATH=. $(PYTHON) multiseed/evaluate_multiseed.py --runs-dir runs_1b
	PYTHONPATH=. $(PYTHON) multiseed/evaluate_multiseed_oracle.py --runs-dir runs_1b

reproduce-all: reproduce-table3 reproduce-table4 reproduce-table8 reproduce-table10

figures:
	cd research/figures && $(PYTHON) make_figures.py

clean-cache:
	find . -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} +
