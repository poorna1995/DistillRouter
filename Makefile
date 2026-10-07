# DistillRouter — thin wrappers around run.py. Binary routing (small / large).
# Nothing here has its own logic; every target chains run.py commands.
#
#   make help              list every target
#   make venv data oracle-data teacher-data   full setup, once
#   make train             one seed, both student variants
#
# Override TEACHER / SEED on the command line, e.g. `make train SEED=7`.
# The submitted paper's targets (3-way, reproduce-table*) live in git tag submitted-v1.

PYTHON  ?= python3
TEACHER ?= qwen2.5-14b
SEED    ?= 42

.PHONY: help venv data oracle-data teacher-data train figures clean-cache

help:
	@echo "Setup (run once, in order):"
	@echo "  make venv              create .venv, install requirements.txt"
	@echo "  make data              download + normalize datasets, check split integrity"
	@echo "  make oracle-data       run both candidates on every split (true labels, slow)"
	@echo "  make teacher-data      teacher-label + build student training data"
	@echo ""
	@echo "Train one seed:"
	@echo "  make train             route-only + reasoning-plus-route students"
	@echo ""
	@echo "  make figures           regenerate every paper figure"
	@echo "  make clean-cache       remove __pycache__"

venv:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

data:
	$(PYTHON) run.py prepare-data --all
	$(PYTHON) run.py check-splits --all

oracle-data:
	$(PYTHON) run.py oracle-label --all --split calibration
	$(PYTHON) run.py oracle-label --all --split validation
	$(PYTHON) run.py oracle-label --all --split test
	$(PYTHON) run.py oracle-label --all --split train

teacher-data:
	$(PYTHON) run.py label-data --all --teacher $(TEACHER) --split validation test train
	$(PYTHON) run.py build-student-data --all --teacher $(TEACHER) --split train

train:
	$(PYTHON) run.py train-student-classifier --all --teacher $(TEACHER) --seed $(SEED) \
		--checkpoint-dir checkpoints/student-classifier
	$(PYTHON) run.py select-router-checkpoint --variant classifier --all --teacher $(TEACHER) \
		--checkpoint-root checkpoints/student-classifier --freeze-to checkpoints/student-classifier-best
	$(PYTHON) run.py evaluate-router --variant classifier --all --teacher $(TEACHER) --split test \
		--checkpoint-dir checkpoints/student-classifier-best
	$(PYTHON) run.py train-student-dual --all --teacher $(TEACHER) --seed $(SEED) \
		--checkpoint-dir checkpoints/student-dual
	$(PYTHON) run.py select-router-checkpoint --variant dual --all --teacher $(TEACHER) \
		--checkpoint-root checkpoints/student-dual --freeze-to checkpoints/student-dual-best
	$(PYTHON) run.py evaluate-router --variant dual --all --teacher $(TEACHER) --split test \
		--checkpoint-dir checkpoints/student-dual-best

figures:
	cd research/figures && $(PYTHON) make_figures.py

clean-cache:
	find . -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} +
