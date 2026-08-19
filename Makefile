SHELL := /bin/sh
.DEFAULT_GOAL := help

PYTHON ?= python
EXPERIMENT_CONFIG ?= configs/experiment.yaml
OUTPUT_ROOT ?= outputs/runs
RUN_ID ?=
WITH_FILTER ?= 0
STRICT ?= 1
WEIGHTS_JSON ?=
EXTRA_ARGS ?=

CLI = PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$(CURDIR)/src" $(PYTHON) -m fashionrec

RUN_ID_ARG = $(if $(strip $(RUN_ID)),--run-id "$(RUN_ID)")
FILTER_ARG = $(if $(filter 1 true yes,$(WITH_FILTER)),--with-filter)
NO_STRICT_ARG = $(if $(filter 0 false no,$(STRICT)),--no-strict)
WEIGHTS_ARG = $(if $(strip $(WEIGHTS_JSON)),--weights-json "$(WEIGHTS_JSON)")

PIPELINE = $(CLI) pipeline \
	--experiment-config "$(EXPERIMENT_CONFIG)" \
	--output-root "$(OUTPUT_ROOT)" \
	$(RUN_ID_ARG) $(NO_STRICT_ARG)

SKIP_DATA := --skip-data-prep
SKIP_TRAIN := --skip-train
SKIP_SELECT := --skip-checkpoint-selection
SKIP_RECALL := --skip-recall
SKIP_CANDIDATES := --skip-candidates
SKIP_WEIGHTS := --skip-weight-search
SKIP_VALID := --skip-valid-eval
SKIP_TEST := --skip-test-eval

.PHONY: help pipeline data train select-checkpoint recall candidates weights evaluate evaluate-valid evaluate-test downstream test check require-run-id

help:
	@printf '%s\n' \
		'FashionRec training commands' \
		'' \
		'  make pipeline [WITH_FILTER=1]       Run the complete formal pipeline' \
		'  make data RUN_ID=<id>               Prepare and split data only' \
		'  make train RUN_ID=<id>              Train SASRecF only' \
		'  make select-checkpoint RUN_ID=<id>  Select checkpoint on valid MAP@12' \
		'  make recall RUN_ID=<id>             Export valid and test sequence recall' \
		'  make candidates RUN_ID=<id>         Materialize valid and test candidates' \
		'  make weights RUN_ID=<id>            Search fusion weights on valid only' \
		'  make evaluate RUN_ID=<id>           Evaluate valid and test with fixed weights' \
		'  make downstream RUN_ID=<id>         Run all stages after model training' \
		'  make test                            Run the unit test suite' \
		'  make check                           Run tests and CLI smoke checks' \
		'' \
		'Variables: PYTHON, EXPERIMENT_CONFIG, OUTPUT_ROOT, RUN_ID,' \
		'           WITH_FILTER=0|1, STRICT=0|1, WEIGHTS_JSON, EXTRA_ARGS'

require-run-id:
	@if [ -z "$(strip $(RUN_ID))" ]; then \
		echo 'RUN_ID is required for staged commands (example: make train RUN_ID=exp-001).'; \
		exit 2; \
	fi

pipeline:
	$(PIPELINE) $(FILTER_ARG) $(WEIGHTS_ARG) $(EXTRA_ARGS)

data: require-run-id
	$(PIPELINE) $(FILTER_ARG) $(SKIP_TRAIN) $(SKIP_SELECT) $(SKIP_RECALL) $(SKIP_CANDIDATES) $(SKIP_WEIGHTS) $(SKIP_VALID) $(SKIP_TEST) $(EXTRA_ARGS)

train: require-run-id
	$(PIPELINE) $(SKIP_DATA) $(SKIP_SELECT) $(SKIP_RECALL) $(SKIP_CANDIDATES) $(SKIP_WEIGHTS) $(SKIP_VALID) $(SKIP_TEST) $(EXTRA_ARGS)

select-checkpoint: require-run-id
	$(PIPELINE) $(SKIP_DATA) $(SKIP_TRAIN) $(SKIP_RECALL) $(SKIP_CANDIDATES) $(SKIP_WEIGHTS) $(SKIP_VALID) $(SKIP_TEST) $(EXTRA_ARGS)

recall: require-run-id
	$(PIPELINE) $(SKIP_DATA) $(SKIP_TRAIN) $(SKIP_SELECT) $(SKIP_CANDIDATES) $(SKIP_WEIGHTS) $(SKIP_VALID) $(SKIP_TEST) $(EXTRA_ARGS)

candidates: require-run-id
	$(PIPELINE) $(SKIP_DATA) $(SKIP_TRAIN) $(SKIP_SELECT) $(SKIP_RECALL) $(SKIP_WEIGHTS) $(SKIP_VALID) $(SKIP_TEST) $(EXTRA_ARGS)

weights: require-run-id
	$(PIPELINE) $(SKIP_DATA) $(SKIP_TRAIN) $(SKIP_SELECT) $(SKIP_RECALL) $(SKIP_CANDIDATES) $(SKIP_VALID) $(SKIP_TEST) $(WEIGHTS_ARG) $(EXTRA_ARGS)

evaluate: require-run-id
	$(PIPELINE) $(SKIP_DATA) $(SKIP_TRAIN) $(SKIP_SELECT) $(SKIP_RECALL) $(SKIP_CANDIDATES) $(SKIP_WEIGHTS) $(WEIGHTS_ARG) $(EXTRA_ARGS)

evaluate-valid: require-run-id
	$(PIPELINE) $(SKIP_DATA) $(SKIP_TRAIN) $(SKIP_SELECT) $(SKIP_RECALL) $(SKIP_CANDIDATES) $(SKIP_WEIGHTS) $(SKIP_TEST) $(WEIGHTS_ARG) $(EXTRA_ARGS)

evaluate-test: require-run-id
	$(PIPELINE) $(SKIP_DATA) $(SKIP_TRAIN) $(SKIP_SELECT) $(SKIP_RECALL) $(SKIP_CANDIDATES) $(SKIP_WEIGHTS) $(SKIP_VALID) $(WEIGHTS_ARG) $(EXTRA_ARGS)

downstream: require-run-id
	$(PIPELINE) $(SKIP_DATA) $(SKIP_TRAIN) $(WEIGHTS_ARG) $(EXTRA_ARGS)

test:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$(CURDIR)/src" $(PYTHON) -m pytest -p no:cacheprovider -q

check: test
	$(CLI) --help >/dev/null
	$(CLI) data --help >/dev/null
	$(CLI) profile-data --help >/dev/null
	$(CLI) train --help >/dev/null
	$(CLI) select-checkpoint --help >/dev/null
	$(CLI) recall --help >/dev/null
	$(CLI) candidates --help >/dev/null
	$(CLI) weights --help >/dev/null
	$(CLI) evaluate --help >/dev/null
	$(CLI) baseline --help >/dev/null
	$(CLI) pipeline --help >/dev/null
	@echo 'CLI smoke checks passed.'
