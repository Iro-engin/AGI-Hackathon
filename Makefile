UV := uv
UV_CACHE_DIR ?= /tmp/uv-cache
SCIENCE_COUNT ?= 5
SCIENCE_CASE ?= scenarios/science/science_001.json
SCIENCE_CASE_DIR ?= scenarios/science
SCIENCE_EXECUTION_LOG ?= results/science/science_result_001.json
SCIENCE_RESULTS_DIR ?= results/science
SCIENCE_EVALUATION ?= evaluate/science/science_eval_001.json
SCIENCE_EVALUATION_DIR ?= evaluate/science

.PHONY: setup export-requirements lint build-notebook eval-sample inspect-case science-dry-run science-generate science-execute science-execute-all science-evaluate science-evaluate-all

setup:
	$(UV) sync

export-requirements:
	$(UV) export --format requirements-txt --no-hashes -o requirements.txt

lint:
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) run ruff check src

build-notebook:
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) run python -m src.build_notebook

eval-sample:
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) run python -m src.evaluate_sample

inspect-case:
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) run python -m src.openai_case_inspector \
		--case scenarios/meeting/meeting_001.json \
		--execution-log results/sample_execution_meeting_001.json

science-dry-run:
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) run python -m src.science.generate_cases \
		--dry-run \
		--count $(SCIENCE_COUNT)

science-generate:
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) run python -m src.science.generate_cases \
		--count $(SCIENCE_COUNT) \
		--overwrite

science-execute:
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) run python -m src.science.executor \
		--case $(SCIENCE_CASE) \
		--output $(SCIENCE_EXECUTION_LOG)

science-execute-all:
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) run python -m src.science.executor \
		--case-dir $(SCIENCE_CASE_DIR) \
		--output-dir $(SCIENCE_RESULTS_DIR)

science-evaluate:
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) run python -m src.science.evaluator \
		--case $(SCIENCE_CASE) \
		--execution-log $(SCIENCE_EXECUTION_LOG) \
		--output $(SCIENCE_EVALUATION)

science-evaluate-all:
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) run python -m src.science.evaluator \
		--case-dir $(SCIENCE_CASE_DIR) \
		--execution-log-dir $(SCIENCE_RESULTS_DIR) \
		--output-dir $(SCIENCE_EVALUATION_DIR)
