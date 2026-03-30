UV := uv

.PHONY: setup export-requirements lint build-notebook eval-sample inspect-case

setup:
	$(UV) sync

export-requirements:
	$(UV) export --format requirements-txt --no-hashes -o requirements.txt

lint:
	$(UV) run ruff check src

build-notebook:
	$(UV) run python -m src.build_notebook

eval-sample:
	$(UV) run python -m src.evaluate_sample

inspect-case:
	$(UV) run python -m src.openai_case_inspector \
		--case scenarios/meeting/meeting_001.json \
		--execution-log results/sample_execution_meeting_001.json
