UV := uv

.PHONY: setup export-requirements lint build-notebook eval-sample eval-batch scaffold-logs inspect-case inspect-case-gemini inspect-all inspect-all-gemini

setup:
	$(UV) sync

export-requirements:
	$(UV) export --format requirements-txt --no-hashes -o requirements.txt

lint:
	$(UV) run ruff check src

build-notebook:
	$(UV) run python -m src.build_notebook

eval-sample:
	$(UV) run python -m src.evaluate_sample --output results/eval_sample.json

eval-batch:
	$(UV) run python -m src.evaluate_sample --batch

scaffold-logs:
	$(UV) run python -m src.evaluate_sample --batch --write-missing-logs

inspect-case:
	$(UV) run python -m src.openai_case_inspector \
		--case scenarios/meeting/meeting_001.json \
		--execution-log results/sample_execution_meeting_001.json \
		--output results/openai_inspection_meeting_001.json

inspect-case-gemini:
	$(UV) run python -m src.openai_case_inspector \
		--provider gemini \
		--case scenarios/meeting/meeting_001.json \
		--execution-log results/sample_execution_meeting_001.json \
		--output results/gemini_inspection_meeting_001.json

inspect-all:
	$(UV) run python -m src.openai_case_inspector \
		--batch \
		--output-dir results/openai_inspections

inspect-all-gemini:
	$(UV) run python -m src.openai_case_inspector \
		--provider gemini \
		--batch \
		--output-dir results/gemini_inspections
