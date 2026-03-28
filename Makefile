VENV_PYTHON := .venv/bin/python

.PHONY: setup lint build-notebook eval-sample inspect-case

setup:
	python3 -m venv .venv
	$(VENV_PYTHON) -m pip install -r requirements.txt

lint:
	$(VENV_PYTHON) -m ruff check src

build-notebook:
	$(VENV_PYTHON) -m src.build_notebook

eval-sample:
	$(VENV_PYTHON) -m src.evaluate_sample

inspect-case:
	$(VENV_PYTHON) -m src.openai_case_inspector \
		--case scenarios/meeting/meeting_001.json \
		--execution-log results/sample_execution_meeting_001.json
