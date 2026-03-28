VENV_PYTHON := .venv/bin/python

.PHONY: setup lint build-notebook eval-sample

setup:
	./runner/bootstrap.sh

lint:
	$(VENV_PYTHON) -m ruff check src

build-notebook:
	./runner/build_notebook.sh

eval-sample:
	./runner/evaluate_sample.sh
