UV := uv

.PHONY: setup export-requirements lint build-notebook eval-sample eval-batch scaffold-logs inspect-case inspect-case-gemini inspect-all inspect-all-gemini gen-inaba-meeting-cases gen-inaba-finance-cases gen-inaba-finance-cases-openai gen-inaba-meeting-cases-openai gen-inaba-meeting-cases-all gen-inaba-finance-cases-all gen-inaba-meeting-cases-openai-all gen-inaba-finance-cases-openai-all execute-inaba-meeting execute-inaba-finance eval-inaba-meeting eval-inaba-finance

INABA_MEETING_COUNT ?= 15
INABA_MEETING_COUNT_PER_DIFFICULTY ?= 5
INABA_MEETING_DIFFICULTY ?= medium
INABA_MEETING_OUTPUT_DIR ?= results/generated/inaba/meeting
INABA_MEETING_SEED ?=
INABA_MEETING_TASK_ID_PREFIX ?= meeting
INABA_MEETING_OPENAI_MODEL ?= gpt-4.1-mini
INABA_MEETING_OPENAI_OUTPUT_DIR ?= results/generated/inaba/meeting_openai
INABA_MEETING_OPENAI_TASK_ID_PREFIX ?= meeting_openai
INABA_MEETING_EVAL_INPUT_DIR ?= results/generated/inaba/meeting_openai
INABA_EVAL_MODEL_DIR_SUFFIX ?= $(INABA_EVAL_DECISION_MODEL)_$(INABA_EVAL_ANSWER_MODEL)
INABA_MEETING_EVAL_OUTPUT_DIR ?= results/execute/inaba/meeting_openai/$(INABA_EVAL_MODEL_DIR_SUFFIX)
INABA_FINANCE_COUNT ?= 15
INABA_FINANCE_COUNT_PER_DIFFICULTY ?= 5
INABA_FINANCE_DIFFICULTY ?= medium
INABA_FINANCE_OUTPUT_DIR ?= results/generated/inaba/finance
INABA_FINANCE_SEED ?=
INABA_FINANCE_TASK_ID_PREFIX ?= finance
INABA_FINANCE_OPENAI_MODEL ?= gpt-4.1-mini
INABA_FINANCE_OPENAI_OUTPUT_DIR ?= results/generated/inaba/finance_openai
INABA_FINANCE_OPENAI_TASK_ID_PREFIX ?= finance_openai
INABA_FINANCE_EVAL_INPUT_DIR ?= results/generated/inaba/finance_openai
INABA_FINANCE_EVAL_OUTPUT_DIR ?= results/execute/inaba/finance_openai/$(INABA_EVAL_MODEL_DIR_SUFFIX)
INABA_EVAL_DECISION_MODEL ?= gpt-4.1-mini
INABA_EVAL_ANSWER_MODEL ?= gpt-4o-mini
INABA_EVAL_MAX_QUESTIONS ?= 4
INABA_EVAL_LIMIT ?=

INABA_MEETING_SCORE_CASES_DIR ?= results/generated/inaba/meeting_openai
INABA_MEETING_SCORE_EXEC_DIR ?= $(INABA_MEETING_EVAL_OUTPUT_DIR)
INABA_SCORE_MODEL_DIR_SUFFIX ?= $(INABA_SCORE_JUDGE_MODEL)
INABA_MEETING_SCORE_OUTPUT_DIR ?= results/evaluate/inaba/meeting_openai/$(INABA_SCORE_MODEL_DIR_SUFFIX)
INABA_FINANCE_SCORE_CASES_DIR ?= results/generated/inaba/finance_openai
INABA_FINANCE_SCORE_EXEC_DIR ?= $(INABA_FINANCE_EVAL_OUTPUT_DIR)
INABA_FINANCE_SCORE_OUTPUT_DIR ?= results/evaluate/inaba/finance_openai/$(INABA_SCORE_MODEL_DIR_SUFFIX)
INABA_SCORE_JUDGE_MODEL ?= gpt-4.1-mini
INABA_SCORE_EARLY_DISCOVERY_N ?= 2
INABA_SCORE_LIMIT ?=

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

gen-inaba-meeting-cases:
	$(UV) run python -m src.inaba.gen_case.base.meeting \
		--count $(INABA_MEETING_COUNT) \
		--difficulty $(INABA_MEETING_DIFFICULTY) \
		--output-dir $(INABA_MEETING_OUTPUT_DIR) \
		--task-id-prefix $(INABA_MEETING_TASK_ID_PREFIX) \
		$(if $(INABA_MEETING_SEED),--seed $(INABA_MEETING_SEED),)

gen-inaba-meeting-cases-openai:
	$(UV) run python -m src.inaba.gen_case.openai.meeting \
		--count $(INABA_MEETING_COUNT) \
		--difficulty $(INABA_MEETING_DIFFICULTY) \
		--model $(INABA_MEETING_OPENAI_MODEL) \
		--output-dir $(INABA_MEETING_OPENAI_OUTPUT_DIR) \
		--task-id-prefix $(INABA_MEETING_OPENAI_TASK_ID_PREFIX) \
		$(if $(INABA_MEETING_SEED),--seed $(INABA_MEETING_SEED),)

gen-inaba-finance-cases:
	$(UV) run python -m src.inaba.gen_case.base.finance \
		--count $(INABA_FINANCE_COUNT) \
		--difficulty $(INABA_FINANCE_DIFFICULTY) \
		--output-dir $(INABA_FINANCE_OUTPUT_DIR) \
		--task-id-prefix $(INABA_FINANCE_TASK_ID_PREFIX) \
		$(if $(INABA_FINANCE_SEED),--seed $(INABA_FINANCE_SEED),)

gen-inaba-finance-cases-openai:
	$(UV) run python -m src.inaba.gen_case.openai.finance \
		--count $(INABA_FINANCE_COUNT) \
		--difficulty $(INABA_FINANCE_DIFFICULTY) \
		--model $(INABA_FINANCE_OPENAI_MODEL) \
		--output-dir $(INABA_FINANCE_OPENAI_OUTPUT_DIR) \
		--task-id-prefix $(INABA_FINANCE_OPENAI_TASK_ID_PREFIX) \
		$(if $(INABA_FINANCE_SEED),--seed $(INABA_FINANCE_SEED),)

gen-inaba-meeting-cases-all:
	$(UV) run python -m src.inaba.gen_case.base.meeting \
		--count $(INABA_MEETING_COUNT_PER_DIFFICULTY) \
		--output-dir $(INABA_MEETING_OUTPUT_DIR) \
		--task-id-prefix $(INABA_MEETING_TASK_ID_PREFIX) \
		--all-difficulties \
		$(if $(INABA_MEETING_SEED),--seed $(INABA_MEETING_SEED),)

gen-inaba-finance-cases-all:
	$(UV) run python -m src.inaba.gen_case.base.finance \
		--count $(INABA_FINANCE_COUNT_PER_DIFFICULTY) \
		--output-dir $(INABA_FINANCE_OUTPUT_DIR) \
		--task-id-prefix $(INABA_FINANCE_TASK_ID_PREFIX) \
		--all-difficulties \
		$(if $(INABA_FINANCE_SEED),--seed $(INABA_FINANCE_SEED),)

gen-inaba-meeting-cases-openai-all:
	$(UV) run python -m src.inaba.gen_case.openai.meeting \
		--count $(INABA_MEETING_COUNT_PER_DIFFICULTY) \
		--model $(INABA_MEETING_OPENAI_MODEL) \
		--output-dir $(INABA_MEETING_OPENAI_OUTPUT_DIR) \
		--task-id-prefix $(INABA_MEETING_OPENAI_TASK_ID_PREFIX) \
		--all-difficulties \
		$(if $(INABA_MEETING_SEED),--seed $(INABA_MEETING_SEED),)

gen-inaba-finance-cases-openai-all:
	$(UV) run python -m src.inaba.gen_case.openai.finance \
		--count $(INABA_FINANCE_COUNT_PER_DIFFICULTY) \
		--model $(INABA_FINANCE_OPENAI_MODEL) \
		--output-dir $(INABA_FINANCE_OPENAI_OUTPUT_DIR) \
		--task-id-prefix $(INABA_FINANCE_OPENAI_TASK_ID_PREFIX) \
		--all-difficulties \
		$(if $(INABA_FINANCE_SEED),--seed $(INABA_FINANCE_SEED),)

execute-inaba-meeting:
	$(UV) run python -m src.inaba.execute.run \
		--domain meeting \
		--input-dir $(INABA_MEETING_EVAL_INPUT_DIR) \
		--output-dir $(INABA_MEETING_EVAL_OUTPUT_DIR) \
		--decision-model $(INABA_EVAL_DECISION_MODEL) \
		--answer-model $(INABA_EVAL_ANSWER_MODEL) \
		--max-questions-per-turn $(INABA_EVAL_MAX_QUESTIONS) \
		$(if $(INABA_EVAL_LIMIT),--limit $(INABA_EVAL_LIMIT),)

execute-inaba-finance:
	$(UV) run python -m src.inaba.execute.run \
		--domain finance \
		--input-dir $(INABA_FINANCE_EVAL_INPUT_DIR) \
		--output-dir $(INABA_FINANCE_EVAL_OUTPUT_DIR) \
		--decision-model $(INABA_EVAL_DECISION_MODEL) \
		--answer-model $(INABA_EVAL_ANSWER_MODEL) \
		--max-questions-per-turn $(INABA_EVAL_MAX_QUESTIONS) \
		$(if $(INABA_EVAL_LIMIT),--limit $(INABA_EVAL_LIMIT),)

eval-inaba-meeting:
	$(UV) run python -m src.inaba.evaluate.run \
		--domain meeting \
		--cases-dir $(INABA_MEETING_SCORE_CASES_DIR) \
		--executions-dir $(INABA_MEETING_SCORE_EXEC_DIR) \
		--output-dir $(INABA_MEETING_SCORE_OUTPUT_DIR) \
		--judge-model $(INABA_SCORE_JUDGE_MODEL) \
		--early-discovery-n $(INABA_SCORE_EARLY_DISCOVERY_N) \
		$(if $(INABA_SCORE_LIMIT),--limit $(INABA_SCORE_LIMIT),)

eval-inaba-finance:
	$(UV) run python -m src.inaba.evaluate.run \
		--domain finance \
		--cases-dir $(INABA_FINANCE_SCORE_CASES_DIR) \
		--executions-dir $(INABA_FINANCE_SCORE_EXEC_DIR) \
		--output-dir $(INABA_FINANCE_SCORE_OUTPUT_DIR) \
		--judge-model $(INABA_SCORE_JUDGE_MODEL) \
		--early-discovery-n $(INABA_SCORE_EARLY_DISCOVERY_N) \
		$(if $(INABA_SCORE_LIMIT),--limit $(INABA_SCORE_LIMIT),)
