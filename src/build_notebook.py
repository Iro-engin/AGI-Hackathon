# ruff: noqa: I001

"""Build a readable Kaggle submission notebook from local project files."""

from __future__ import annotations

import json
from pathlib import Path
from pprint import pformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "src" / "kaggle_submission_benchmark.ipynb"
CELL_COUNTER = 0

INTRO_CODE = """from __future__ import annotations

import json
from typing import Any


REQUIRED_TOP_LEVEL_FIELDS = [
    "task_id",
    "domain",
    "difficulty",
    "initial_request",
    "initial_state",
    "allowed_actions",
    "events",
    "goal_condition",
    "rubric",
]


def validate_case_structure(case: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field_name in REQUIRED_TOP_LEVEL_FIELDS:
        if field_name not in case:
            errors.append(f"missing top-level field: {field_name}")

    initial_state = case.get("initial_state", {})
    for field_name in [
        "timezone",
        "deadline",
        "participants",
        "budget",
        "required_artifacts",
        "constraints",
    ]:
        if field_name not in initial_state:
            errors.append(f"missing initial_state field: {field_name}")

    rubric = case.get("rubric", {})
    for field_name in ["outcome", "process", "recovery"]:
        if field_name not in rubric:
            errors.append(f"missing rubric field: {field_name}")

    return errors
"""

VALIDATION_DEMO_CODE = """validation_summary = []
for case in MEETING_CASES:
    errors = validate_case_structure(case)
    initial_state = case.get("initial_state", {})
    events = case.get("events", [])
    participants = initial_state.get("participants", [])
    required_artifacts = initial_state.get("required_artifacts", [])
    validation_summary.append(
        {
            "task_id": case.get("task_id", "<missing-task-id>"),
            "status": "OK" if not errors else "ERROR",
            "errors": errors,
            "events": len(events) if isinstance(events, list) else 0,
            "participants": len(participants) if isinstance(participants, list) else 0,
            "artifacts": len(required_artifacts) if isinstance(required_artifacts, list) else 0,
        }
    )

print(json.dumps(validation_summary, ensure_ascii=False, indent=2))
"""

EVALUATION_DEMO_CODE = """demo_case = next(
    case for case in MEETING_CASES if case["task_id"] == "meeting_001"
)
demo_execution_log = SAMPLE_EXECUTION_LOGS["meeting_001"]
demo_result = evaluate_case(demo_case, demo_execution_log)

evaluation_payload = {
    "task_id": demo_case["task_id"],
    "outcome_score": demo_result.outcome_score,
    "process_score": demo_result.process_score,
    "recovery_score": demo_result.recovery_score,
    "total_score": demo_result.total_score,
    "failure_labels": demo_result.failure_labels,
    "deductions": demo_result.deductions,
}

print(json.dumps(evaluation_payload, ensure_ascii=False, indent=2))
"""


def markdown_cell(text: str) -> dict[str, object]:
    global CELL_COUNTER
    CELL_COUNTER += 1
    return {
        "cell_type": "markdown",
        "id": f"cell-{CELL_COUNTER}",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code_cell(text: str) -> dict[str, object]:
    global CELL_COUNTER
    CELL_COUNTER += 1
    return {
        "cell_type": "code",
        "id": f"cell-{CELL_COUNTER}",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_cases_block() -> str:
    case_paths = sorted((ROOT / "scenarios" / "meeting").glob("*.json"))
    cases = [read_json(path) for path in case_paths]
    return (
        "MEETING_CASES = "
        + pformat(cases, width=100, sort_dicts=False)
        + "\n\n"
        + 'print(f"Loaded {len(MEETING_CASES)} benchmark cases")\n'
        + 'print([case["task_id"] for case in MEETING_CASES])\n'
    )


def build_execution_log_block() -> str:
    sample_log = read_json(ROOT / "results" / "sample_execution_meeting_001.json")
    return (
        "SAMPLE_EXECUTION_LOGS = "
        + pformat({"meeting_001": sample_log}, width=100, sort_dicts=False)
        + "\n\n"
        + 'print("Available sample execution logs:", list(SAMPLE_EXECUTION_LOGS.keys()))\n'
    )


def build_notebook() -> dict[str, object]:
    evaluator_code = (ROOT / "src" / "rule_evaluator.py").read_text(encoding="utf-8")
    cells = [
        markdown_cell(
            "# Dynamic Agent Correctness Benchmark\n\n"
            "Kaggle 提出向けの self-contained notebook です。\n\n"
            "この notebook にはベンチマーク概要、ケース定義、"
            "ルールベース evaluator、検証と評価のデモを 1 本に集約しています。\n"
        ),
        markdown_cell(
            "## Structure\n\n"
            "1. 共通 helper\n"
            "2. 会議準備ケース\n"
            "3. サンプル execution log\n"
            "4. evaluator\n"
            "5. validation summary\n"
            "6. evaluation demo\n"
        ),
        code_cell(INTRO_CODE),
        markdown_cell(
            "## Meeting Cases\n\n"
            "会議準備ドメインの 5 ケースを notebook 内に埋め込みます。\n"
        ),
        code_cell(build_cases_block()),
        markdown_cell("## Sample Execution Log\n\n評価デモ用の execution log です。\n"),
        code_cell(build_execution_log_block()),
        markdown_cell("## Rule-Based Evaluator\n\n最終成果物、プロセス、復帰力を採点します。\n"),
        code_cell(evaluator_code),
        markdown_cell("## Validation Summary\n\nケース構造の妥当性を確認します。\n"),
        code_cell(VALIDATION_DEMO_CODE),
        markdown_cell("## Evaluation Demo\n\n`meeting_001` をサンプル log で評価します。\n"),
        code_cell(EVALUATION_DEMO_CODE),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    notebook = build_notebook()
    NOTEBOOK_PATH.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
