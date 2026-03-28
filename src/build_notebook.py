# ruff: noqa: I001

"""ローカルのプロジェクトファイルから Kaggle 提出用 notebook を生成する。"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.logging_config import configure_logging

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "src" / "kaggle_submission_benchmark.ipynb"
logger = logging.getLogger(__name__)

INTRO_CODE = """from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(".")
LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


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
    \"\"\"notebook 上で手早く確認するための簡易構造検証を行う。\"\"\"

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


def configure_logging(log_level: str | None = None) -> int:
    \"\"\"環境変数または引数に基づいて notebook のログレベルを設定する。\"\"\"

    resolved_level = (log_level or os.getenv("LOG_LEVEL") or "INFO").upper()
    logging.basicConfig(level=getattr(logging, resolved_level, logging.INFO), format=LOG_FORMAT)
    logging.getLogger().setLevel(getattr(logging, resolved_level, logging.INFO))
    return logging.getLogger().level


def read_json(path: Path) -> dict[str, Any]:
    \"\"\"JSON ファイルを読み込む。\"\"\"

    return json.loads(path.read_text(encoding="utf-8"))


def load_meeting_cases() -> list[dict[str, Any]]:
    \"\"\"meeting case 一覧をファイルから読み込む。\"\"\"

    case_dir = PROJECT_ROOT / "scenarios" / "meeting"
    return [read_json(path) for path in sorted(case_dir.glob("*.json"))]


def build_case_summaries(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    \"\"\"case 一覧を俯瞰しやすい要約へ変換する。\"\"\"

    summaries: list[dict[str, Any]] = []
    for case in cases:
        initial_state = case.get("initial_state", {})
        summaries.append(
            {
                "task_id": case.get("task_id"),
                "difficulty": case.get("difficulty"),
                "event_count": len(case.get("events", [])),
                "required_artifacts": [
                    artifact.get("artifact_id")
                    for artifact in initial_state.get("required_artifacts", [])
                ],
                "dependency_count": len(initial_state.get("task_dependencies", [])),
                "constraint_count": len(initial_state.get("constraints", [])),
            }
        )
    return summaries


def load_sample_execution_logs() -> dict[str, dict[str, Any]]:
    \"\"\"サンプル execution log をファイルから読み込む。\"\"\"

    return {
        "meeting_001": read_json(
            PROJECT_ROOT / "results" / "sample_execution_meeting_001.json"
        )
    }


configure_logging()
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
demo_case_model = Case.model_validate(demo_case)
demo_execution_log = ExecutionLog.model_validate(SAMPLE_EXECUTION_LOGS["meeting_001"])
demo_result = evaluate_case(demo_case_model, demo_execution_log)

evaluation_payload = {
    "task_id": demo_case_model.task_id,
    "outcome_score": demo_result.outcome_score,
    "process_score": demo_result.process_score,
    "recovery_score": demo_result.recovery_score,
    "total_score": demo_result.total_score,
    "failure_labels": demo_result.failure_labels,
    "deductions": demo_result.deductions,
}

print(json.dumps(evaluation_payload, ensure_ascii=False, indent=2))
"""


class NotebookCellFactory:
    """安定した疑似 ID 付きで notebook cell を生成する。"""

    def __init__(self) -> None:
        self._cell_counter = 0

    def markdown(self, text: str) -> dict[str, object]:
        """markdown cell を 1 つ生成する。"""

        self._cell_counter += 1
        return {
            "cell_type": "markdown",
            "id": f"cell-{self._cell_counter}",
            "metadata": {},
            "source": text.splitlines(keepends=True),
        }

    def code(self, text: str) -> dict[str, object]:
        """code cell を 1 つ生成する。"""

        self._cell_counter += 1
        return {
            "cell_type": "code",
            "id": f"cell-{self._cell_counter}",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": text.splitlines(keepends=True),
        }


class BenchmarkNotebookBuilder:
    """ローカルソースから Kaggle notebook を組み立てる。"""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.cell_factory = NotebookCellFactory()

    def build(self) -> dict[str, object]:
        """notebook 全体の JSON ペイロードを構築する。"""

        cells = [
            self.cell_factory.markdown(
                "# Dynamic Agent Correctness Benchmark\n\n"
                "Kaggle 提出向け notebook です。\n\n"
                "この notebook にはベンチマーク概要、ケース定義、"
                "Pydantic モデル、ルールベース evaluator、検証と評価デモを集約しています。\n"
                "ケースと sample log はリポジトリ内の JSON ファイルから読み込みます。\n"
            ),
            self.cell_factory.markdown(
                "## Structure\n\n"
                "1. 共通 helper\n"
                "2. 会議準備ケース\n"
                "3. サンプル execution log\n"
                "4. Pydantic models\n"
                "5. evaluator\n"
                "6. validation summary\n"
                "7. evaluation demo\n"
            ),
            self.cell_factory.code(INTRO_CODE),
            self.cell_factory.markdown(
                "## Case Summary\n\n"
                "各 case の難易度、イベント数、成果物数、依存数を先に確認できる一覧です。\n"
            ),
            self.cell_factory.code(self._build_case_summary_block()),
            self.cell_factory.markdown(
                "## Meeting Cases\n\n"
                "会議準備ドメインの 5 ケースを、整形済み JSON として notebook 内に埋め込みます。\n"
            ),
            self.cell_factory.code(self._build_cases_block()),
            self.cell_factory.markdown(
                "## Sample Execution Log\n\n評価デモ用の execution log です。\n"
            ),
            self.cell_factory.code(self._build_execution_log_block()),
            self.cell_factory.markdown(
                "## Pydantic Models\n\ncase と execution log の型定義です。\n"
            ),
            self.cell_factory.code(self._read_source("src/models.py")),
            self.cell_factory.markdown(
                "## Rule-Based Evaluator\n\n"
                "最終成果物、質問、タスク分解、タスク順序、復帰力を採点します。\n"
            ),
            self.cell_factory.code(self._build_evaluator_block()),
            self.cell_factory.markdown(
                "## Validation Summary\n\nケース構造の妥当性を確認します。\n"
            ),
            self.cell_factory.code(VALIDATION_DEMO_CODE),
            self.cell_factory.markdown(
                "## Evaluation Demo\n\n`meeting_001` をサンプル log で評価します。\n"
            ),
            self.cell_factory.code(EVALUATION_DEMO_CODE),
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

    def _read_source(self, relative_path: str) -> str:
        """リポジトリ配下のソースファイルを読み込む。"""

        return (self.root / relative_path).read_text(encoding="utf-8")

    def _build_cases_block(self) -> str:
        """meeting case を notebook 実行時に読み込むコードを返す。"""

        return (
            "MEETING_CASES = load_meeting_cases()\n\n"
            + 'print(f"Loaded {len(MEETING_CASES)} benchmark cases")\n'
            + 'print([case["task_id"] for case in MEETING_CASES])\n'
        )

    def _build_case_summary_block(self) -> str:
        """case 一覧の読みやすいサマリー生成コードを返す。"""

        return (
            "MEETING_CASES = load_meeting_cases()\n"
            "CASE_SUMMARIES = build_case_summaries(MEETING_CASES)\n\n"
            + "print(json.dumps(CASE_SUMMARIES, ensure_ascii=False, indent=2))\n"
        )

    def _build_execution_log_block(self) -> str:
        """sample execution log を notebook 実行時に読み込むコードを返す。"""

        return (
            "SAMPLE_EXECUTION_LOGS = load_sample_execution_logs()\n\n"
            + 'print("Available sample execution logs:", list(SAMPLE_EXECUTION_LOGS.keys()))\n'
        )

    def _build_evaluator_block(self) -> str:
        """リポジトリ依存 import を外した evaluator コードを埋め込む。"""

        evaluator_code = self._read_source("src/rule_evaluator.py")
        return evaluator_code.replace("from src.models import (\n", "from __main__ import (\n")


def main() -> None:
    """`src/` 配下に notebook ファイルを生成する。"""

    configure_logging()
    logger.info("notebook 生成を開始します: output=%s", NOTEBOOK_PATH)
    notebook = BenchmarkNotebookBuilder(ROOT).build()
    NOTEBOOK_PATH.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("notebook を生成しました: output=%s", NOTEBOOK_PATH)
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
