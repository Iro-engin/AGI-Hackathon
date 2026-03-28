"""サンプル case / execution log を評価する CLI エントリポイント。"""

from __future__ import annotations

from pathlib import Path

from src.logging_config import configure_logging
from src.models import Case, ExecutionLog
from src.rule_evaluator import RuleBasedEvaluator


def main() -> None:
    """同梱サンプルを評価して結果を表示する。"""

    configure_logging()
    case = Case.from_path(Path("scenarios/meeting/meeting_001.json"))
    log = ExecutionLog.from_path(Path("results/sample_execution_meeting_001.json"))
    print(RuleBasedEvaluator().evaluate(case, log))


if __name__ == "__main__":
    main()
