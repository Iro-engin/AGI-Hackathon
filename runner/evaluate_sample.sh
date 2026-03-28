#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

cd "$ROOT_DIR"

"$ROOT_DIR/.venv/bin/python" -c '
from pathlib import Path

from src.logging_config import configure_logging
from src.models import BenchmarkCase, ExecutionLog
from src.rule_evaluator import RuleBasedEvaluator

configure_logging()
case = BenchmarkCase.from_path(Path("scenarios/meeting/meeting_001.json"))
log = ExecutionLog.from_path(Path("results/sample_execution_meeting_001.json"))
print(RuleBasedEvaluator().evaluate(case, log))
'
