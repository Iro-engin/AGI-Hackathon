from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

from ..models.base import Case
from ..models.finance import FinanceCase
from ..models.meeting import MeetingCase
from .base import DEFAULT_ANSWER_MODEL, DEFAULT_DECISION_MODEL, run_case_with_openai

Domain = Literal["meeting", "finance"]


def parse_args() -> argparse.Namespace:
    """inaba evaluation runner 用の CLI 引数を解釈する。"""
    parser = argparse.ArgumentParser(description="Run OpenAI-based inaba evaluation.")
    parser.add_argument(
        "--domain",
        choices=("meeting", "finance"),
        required=True,
        help="Target inaba domain.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing inaba case JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where evaluation run JSON files are written.",
    )
    parser.add_argument(
        "--decision-model",
        default=DEFAULT_DECISION_MODEL,
        help="Model used to choose Question or Do.",
    )
    parser.add_argument(
        "--answer-model",
        default=DEFAULT_ANSWER_MODEL,
        help="Model used to answer questions from hidden_info.",
    )
    parser.add_argument(
        "--max-questions-per-turn",
        type=int,
        default=4,
        help="Maximum number of questions allowed before forcing Do.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of cases to execute.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI から inaba evaluation を実行する。"""
    args = parse_args()
    input_paths = _discover_case_paths(args.input_dir, limit=args.limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    result_paths: list[Path] = []
    for case_path in input_paths:
        case = _load_case(case_path=case_path, domain=args.domain)
        run = run_case_with_openai(
            case=case,
            decision_model=args.decision_model,
            answer_model=args.answer_model,
            max_questions_per_turn=args.max_questions_per_turn,
        )
        output_path = args.output_dir / f"{case.task_id}.evaluation.json"
        output_path.write_text(run.model_dump_json(indent=2), encoding="utf-8")
        result_paths.append(output_path)

    summary = {
        "domain": args.domain,
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "decision_model": args.decision_model,
        "answer_model": args.answer_model,
        "case_count": len(result_paths),
        "result_paths": [str(path) for path in result_paths],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _discover_case_paths(input_dir: Path, limit: int | None) -> list[Path]:
    """評価対象にする case JSON 一覧を返す。"""
    paths = [
        path
        for path in sorted(input_dir.glob("*.json"))
        if not path.name.endswith("_template.json")
    ]
    if limit is not None:
        return paths[:limit]
    return paths


def _load_case(case_path: Path, domain: Domain) -> Case:
    """domain に応じて case JSON を読み込む。"""
    if domain == "meeting":
        return MeetingCase.model_validate_json(case_path.read_text(encoding="utf-8"))
    if domain == "finance":
        return FinanceCase.model_validate_json(case_path.read_text(encoding="utf-8"))
    raise ValueError(f"unsupported domain: {domain}")


if __name__ == "__main__":
    main()
