"""OpenAI Responses API を使って case / execution log を検査する CLI。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from src.models import Case, ExecutionLog

load_dotenv()
DEFAULT_MODEL = "gpt-4.1-mini"


def parse_args() -> argparse.Namespace:
    """CLI 引数を解釈する。"""

    parser = argparse.ArgumentParser(
        description="OpenAI Responses API で case / execution log を検査する"
    )
    parser.add_argument(
        "--case",
        dest="case_path",
        type=Path,
        required=True,
        help="検査対象の case JSON パス",
    )
    parser.add_argument(
        "--execution-log",
        dest="execution_log_path",
        type=Path,
        help="任意の execution log JSON パス",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
        help=f"Responses API で使うモデル名。既定値: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="API は呼ばず、送信内容の要約だけを表示する",
    )
    return parser.parse_args()


def build_inspection_prompt(case: Case, execution_log: ExecutionLog | None) -> str:
    """case と execution log を検査させるためのプロンプトを組み立てる。"""

    sections = [
        "あなたは benchmark case のレビュアーです。",
        "次の観点で簡潔に検査してください。",
        "1. case 定義の矛盾や不足",
        "2. events が initial_state / required_artifacts に与える影響",
        "3. evaluator で見落としそうな観点",
        "4. execution log がある場合は、case に対して妥当か",
        "",
        "出力形式:",
        "- summary: 2-4文",
        "- findings: 箇条書き",
        "- suggested_fixes: 箇条書き",
        "",
        "Case JSON:",
        json.dumps(case.model_dump(mode="json"), ensure_ascii=False, indent=2),
    ]

    if execution_log is not None:
        sections.extend(
            [
                "",
                "ExecutionLog JSON:",
                json.dumps(execution_log.model_dump(mode="json"), ensure_ascii=False, indent=2),
            ]
        )

    return "\n".join(sections)


def main() -> None:
    """CLI エントリポイント。"""

    args = parse_args()
    case = Case.from_path(args.case_path)
    execution_log = (
        ExecutionLog.from_path(args.execution_log_path) if args.execution_log_path else None
    )
    prompt = build_inspection_prompt(case=case, execution_log=execution_log)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "model": args.model,
                    "case_path": str(args.case_path),
                    "execution_log_path": (
                        str(args.execution_log_path) if args.execution_log_path else None
                    ),
                    "prompt_preview": prompt[:2000],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY が未設定です。")

    client = OpenAI()
    response = client.responses.create(
        model=args.model,
        instructions=(
            "You are a strict benchmark reviewer. Be concrete, concise, and point out "
            "contradictions, ambiguity, and evaluator gaps."
        ),
        input=prompt,
    )
    print(response.output_text)


if __name__ == "__main__":
    main()
