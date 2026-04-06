"""Transformers でローカル LLM を使って case / execution log を検査する CLI。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.models import Case, ExecutionLog

load_dotenv()

DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DEFAULT_TORCH_DTYPE = "bfloat16"
DEFAULT_DEVICE_MAP = "auto"

SYSTEM_PROMPT = """
あなたは benchmark case の厳格なレビュアーです。
与えられた入力だけを根拠に、矛盾・不足・曖昧さ・evaluator の見落としを具体的に指摘してください。
回答は必ず日本語で返してください。
出力は必ず次の3セクションだけにしてください。

summary:
2-4文で要点をまとめる

findings:
- severity を high / medium / low のいずれかで付けた箇条書き

suggested_fixes:
- 実行可能な修正案の箇条書き

余計な前置き、コードブロック、JSON、見出しの追加は禁止です。
""".strip()

EXAMPLE_INPUT = {
    "task": "benchmark_case_review",
    "review_dimensions": [
        "case definition consistency",
        "rubric and goal alignment",
        "event impact on artifacts and state",
        "evaluator blind spots",
        "execution validity against the case",
    ],
    "case_context": {
        "metadata": {
            "task_id": "toy_001",
            "domain": "doc_prep",
            "difficulty": "easy",
        },
        "initial_request": "会議メモを1ページでまとめてください。",
        "state_snapshot": {
            "timezone": "Asia/Tokyo",
            "deadline": "2026-04-01T12:00:00+09:00",
            "participants": ["A", "B"],
            "budget": 0,
            "constraints": ["要点は3件以上"],
            "task_dependencies": [],
            "reference_data": {},
        },
        "artifact_requirements": [
            {
                "artifact_id": "memo",
                "artifact_type": "document",
                "required_fields": ["summary", "actions"],
                "state_links": ["deadline", "participants"],
                "latest_version_required": True,
            }
        ],
        "allowed_actions": ["update_plan", "create_artifact", "finalize"],
        "events": [
            {
                "turn": 2,
                "type": "state_change",
                "message": "参加者Cが必須になりました。",
                "delta": {"participants_added": ["C"]},
                "expected_artifact_updates": ["memo"],
                "expected_replan_within_turns": 1,
            }
        ],
        "goal_condition": {
            "must_satisfy_latest_state": True,
            "required_artifacts": ["memo"],
            "no_constraint_violation": True,
            "must_acknowledge_changes": True,
        },
        "rubric": {
            "outcome": ["最終成果物が参加者Cを反映している"],
            "process": ["変更を認識したあとに再評価した"],
            "recovery": ["1ターン以内に再計画を表明した"],
        },
        "notes": "簡易例",
    },
    "execution_evidence": {
        "actions_by_turn": [
            "turn 1: create_artifact | updates=[] | notes=初稿を作成",
            "turn 2: finalize | updates=['memo'] | notes=そのまま確定",
        ],
        "completed_tasks": ["draft_memo"],
        "questions_asked": [],
        "constraint_violations": [],
        "unsafe_commit": False,
        "final_state": {"participants": ["A", "B"]},
        "final_artifacts": {
            "memo": {
                "fields_completed": ["summary", "actions"],
                "field_values": {"participants": ["A", "B"]},
            }
        },
    },
}

EXAMPLE_OUTPUT = """
summary:
execution evidence は変更後の state を反映できておらず、
goal_condition と recovery rubric に未達です。
case 自体の構成は単純ですが、イベント後の影響分析を確認できる行動ログが不足しています。

findings:
- high: final_state と final_artifacts の participants に C が含まれておらず、
  must_satisfy_latest_state に違反している可能性が高い。
- high: turn 2 で finalize しており、変更認識や再計画の行動が見えないため、
  must_acknowledge_changes と recovery rubric を満たせていない。
- medium: event が memo 更新を要求している一方で、
  影響範囲の説明や再作成手順が execution evidence に残っていない。

suggested_fixes:
- event 発生直後に confirm_state か update_plan を追加し、変更認識を明示する。
- final_artifacts の participants を最新 state に合わせて更新する。
- affected artifact の特定と再計画内容を action log に残す。
""".strip()


def parse_args() -> argparse.Namespace:
    """CLI 引数を解釈する。"""

    parser = argparse.ArgumentParser(
        description="Transformers でローカル LLM を使って case / execution log を検査する"
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
        default=os.getenv("LOCAL_LLM_MODEL", DEFAULT_MODEL),
        help=f"ローカル推論で使う Hugging Face model id。既定値: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--torch-dtype",
        default=os.getenv("LOCAL_LLM_TORCH_DTYPE", DEFAULT_TORCH_DTYPE),
        choices=["auto", "bfloat16", "float16", "float32"],
        help=f"モデル読み込み時の torch dtype。既定値: {DEFAULT_TORCH_DTYPE}",
    )
    parser.add_argument(
        "--device-map",
        default=os.getenv("LOCAL_LLM_DEVICE_MAP", DEFAULT_DEVICE_MAP),
        help=f"Transformers の device_map。既定値: {DEFAULT_DEVICE_MAP}",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=int(os.getenv("LOCAL_LLM_MAX_NEW_TOKENS", "900")),
        help="生成する最大新規トークン数。既定値: 900",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.getenv("LOCAL_LLM_TEMPERATURE", "0.2")),
        help="生成温度。0 でほぼ決定論的。既定値: 0.2",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=float(os.getenv("LOCAL_LLM_TOP_P", "0.9")),
        help="nucleus sampling の top_p。既定値: 0.9",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="モデルは読み込まず、送信する messages と入力 payload の要約だけを表示する",
    )
    return parser.parse_args()


def build_local_llm_input(case: Case, execution_log: ExecutionLog | None) -> dict[str, Any]:
    """ローカル LLM に渡すための入力 schema に整形する。"""

    artifact_requirements = [
        {
            "artifact_id": artifact.artifact_id,
            "artifact_type": artifact.artifact_type,
            "required_fields": artifact.required_fields,
            "state_links": [check.state_path for check in artifact.semantic_checks],
            "latest_version_required": artifact.latest_version_required,
        }
        for artifact in case.initial_state.required_artifacts
    ]

    case_context = {
        "metadata": {
            "task_id": case.task_id,
            "domain": case.domain,
            "difficulty": case.difficulty,
        },
        "initial_request": case.initial_request,
        "state_snapshot": {
            "timezone": case.initial_state.timezone,
            "deadline": case.initial_state.deadline,
            "participants": case.initial_state.participants,
            "budget": case.initial_state.budget,
            "constraints": case.initial_state.constraints,
            "task_dependencies": [
                {
                    "before": dependency.before,
                    "after": dependency.after,
                    "reason": dependency.reason,
                }
                for dependency in case.initial_state.task_dependencies
            ],
            "reference_data": case.initial_state.reference_data,
        },
        "artifact_requirements": artifact_requirements,
        "allowed_actions": case.allowed_actions,
        "events": [event.model_dump(mode="json") for event in case.events],
        "goal_condition": case.goal_condition.model_dump(mode="json"),
        "rubric": case.rubric.model_dump(mode="json"),
        "notes": case.notes,
    }

    execution_evidence = None
    if execution_log is not None:
        execution_evidence = {
            "actions_by_turn": [
                (
                    f"turn {action.turn}: {action.action_type} | "
                    f"ack={action.acknowledged_event_turns} | "
                    f"updates={action.artifact_updates} | "
                    f"notes={action.notes or ''}"
                )
                for action in execution_log.actions
            ],
            "completed_tasks": execution_log.completed_tasks,
            "task_breakdown": [
                item.model_dump(mode="json") for item in execution_log.task_breakdown
            ],
            "questions_asked": [
                question.model_dump(mode="json") for question in execution_log.questions_asked
            ],
            "constraint_violations": execution_log.constraint_violations,
            "unsafe_commit": execution_log.unsafe_commit,
            "final_state": execution_log.final_state,
            "final_artifacts": {
                artifact_id: artifact.model_dump(mode="json")
                for artifact_id, artifact in execution_log.final_artifacts.items()
            },
        }

    return {
        "task": "benchmark_case_review",
        "review_dimensions": [
            "case definition consistency",
            "rubric and goal alignment",
            "event impact on artifacts and state",
            "evaluator blind spots",
            "execution validity against the case",
        ],
        "case_context": case_context,
        "execution_evidence": execution_evidence,
    }


def build_local_llm_messages(
    case: Case,
    execution_log: ExecutionLog | None,
) -> list[dict[str, str]]:
    """Llama 系 chat template に渡す messages を組み立てる。"""

    actual_input = build_local_llm_input(case=case, execution_log=execution_log)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "以下は形式例です。この入力に対する出力例を参考にしてください。\n"
            + json.dumps(EXAMPLE_INPUT, ensure_ascii=False, indent=2),
        },
        {"role": "assistant", "content": EXAMPLE_OUTPUT},
        {
            "role": "user",
            "content": "次を検査してください。出力形式は例と同じにしてください。\n"
            + json.dumps(actual_input, ensure_ascii=False, indent=2),
        },
    ]


def resolve_torch_dtype(dtype_name: str) -> Any:
    """文字列指定から torch dtype を解決する。"""

    if dtype_name == "auto":
        return "auto"

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "torch が見つかりません。`uv sync` で依存関係を入れてください。"
        ) from exc

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return dtype_map[dtype_name]


def build_generation_pipeline(model_id: str, torch_dtype: str, device_map: str):
    """ローカル推論用の Transformers pipeline を構築する。"""

    try:
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError(
            "transformers が見つかりません。`uv sync` で依存関係を入れてください。"
        ) from exc

    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    model_kwargs = {"torch_dtype": resolve_torch_dtype(torch_dtype)}

    return pipeline(
        task="text-generation",
        model=model_id,
        tokenizer=model_id,
        device_map=device_map,
        model_kwargs=model_kwargs,
        token=hf_token,
    )


def render_chat_prompt(generator: Any, messages: list[dict[str, str]]) -> str:
    """chat messages をモデルの chat template で単一プロンプトへ変換する。"""

    tokenizer = generator.tokenizer
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_review(
    generator: Any,
    messages: list[dict[str, str]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    """ローカル LLM でレビュー文を生成する。"""

    prompt = render_chat_prompt(generator=generator, messages=messages)
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "return_full_text": False,
        "pad_token_id": generator.tokenizer.eos_token_id,
    }

    if temperature > 0:
        generation_kwargs.update(
            {
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
            }
        )
    else:
        generation_kwargs["do_sample"] = False

    outputs = generator(prompt, **generation_kwargs)
    if not outputs:
        raise RuntimeError("ローカル LLM から生成結果が返りませんでした。")

    output_text = outputs[0].get("generated_text", "").strip()
    if not output_text:
        raise RuntimeError("ローカル LLM の出力本文を抽出できませんでした。")

    return output_text


def main() -> None:
    """CLI エントリポイント。"""

    args = parse_args()
    case = Case.from_path(args.case_path)
    execution_log = (
        ExecutionLog.from_path(args.execution_log_path) if args.execution_log_path else None
    )
    messages = build_local_llm_messages(case=case, execution_log=execution_log)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "model": args.model,
                    "torch_dtype": args.torch_dtype,
                    "device_map": args.device_map,
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "messages_preview": messages,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    generator = build_generation_pipeline(
        model_id=args.model,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
    )
    output_text = generate_review(
        generator=generator,
        messages=messages,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(output_text)


if __name__ == "__main__":
    main()
