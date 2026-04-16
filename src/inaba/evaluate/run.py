"""inaba evaluate runner: 実行ログを採点して EvalResult を書き出す。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv

from ..execute.base import EvaluationRun
from ..models.base import Case, Do
from ..models.finance import FinanceCase
from ..models.meeting import MeetingCase
from .base import (
    DEFAULT_JUDGE_MODEL,
    DomainAdapter,
    EvalResult,
    TurnContext,
    TurnRecord,
    compute_eval_result,
)

load_dotenv()

Domain = Literal["meeting", "finance"]


def parse_args() -> argparse.Namespace:
    """inaba 採点 runner 用の CLI 引数を解釈する。"""
    parser = argparse.ArgumentParser(description="Score inaba evaluation runs.")
    parser.add_argument(
        "--domain",
        choices=("meeting", "finance"),
        required=True,
        help="Target inaba domain.",
    )
    parser.add_argument(
        "--cases-dir",
        type=Path,
        required=True,
        help="Directory containing the original case JSON files.",
    )
    parser.add_argument(
        "--executions-dir",
        type=Path,
        required=True,
        help="Directory containing execution run JSON files (*.evaluation.json).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where per-case score JSON files are written.",
    )
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help="OpenAI model used for require-coverage and feasibility judgment.",
    )
    parser.add_argument(
        "--early-discovery-n",
        type=int,
        default=2,
        help="N used in Early Discovery Rate (requires covered within first N questions).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of execution files to score.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI から inaba 採点を実行する。"""
    args = parse_args()
    adapter = _get_domain_adapter(args.domain, judge_model=args.judge_model)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    case_map = _load_case_map(args.cases_dir, domain=args.domain)
    print(f"loaded {len(case_map)} cases from {args.cases_dir}")

    exec_paths = _discover_execution_paths(args.executions_dir, limit=args.limit)
    all_results: list[dict[str, Any]] = []

    for exec_path in exec_paths:
        run = EvaluationRun.model_validate_json(exec_path.read_text(encoding="utf-8"))
        case = case_map.get(run.task_id)
        if case is None:
            print(f"[WARN] no case matched task_id={run.task_id!r} in {args.cases_dir}")
            continue
        records = build_turn_records(run=run, case=case, adapter=adapter)
        result = compute_eval_result(records, early_discovery_n=args.early_discovery_n)

        output = _format_result(
            task_id=run.task_id,
            result=result,
            records=records,
            decision_model=run.decision_model,
            difficulty=case.difficulty,
        )
        out_path = args.output_dir / f"{run.task_id}.score.json"
        out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        all_results.append(output)
        print(f"scored: {run.task_id}")

    summary = _aggregate_summary(
        domain=args.domain,
        results=all_results,
        early_discovery_n=args.early_discovery_n,
    )
    summary_path = args.output_dir / "_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_turn_records(
    *,
    run: EvaluationRun,
    case: Case,
    adapter: DomainAdapter,
) -> list[TurnRecord]:
    """EvaluationRun と Case から TurnRecord のリストを構築する。"""
    records: list[TurnRecord] = []

    for turn_exec in run.turns:
        event_index = turn_exec.event_index
        if event_index is None:
            # 初期ターン
            should_require = list(case.initial_requires)
            exp_do_obj: Do | None = case.initial_do
            context = _build_turn_context_from_case(case=case, event_index=None)
        else:
            event = case.events[event_index]
            should_require = list(event.should_require)
            exp_do_obj = event.exp_do
            context = _build_turn_context_from_case(case=case, event_index=event_index)

        questions_asked = [qa.question for qa in turn_exec.question_answers]

        # OpenAI で各質問が should_require のどの項目をカバーするか判定する
        covered_per_question, covered_requires = adapter.build_covered_requires(
            should_require,
            questions_asked,
        )

        # 冗長質問: 既に前の質問でカバー済みの項目しかカバーしない質問
        redundant_questions = _find_redundant_questions(
            questions_asked=questions_asked,
            covered_per_question=covered_per_question,
        )

        did_without_full_coverage = len(covered_requires) < len(should_require)

        # final_do の feasibility と exact_match を判定する
        try:
            final_do_obj = adapter.parse_do(turn_exec.final_do)
            do_is_feasible = adapter.is_feasible(final_do_obj, context)
            do_exact_match = (
                adapter.is_exact_match(final_do_obj, exp_do_obj)
                if exp_do_obj is not None
                else False
            )
        except Exception:  # noqa: BLE001
            final_do_obj = None
            do_is_feasible = False
            do_exact_match = False

        records.append(
            TurnRecord(
                event_index=event_index if event_index is not None else -1,
                should_require=should_require,
                questions_asked=questions_asked,
                covered_per_question=covered_per_question,
                covered_requires=covered_requires,
                redundant_questions=redundant_questions,
                did_without_full_coverage=did_without_full_coverage,
                final_do=final_do_obj,
                exp_do=exp_do_obj,
                do_is_feasible=do_is_feasible,
                do_exact_match=do_exact_match,
            )
        )

    return records


def _find_redundant_questions(
    *,
    questions_asked: list[str],
    covered_per_question: list[list[str]],
) -> list[str]:
    """既カバー済みの項目しか含まない質問を冗長質問として返す。"""
    already_covered: set[str] = set()
    redundant: list[str] = []
    for question, covers in zip(questions_asked, covered_per_question, strict=True):
        new_covers = set(covers) - already_covered
        if covers and not new_covers:
            # カバーしようとしているが全て既カバー済み
            redundant.append(question)
        already_covered.update(covers)
    return redundant


def _build_turn_context_from_case(*, case: Case, event_index: int | None) -> TurnContext:
    """case から TurnContext（evaluate 用）を組み立てる。"""
    initial_state = case.initial_state
    if event_index is None:
        return TurnContext(
            initial_got_info=initial_state.got_info.model_dump(mode="json"),
            stage="initial",
            current_constraint=list(initial_state.constraint),
            event_message=None,
            current_got_info=initial_state.got_info.model_dump(mode="json"),
            should_require=list(case.initial_requires),
            question_answers=[],
            previous_do=None,
            current_hidden_info=initial_state.hidden_info.model_dump(mode="json"),
        )
    event = case.events[event_index]
    state = event.state_before
    return TurnContext(
        initial_got_info=initial_state.got_info.model_dump(mode="json"),
        stage="event",
        current_constraint=list(state.constraint),
        event_message=event.message,
        current_got_info=state.got_info.model_dump(mode="json"),
        should_require=list(event.should_require),
        question_answers=[],
        previous_do=None,
        current_hidden_info=state.hidden_info.model_dump(mode="json"),
    )


def _discover_execution_paths(executions_dir: Path, limit: int | None) -> list[Path]:
    """採点対象の実行ログ JSON 一覧を返す。"""
    paths = sorted(executions_dir.glob("*.evaluation.json"))
    if limit is not None:
        return paths[:limit]
    return paths


def _load_case_map(cases_dir: Path, domain: Domain) -> dict[str, Case]:
    """cases_dir 内の全 JSON を読み込み task_id → Case の辞書を返す。"""
    case_map: dict[str, Case] = {}
    for path in sorted(cases_dir.glob("*.json")):
        try:
            text = path.read_text(encoding="utf-8")
            if domain == "meeting":
                case = MeetingCase.model_validate_json(text)
            elif domain == "finance":
                case = FinanceCase.model_validate_json(text)
            else:
                raise ValueError(f"unsupported domain: {domain}")
            case_map[case.task_id] = case
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] failed to load case {path.name}: {exc}")
    return case_map


def _get_domain_adapter(domain: Domain, *, judge_model: str) -> DomainAdapter:
    """domain に応じた evaluate adapter を返す。"""
    if domain == "meeting":
        from .meeting import MeetingAdapter

        return MeetingAdapter(model=judge_model)
    if domain == "finance":
        from .finance import FinanceAdapter

        return FinanceAdapter(model=judge_model)
    raise ValueError(f"unsupported domain: {domain}")


def _format_result(
    *,
    task_id: str,
    result: EvalResult,
    records: list[TurnRecord],
    decision_model: str,
    difficulty: str,
) -> dict[str, Any]:
    """1ケース分の採点結果を辞書に整形する。"""
    return {
        "task_id": task_id,
        "scores": {
            "final_pass_rate": result.final_pass_rate,
            "exact_match": result.exact_match,
            "vague_ask_rate": result.vague_ask_rate,
            "rubric_score": result.rubric_score,
            "redundant_question_rate": result.redundant_question_rate,
            "early_discovery_rate": result.early_discovery_rate,
        },
        "meta": {
            "total_events": result.total_events,
            "early_discovery_n": result.early_discovery_n,
            "decision_model": decision_model,
            "difficulty": difficulty,
        },
        "turns": [
            {
                "event_index": r.event_index,
                "should_require": r.should_require,
                "questions_asked": r.questions_asked,
                "covered_requires": r.covered_requires,
                "redundant_questions": r.redundant_questions,
                "did_without_full_coverage": r.did_without_full_coverage,
                "do_is_feasible": r.do_is_feasible,
                "do_exact_match": r.do_exact_match,
            }
            for r in records
        ],
    }


def _aggregate_summary(
    *,
    domain: str,
    results: list[dict[str, Any]],
    early_discovery_n: int,
) -> dict[str, Any]:
    """全ケースのスコアを平均して summary を作る。"""
    if not results:
        return {"domain": domain, "case_count": 0}

    metrics = [
        "final_pass_rate",
        "exact_match",
        "vague_ask_rate",
        "rubric_score",
        "redundant_question_rate",
        "early_discovery_rate",
    ]

    decision_models = list({r["meta"]["decision_model"] for r in results})
    decision_model = decision_models[0] if len(decision_models) == 1 else decision_models

    avg: dict[str, float] = {}
    for metric in metrics:
        values = [r["scores"][metric] for r in results]
        avg[metric] = round(sum(values) / len(values), 4)

    difficulties = sorted({r["meta"]["difficulty"] for r in results})
    by_difficulty: dict[str, Any] = {}
    for difficulty in difficulties:
        subset = [r for r in results if r["meta"]["difficulty"] == difficulty]
        diff_avg: dict[str, float] = {}
        for metric in metrics:
            values = [r["scores"][metric] for r in subset]
            diff_avg[metric] = round(sum(values) / len(values), 4)
        by_difficulty[difficulty] = {"case_count": len(subset), "average_scores": diff_avg}

    return {
        "domain": domain,
        "case_count": len(results),
        "early_discovery_n": early_discovery_n,
        "decision_model": decision_model,
        "average_scores": avg,
        "by_difficulty": by_difficulty,
    }


if __name__ == "__main__":
    main()
