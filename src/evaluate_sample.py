"""CLI entrypoint for single-case and batch evaluation."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from src.logging_config import configure_logging
from src.models import Case, ExecutionLog
from src.rule_evaluator import RuleBasedEvaluator


@dataclass(frozen=True)
class CaseEvaluationRow:
    task_id: str
    domain: str
    difficulty: str
    scenario_path: str
    execution_log_path: str
    total_score: int
    outcome_score: int
    process_score: int
    recovery_score: int
    failure_labels: list[str]
    deduction_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a sample case or run batch evaluation across scenarios."
    )
    parser.add_argument(
        "--case",
        type=Path,
        default=Path("scenarios/meeting/meeting_001.json"),
        help="Scenario JSON path used by single-case evaluation mode.",
    )
    parser.add_argument(
        "--execution-log",
        type=Path,
        default=Path("results/sample_execution_meeting_001.json"),
        help="Execution log JSON path used by single-case evaluation mode.",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Evaluate every matched scenario/execution-log pair under the dataset roots.",
    )
    parser.add_argument(
        "--scenarios-dir",
        type=Path,
        default=Path("scenarios"),
        help="Root directory that contains scenario JSON files.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Root directory that contains execution log JSON files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results") / "benchmark_summary.json",
        help="Summary JSON output path for batch mode.",
    )
    parser.add_argument(
        "--domain",
        action="append",
        default=[],
        help="Filter batch mode by domain directory name. Can be passed multiple times.",
    )
    parser.add_argument(
        "--task-id",
        action="append",
        default=[],
        help="Filter by task_id in either single or batch mode. Can be passed multiple times.",
    )
    parser.add_argument(
        "--include-templates",
        action="store_true",
        help="Include *_template.json scenarios in batch mode.",
    )
    parser.add_argument(
        "--write-missing-logs",
        action="store_true",
        help="Generate stub execution logs for missing scenarios during batch mode.",
    )
    parser.add_argument(
        "--missing-logs-dir",
        type=Path,
        default=Path("results") / "generated",
        help="Output directory for generated stub execution logs.",
    )
    parser.add_argument(
        "--strict-missing",
        action="store_true",
        help="Exit with code 1 when batch mode finds selected cases without execution logs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()

    if args.batch:
        run_batch(args)
        return

    evaluator = RuleBasedEvaluator()
    case_path = resolve_single_case_path(args)
    log_path = resolve_single_log_path(args)
    case = Case.from_path(case_path)
    log = ExecutionLog.from_path(log_path)
    result = evaluator.evaluate(case, log)
    payload = {
        "report_type": "single_case_evaluation_report",
        "task_id": case.task_id,
        "scenario_path": str(case_path.as_posix()),
        "execution_log_path": str(log_path.as_posix()),
        "outcome_score": result.outcome_score,
        "process_score": result.process_score,
        "recovery_score": result.recovery_score,
        "total_score": result.total_score,
        "failure_labels": result.failure_labels,
        "deductions": result.deductions,
        "human_report": build_human_report(case=case, execution_log=log, payload=result),
        "readable_summary": build_readable_summary(case=case, execution_log=log, payload=result),
        "conversation_timeline": build_conversation_timeline(case=case, execution_log=log),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def run_batch(args: argparse.Namespace) -> None:
    evaluator = RuleBasedEvaluator()
    cases = discover_cases(
        scenarios_dir=args.scenarios_dir,
        include_templates=args.include_templates,
        domain_filters=set(args.domain),
        task_id_filters=set(args.task_id),
    )
    log_index = discover_execution_logs(args.results_dir)

    if args.write_missing_logs:
        write_missing_log_templates(
            cases=cases,
            log_index=log_index,
            output_dir=args.missing_logs_dir,
        )

    rows: list[CaseEvaluationRow] = []
    missing_task_ids: list[str] = []

    for case_path, case in cases:
        matched_log_path = log_index.get(case.task_id)
        if matched_log_path is None:
            missing_task_ids.append(case.task_id)
            continue

        execution_log = ExecutionLog.from_path(matched_log_path)
        result = evaluator.evaluate(case=case, execution_log=execution_log)
        rows.append(
            CaseEvaluationRow(
                task_id=case.task_id,
                domain=case.domain,
                difficulty=case.difficulty,
                scenario_path=case_path.as_posix(),
                execution_log_path=matched_log_path.as_posix(),
                total_score=result.total_score,
                outcome_score=result.outcome_score,
                process_score=result.process_score,
                recovery_score=result.recovery_score,
                failure_labels=result.failure_labels,
                deduction_count=len(result.deductions),
            )
        )

    summary = build_summary(
        rows=rows,
        selected_cases=len(cases),
        missing_task_ids=missing_task_ids,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(render_cli_summary(summary, args.output))

    if args.strict_missing and missing_task_ids:
        raise SystemExit(1)


def resolve_single_case_path(args: argparse.Namespace) -> Path:
    if not args.task_id:
        return args.case
    task_id = args.task_id[0]
    matched_cases = discover_cases(
        scenarios_dir=args.scenarios_dir,
        include_templates=args.include_templates,
        domain_filters=set(args.domain),
        task_id_filters={task_id},
    )
    if not matched_cases:
        raise FileNotFoundError(f"task_id not found under scenarios: {task_id}")
    return matched_cases[0][0]


def resolve_single_log_path(args: argparse.Namespace) -> Path:
    if not args.task_id:
        return args.execution_log
    task_id = args.task_id[0]
    log_index = discover_execution_logs(args.results_dir)
    matched_log_path = log_index.get(task_id)
    if matched_log_path is None:
        raise FileNotFoundError(f"execution log not found under results for task_id: {task_id}")
    return matched_log_path


def discover_cases(
    scenarios_dir: Path,
    include_templates: bool,
    domain_filters: set[str],
    task_id_filters: set[str],
) -> list[tuple[Path, Case]]:
    discovered: list[tuple[Path, Case]] = []
    for case_path in sorted(scenarios_dir.rglob("*.json")):
        if not include_templates and case_path.stem.endswith("_template"):
            continue

        case = Case.from_path(case_path)
        domain_name = case_path.parent.name
        if domain_filters and domain_name not in domain_filters:
            continue
        if task_id_filters and case.task_id not in task_id_filters:
            continue
        discovered.append((case_path, case))
    return discovered


def discover_execution_logs(results_dir: Path) -> dict[str, Path]:
    log_index: dict[str, Path] = {}
    for log_path in sorted(results_dir.rglob("*.json")):
        try:
            execution_log = ExecutionLog.from_path(log_path)
        except Exception:
            continue

        if execution_log.task_id not in log_index:
            log_index[execution_log.task_id] = log_path
    return log_index


def write_missing_log_templates(
    cases: list[tuple[Path, Case]],
    log_index: dict[str, Path],
    output_dir: Path,
) -> None:
    for scenario_path, case in cases:
        if case.task_id in log_index:
            continue

        output_path = output_dir / scenario_path.parent.name / f"{case.task_id}.execution_log.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            continue

        output_path.write_text(
            json.dumps(build_stub_execution_log(case), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def build_stub_execution_log(case: Case) -> dict[str, object]:
    return {
        "task_id": case.task_id,
        "actions": [],
        "completed_tasks": [],
        "task_breakdown": [],
        "questions_asked": [],
        "constraint_violations": [],
        "unsafe_commit": False,
        "final_state": {
            "deadline": case.initial_state.deadline,
            "participants": case.initial_state.participants,
            "budget": case.initial_state.budget,
            "constraints": case.initial_state.constraints,
            "reference_data": case.initial_state.reference_data,
        },
        "final_artifacts": {
            artifact.artifact_id: {"fields_completed": [], "field_values": {}}
            for artifact in case.initial_state.required_artifacts
        },
    }


def build_summary(
    rows: list[CaseEvaluationRow],
    selected_cases: int,
    missing_task_ids: list[str],
) -> dict[str, object]:
    rows_by_score = sorted(rows, key=lambda row: (row.total_score, row.task_id))
    return {
        "selected_cases": selected_cases,
        "evaluated_cases": len(rows),
        "missing_cases": len(missing_task_ids),
        "missing_task_ids": sorted(missing_task_ids),
        "average_total_score": round(mean([row.total_score for row in rows]), 2) if rows else None,
        "average_outcome_score": round(mean([row.outcome_score for row in rows]), 2)
        if rows
        else None,
        "average_process_score": round(mean([row.process_score for row in rows]), 2)
        if rows
        else None,
        "average_recovery_score": round(mean([row.recovery_score for row in rows]), 2)
        if rows
        else None,
        "lowest_scoring_cases": [asdict(row) for row in rows_by_score[:5]],
        "highest_scoring_cases": [asdict(row) for row in rows_by_score[-5:]],
        "cases": [asdict(row) for row in sorted(rows, key=lambda row: row.task_id)],
    }


def render_cli_summary(summary: dict[str, object], output_path: Path) -> str:
    lines = [
        f"selected_cases: {summary['selected_cases']}",
        f"evaluated_cases: {summary['evaluated_cases']}",
        f"missing_cases: {summary['missing_cases']}",
        f"average_total_score: {summary['average_total_score']}",
        f"summary_json: {output_path.as_posix()}",
    ]
    missing_task_ids = summary["missing_task_ids"]
    if isinstance(missing_task_ids, list) and missing_task_ids:
        lines.append(f"missing_task_ids: {', '.join(missing_task_ids[:10])}")
    return "\n".join(lines)


def build_readable_summary(
    case: Case,
    execution_log: ExecutionLog,
    payload: Any,
) -> dict[str, object]:
    event_turns = {event.turn for event in case.events}
    acknowledged_turns = {
        turn
        for action in execution_log.actions
        for turn in action.acknowledged_event_turns
    }
    updated_artifacts = sorted(
        {
            artifact_id
            for action in execution_log.actions
            for artifact_id in action.artifact_updates
        }
    )
    return {
        "task_overview": {
            "task_id": case.task_id,
            "domain": case.domain,
            "difficulty": case.difficulty,
            "required_artifacts": [
                artifact.artifact_id for artifact in case.initial_state.required_artifacts
            ],
            "event_turns": sorted(event_turns),
        },
        "scorecard": {
            "total_score": payload.total_score,
            "outcome_score": payload.outcome_score,
            "process_score": payload.process_score,
            "recovery_score": payload.recovery_score,
            "failure_labels": payload.failure_labels,
            "deduction_count": len(payload.deductions),
        },
        "execution_overview": {
            "action_count": len(execution_log.actions),
            "question_count": len(execution_log.questions_asked),
            "completed_tasks": execution_log.completed_tasks,
            "acknowledged_event_turns": sorted(acknowledged_turns),
            "artifact_updates_seen": updated_artifacts,
            "finalized": any(action.action_type == "finalize" for action in execution_log.actions),
        },
        "event_status": build_event_status(case=case, execution_log=execution_log),
        "top_deductions": payload.deductions[:10],
    }


def build_human_report(
    case: Case,
    execution_log: ExecutionLog,
    payload: Any,
) -> dict[str, object]:
    event_status = build_event_status(case=case, execution_log=execution_log)
    return {
        "viewer_order": [
            "headline",
            "verdict",
            "score_explainer",
            "benchmark_case",
            "what_happened",
            "why_it_lost_points",
            "conversation_timeline",
        ],
        "headline": build_headline(case=case, execution_log=execution_log, payload=payload),
        "verdict": build_verdict(case=case, execution_log=execution_log, payload=payload),
        "score_explainer": build_score_explainer(payload=payload),
        "benchmark_case": build_benchmark_case_section(case=case),
        "what_happened": build_what_happened_section(
            case=case,
            execution_log=execution_log,
            event_status=event_status,
        ),
        "why_it_lost_points": build_loss_analysis(
            case=case,
            execution_log=execution_log,
            payload=payload,
            event_status=event_status,
        ),
    }


def build_event_status(case: Case, execution_log: ExecutionLog) -> list[dict[str, object]]:
    questions_by_turn = {question.turn: question for question in execution_log.questions_asked}
    actions = execution_log.actions
    statuses: list[dict[str, object]] = []
    for event in case.events:
        response_window_end = event.turn + event.expected_replan_within_turns
        window_actions = [
            action
            for action in actions
            if event.turn <= action.turn <= response_window_end
        ]
        statuses.append(
            {
                "event_turn": event.turn,
                "event_type": event.type,
                "expected_replan_within_turns": event.expected_replan_within_turns,
                "expected_tasks": event.expected_tasks,
                "expected_artifact_updates": event.expected_artifact_updates,
                "acknowledged": any(
                    event.turn in action.acknowledged_event_turns for action in actions
                ),
                "responded_in_window": any(
                    action.action_type in {"update_plan", "confirm_state"}
                    for action in window_actions
                ),
                "artifact_updates_in_window": sorted(
                    {
                        artifact_id
                        for action in window_actions
                        for artifact_id in action.artifact_updates
                    }
                ),
                "clarification_in_window": any(
                    event.turn <= turn <= response_window_end for turn in questions_by_turn
                ),
            }
        )
    return statuses


def build_headline(case: Case, execution_log: ExecutionLog, payload: Any) -> dict[str, object]:
    acknowledged_turns = [
        turn for action in execution_log.actions for turn in action.acknowledged_event_turns
    ]
    acknowledged_event_count = sum(1 for event in case.events if event.turn in acknowledged_turns)
    return {
        "task_id": case.task_id,
        "domain": case.domain,
        "difficulty": case.difficulty,
        "total_score": payload.total_score,
        "one_line_summary": (
            f"{case.task_id} は total_score={payload.total_score}。"
            f" {len(case.events)} 件のイベント中、"
            f"{acknowledged_event_count}"
            " 件しか明示 acknowledgement がなく、"
            f"{len(execution_log.questions_asked)} 件しか clarification がありません。"
        ),
    }


def build_verdict(case: Case, execution_log: ExecutionLog, payload: Any) -> dict[str, object]:
    finalized_early = (
        any(action.action_type == "finalize" for action in execution_log.actions)
        and execution_log.actions[-1].turn < max(event.turn for event in case.events)
    )
    finalize_reason = (
        "finalize がイベント完了前に走っている"
        if finalized_early
        else "finalize timing は致命的ではない"
    )
    return {
        "result": "fail" if payload.total_score < 60 else "pass",
        "reason": [
            "latest state を final_state が取り切れていない",
            "event 後の replan/acknowledgement が不足している",
            "ambiguity に対する ask_clarification が無い",
            finalize_reason,
        ],
        "failure_labels": payload.failure_labels,
    }


def build_score_explainer(payload: Any) -> dict[str, object]:
    return {
        "total_score": payload.total_score,
        "breakdown": {
            "outcome_score": payload.outcome_score,
            "process_score": payload.process_score,
            "recovery_score": payload.recovery_score,
        },
        "interpretation": {
            "outcome": "最終成果物と final_state が最新条件を満たしたか",
            "process": "質問、分解、依存順、イベント後の行動が妥当か",
            "recovery": "イベント発生後にどれだけ追従・再計画できたか",
        },
    }


def build_benchmark_case_section(case: Case) -> dict[str, object]:
    return {
        "initial_request": case.initial_request,
        "world_state": {
            "deadline": case.initial_state.deadline,
            "participants": case.initial_state.participants,
            "constraints": case.initial_state.constraints,
            "required_artifacts": [
                {
                    "artifact_id": artifact.artifact_id,
                    "required_fields": artifact.required_fields,
                }
                for artifact in case.initial_state.required_artifacts
            ],
        },
        "allowed_actions": case.allowed_actions,
        "environment_events": [
            {
                "turn": event.turn,
                "type": event.type,
                "message": event.message,
                "expected_tasks": event.expected_tasks,
                "expected_artifact_updates": event.expected_artifact_updates,
            }
            for event in case.events
        ],
        "goal_condition": case.goal_condition.model_dump(mode="json"),
        "rubric": case.rubric.model_dump(mode="json"),
    }


def build_what_happened_section(
    case: Case,
    execution_log: ExecutionLog,
    event_status: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "execution_log_summary": {
            "actions": len(execution_log.actions),
            "questions": len(execution_log.questions_asked),
            "completed_tasks": execution_log.completed_tasks,
            "finalized": any(action.action_type == "finalize" for action in execution_log.actions),
        },
        "event_follow_up": event_status,
        "final_state_snapshot": execution_log.final_state,
        "final_artifact_snapshot": {
            artifact_id: artifact.model_dump(mode="json")
            for artifact_id, artifact in execution_log.final_artifacts.items()
        },
    }


def build_loss_analysis(
    case: Case,
    execution_log: ExecutionLog,
    payload: Any,
    event_status: list[dict[str, object]],
) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    if payload.total_score < 60:
        findings.append(
            {
                "title": "final_state が latest state を保持できていない",
                "evidence": payload.deductions[:8],
            }
        )
    missing_event_turns = [
        status["event_turn"] for status in event_status if not status["acknowledged"]
    ]
    if missing_event_turns:
        findings.append(
            {
                "title": "イベント acknowledgement が欠落",
                "evidence": {
                    "missing_event_turns": missing_event_turns,
                    "logged_acknowledged_event_turns": sorted(
                        {
                            turn
                            for action in execution_log.actions
                            for turn in action.acknowledged_event_turns
                        }
                    ),
                },
            }
        )
    failed_recovery_turns = [
        status["event_turn"] for status in event_status if not status["responded_in_window"]
    ]
    if failed_recovery_turns:
        findings.append(
            {
                "title": "イベント後の timely replan が欠落",
                "evidence": {"failed_event_turns": failed_recovery_turns},
            }
        )
    ambiguity_turns = [event.turn for event in case.events if event.type == "ambiguity"]
    if not execution_log.questions_asked and ambiguity_turns:
        findings.append(
            {
                "title": "ambiguity に ask_clarification が無い",
                "evidence": {
                    "ambiguity_turns": ambiguity_turns,
                    "questions_asked": [],
                },
            }
        )
    if any(action.action_type == "finalize" for action in execution_log.actions):
        finalize_turn = max(
            action.turn for action in execution_log.actions if action.action_type == "finalize"
        )
        if finalize_turn < max(event.turn for event in case.events):
            findings.append(
                {
                    "title": "finalize が最後のイベントより前に走っている",
                    "evidence": {
                        "finalize_turn": finalize_turn,
                        "last_event_turn": max(event.turn for event in case.events),
                    },
                }
            )
    return findings


def build_conversation_timeline(case: Case, execution_log: ExecutionLog) -> list[dict[str, object]]:
    timeline: list[dict[str, object]] = []
    action_map = {}
    for action in execution_log.actions:
        action_map.setdefault(action.turn, []).append(action)
    question_map = {}
    for question in execution_log.questions_asked:
        question_map.setdefault(question.turn, []).append(question)

    all_turns = sorted(
        {
            *(event.turn for event in case.events),
            *(action.turn for action in execution_log.actions),
            *(question.turn for question in execution_log.questions_asked),
        }
    )
    for turn in all_turns:
        turn_events = [event for event in case.events if event.turn == turn]
        turn_actions = action_map.get(turn, [])
        turn_questions = question_map.get(turn, [])
        timeline.append(
            {
                "turn": turn,
                "events": [
                    {
                        "type": event.type,
                        "message": event.message,
                        "expected_tasks": event.expected_tasks,
                        "expected_artifact_updates": event.expected_artifact_updates,
                    }
                    for event in turn_events
                ],
                "actions": [
                    {
                        "action_type": action.action_type,
                        "acknowledged_event_turns": action.acknowledged_event_turns,
                        "artifact_updates": action.artifact_updates,
                        "notes": action.notes,
                    }
                    for action in turn_actions
                ],
                "questions": [
                    {
                        "question": question.question,
                        "reason": question.reason,
                        "blocks_tasks": question.blocks_tasks,
                    }
                    for question in turn_questions
                ],
            }
        )
    return timeline


if __name__ == "__main__":
    main()
