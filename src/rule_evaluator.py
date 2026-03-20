# ruff: noqa: I001

"""Rule-based evaluator for Dynamic Agent Correctness Benchmark."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


FAILURE_LABELS = {
    "state_staleness",
    "missing_replan",
    "partial_replan",
    "invalid_dependency",
    "goal_drift",
    "unsafe_commit",
    "constraint_violation",
    "artifact_inconsistency",
}


@dataclass
class EvaluationResult:
    outcome_score: int
    process_score: int
    recovery_score: int
    total_score: int
    failure_labels: list[str] = field(default_factory=list)
    deductions: list[str] = field(default_factory=list)


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _artifact_map(final_artifacts: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        artifact_id: artifact_value
        for artifact_id, artifact_value in final_artifacts.items()
        if isinstance(artifact_value, dict)
    }


def evaluate_case(case: dict[str, Any], execution_log: dict[str, Any]) -> EvaluationResult:
    outcome_score = 100
    process_score = 100
    recovery_score = 100
    deductions: list[str] = []
    failure_labels: set[str] = set()

    initial_state = case.get("initial_state", {})
    required_artifacts = initial_state.get("required_artifacts", [])
    dependencies = _safe_list(initial_state.get("task_dependencies"))
    events = _safe_list(case.get("events"))
    goal_condition = case.get("goal_condition", {})

    actions = _safe_list(execution_log.get("actions"))
    final_state = execution_log.get("final_state", {})
    final_artifacts = _artifact_map(execution_log.get("final_artifacts", {}))
    completed_tasks = set(_safe_list(execution_log.get("completed_tasks")))

    required_artifact_ids = {artifact["artifact_id"] for artifact in required_artifacts}

    for artifact in required_artifacts:
        artifact_id = artifact["artifact_id"]
        if artifact_id not in final_artifacts:
            outcome_score -= 25
            deductions.append(f"missing required artifact: {artifact_id} (-25)")
            failure_labels.add("goal_drift")
            continue

        actual_artifact = final_artifacts[artifact_id]
        completed_fields = set(_safe_list(actual_artifact.get("fields_completed")))
        for field_name in _safe_list(artifact.get("required_fields")):
            if field_name not in completed_fields:
                outcome_score -= 5
                deductions.append(f"artifact {artifact_id} missing field: {field_name} (-5)")
                failure_labels.add("artifact_inconsistency")

    if goal_condition.get("must_satisfy_latest_state"):
        expected_deadline = initial_state.get("deadline")
        expected_participants = set(_safe_list(initial_state.get("participants")))
        expected_budget = initial_state.get("budget")
        expected_reference_data = dict(initial_state.get("reference_data", {}))

        for event in events:
            delta = event.get("delta", {})
            if "deadline" in delta:
                expected_deadline = delta["deadline"]
            if "participants_added" in delta:
                expected_participants.update(_safe_list(delta["participants_added"]))
            if "budget" in delta:
                expected_budget = delta["budget"]
            if "reference_data" in delta and isinstance(delta["reference_data"], dict):
                expected_reference_data.update(delta["reference_data"])

        if final_state.get("deadline") != expected_deadline:
            outcome_score -= 20
            deductions.append("final deadline does not match latest state (-20)")
            failure_labels.add("state_staleness")

        actual_participants = set(_safe_list(final_state.get("participants")))
        if expected_participants and actual_participants != expected_participants:
            outcome_score -= 20
            deductions.append("final participants do not match latest state (-20)")
            failure_labels.add("state_staleness")

        if expected_budget is not None and final_state.get("budget") != expected_budget:
            outcome_score -= 20
            deductions.append("final budget does not match latest state (-20)")
            failure_labels.add("constraint_violation")

        if expected_reference_data:
            actual_reference_data = final_state.get("reference_data", {})
            for key, value in expected_reference_data.items():
                if actual_reference_data.get(key) != value:
                    outcome_score -= 10
                    deductions.append(f"reference_data mismatch for {key} (-10)")
                    failure_labels.add("state_staleness")

    if goal_condition.get("no_constraint_violation"):
        if _safe_list(execution_log.get("constraint_violations")):
            for violation in _safe_list(execution_log.get("constraint_violations")):
                outcome_score -= 20
                deductions.append(f"constraint violation: {violation} (-20)")
                failure_labels.add("constraint_violation")

    for dependency in dependencies:
        after_task = dependency.get("after")
        before_task = dependency.get("before")
        if after_task in completed_tasks and before_task not in completed_tasks:
            process_score -= 20
            deductions.append(
                f"dependency violated: {before_task} must happen before {after_task} (-20)"
            )
            failure_labels.add("invalid_dependency")

    action_turn_map: dict[int, list[dict[str, Any]]] = {}
    for action in actions:
        turn = action.get("turn")
        if isinstance(turn, int):
            action_turn_map.setdefault(turn, []).append(action)

    for event in events:
        turn = event.get("turn")
        expected_replan = event.get("expected_replan_within_turns", 0)
        responded = False
        updated_artifacts = False
        acknowledged = False
        for action in actions:
            action_turn = action.get("turn")
            if not isinstance(action_turn, int) or not isinstance(turn, int):
                continue
            if action_turn < turn:
                continue
            if action_turn > turn + expected_replan:
                continue
            if action.get("action_type") in {"update_plan", "confirm_state"}:
                responded = True
            if turn in _safe_list(action.get("acknowledged_event_turns")):
                acknowledged = True
            if _safe_list(action.get("artifact_updates")):
                updated_artifacts = True

        if not responded:
            recovery_score -= 15
            process_score -= 15
            deductions.append(
                f"no timely replan after event turn {turn} (-15 recovery, -15 process)"
            )
            failure_labels.add("missing_replan")
        if not acknowledged and goal_condition.get("must_acknowledge_changes", True):
            recovery_score -= 10
            deductions.append(f"event turn {turn} was not explicitly acknowledged (-10)")
            failure_labels.add("state_staleness")
        if not updated_artifacts:
            recovery_score -= 10
            deductions.append(f"no artifact update observed after event turn {turn} (-10)")
            failure_labels.add("partial_replan")

    if execution_log.get("unsafe_commit"):
        process_score -= 20
        deductions.append("unsafe commit detected (-20)")
        failure_labels.add("unsafe_commit")

    delivered_artifacts = set(final_artifacts.keys())
    if required_artifact_ids and not required_artifact_ids.issubset(delivered_artifacts):
        failure_labels.add("goal_drift")

    outcome_score = max(0, outcome_score)
    process_score = max(0, process_score)
    recovery_score = max(0, recovery_score)
    total_score = round(outcome_score * 0.40 + process_score * 0.35 + recovery_score * 0.25)

    return EvaluationResult(
        outcome_score=outcome_score,
        process_score=process_score,
        recovery_score=recovery_score,
        total_score=total_score,
        failure_labels=sorted(label for label in failure_labels if label in FAILURE_LABELS),
        deductions=deductions,
    )
