# ruff: noqa: I001

"""Dynamic Agent Correctness Benchmark 向けのルールベース evaluator。"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from src.models import (
    ActionLog,
    BenchmarkEvent,
    Case,
    ExecutionLog,
    TaskBreakdownItem,
)

logger = logging.getLogger(__name__)


FAILURE_LABELS = {
    "state_staleness",
    "missing_replan",
    "partial_replan",
    "invalid_dependency",
    "goal_drift",
    "unsafe_commit",
    "constraint_violation",
    "artifact_inconsistency",
    "poor_task_decomposition",
    "question_handling",
}


@dataclass
class EvaluationResult:
    """ルールベース evaluator が返す採点結果。"""

    outcome_score: int
    process_score: int
    recovery_score: int
    total_score: int
    failure_labels: list[str] = field(default_factory=list)
    deductions: list[str] = field(default_factory=list)


class RuleBasedEvaluator:
    """1 件の benchmark case と execution log を評価する。"""

    def __init__(self) -> None:
        self._tracked_state_keys = {"deadline", "participants", "budget", "reference_data"}

    def evaluate(
        self,
        case: Case,
        execution_log: ExecutionLog,
    ) -> EvaluationResult:
        """検証済みの Case と ExecutionLog の組を採点する。"""
        logger.info(
            "評価を開始します: task_id=%s actions=%s events=%s",
            case.task_id,
            len(execution_log.actions),
            len(case.events),
        )

        outcome_score = 100
        process_score = 100
        recovery_score = 100
        deductions: list[str] = []
        failure_labels: set[str] = set()

        initial_state = case.initial_state
        required_artifacts = initial_state.required_artifacts
        dependencies = initial_state.task_dependencies
        events = case.events
        goal_condition = case.goal_condition
        actions = execution_log.actions
        final_state = execution_log.final_state
        final_artifacts = execution_log.final_artifacts
        completed_tasks = execution_log.completed_tasks
        task_order = self._task_order_map(completed_tasks)
        allowed_actions = set(case.allowed_actions)
        goal_required_artifact_ids = set(goal_condition.required_artifacts)
        required_artifacts = self._select_required_artifacts(
            required_artifacts=required_artifacts,
            goal_required_artifact_ids=goal_required_artifact_ids,
            deductions=deductions,
            failure_labels=failure_labels,
        )
        required_artifact_ids = {artifact.artifact_id for artifact in required_artifacts}

        process_score -= self._score_action_constraints(
            actions=actions,
            allowed_actions=allowed_actions,
            deductions=deductions,
            failure_labels=failure_labels,
        )
        outcome_score -= self._score_required_artifacts(
            required_artifacts=required_artifacts,
            final_artifacts=final_artifacts,
            final_state=final_state,
            deductions=deductions,
            failure_labels=failure_labels,
        )

        if goal_condition.must_satisfy_latest_state:
            expected_state = self._build_tracked_expected_state(
                initial_state.model_dump(mode="python"),
                [event.model_dump(mode="python") for event in events],
            )
            outcome_score -= self._compare_expected_state(
                expected_state=expected_state,
                actual_state=final_state,
                deductions=deductions,
                failure_labels=failure_labels,
            )

        if goal_condition.no_constraint_violation:
            for violation in execution_log.constraint_violations:
                outcome_score -= 20
                deductions.append(f"constraint violation: {violation} (-20)")
                failure_labels.add("constraint_violation")

        process_score -= self._score_task_dependencies(
            dependencies=dependencies,
            task_order=task_order,
            deductions=deductions,
            failure_labels=failure_labels,
        )
        process_score -= self._score_task_breakdown(
            task_breakdown=execution_log.task_breakdown,
            completed_tasks=completed_tasks,
            deductions=deductions,
            failure_labels=failure_labels,
        )
        process_score -= self._score_questions(
            questions_asked=execution_log.questions_asked,
            task_order=task_order,
            deductions=deductions,
            failure_labels=failure_labels,
        )
        process_score, recovery_score = self._score_events(
            events=events,
            actions=actions,
            required_artifact_ids=required_artifact_ids,
            must_acknowledge_changes=goal_condition.must_acknowledge_changes,
            process_score=process_score,
            recovery_score=recovery_score,
            deductions=deductions,
            failure_labels=failure_labels,
        )

        if execution_log.unsafe_commit:
            process_score -= 20
            deductions.append("unsafe commit detected (-20)")
            failure_labels.add("unsafe_commit")

        delivered_artifacts = set(final_artifacts.keys())
        if goal_required_artifact_ids and not goal_required_artifact_ids.issubset(delivered_artifacts):
            failure_labels.add("goal_drift")

        outcome_score = max(0, outcome_score)
        process_score = max(0, process_score)
        recovery_score = max(0, recovery_score)
        total_score = round(outcome_score * 0.40 + process_score * 0.35 + recovery_score * 0.25)

        logger.info(
            "評価が完了しました: task_id=%s total_score=%s failures=%s",
            case.task_id,
            total_score,
            sorted(label for label in failure_labels if label in FAILURE_LABELS),
        )

        return EvaluationResult(
            outcome_score=outcome_score,
            process_score=process_score,
            recovery_score=recovery_score,
            total_score=total_score,
            failure_labels=sorted(label for label in failure_labels if label in FAILURE_LABELS),
            deductions=deductions,
        )

    def _select_required_artifacts(
        self,
        required_artifacts: list[Any],
        goal_required_artifact_ids: set[str],
        deductions: list[str],
        failure_labels: set[str],
    ) -> list[Any]:
        """goal_condition.required_artifacts に基づいて採点対象の成果物を決める。"""

        if not goal_required_artifact_ids:
            return required_artifacts

        artifact_map = {artifact.artifact_id: artifact for artifact in required_artifacts}
        selected_artifacts: list[Any] = []

        for artifact_id in goal_required_artifact_ids:
            artifact = artifact_map.get(artifact_id)
            if artifact is None:
                deductions.append(
                    f"goal_condition references undefined required artifact: {artifact_id} (-10)"
                )
                failure_labels.add("goal_drift")
                continue
            selected_artifacts.append(artifact)

        return selected_artifacts

    def _score_action_constraints(
        self,
        actions: list[ActionLog],
        allowed_actions: set[str],
        deductions: list[str],
        failure_labels: set[str],
    ) -> int:
        """ケース定義で許可されていない action を減点する。"""

        penalty = 0
        for action in actions:
            if allowed_actions and action.action_type not in allowed_actions:
                penalty += 20
                deductions.append(f"disallowed action used: {action.action_type} (-20)")
                failure_labels.add("constraint_violation")
        return penalty

    def _score_required_artifacts(
        self,
        required_artifacts: list[Any],
        final_artifacts: dict[str, Any],
        final_state: dict[str, Any],
        deductions: list[str],
        failure_labels: set[str],
    ) -> int:
        """成果物の存在、必須項目の充足、state との整合性を検証する。"""

        penalty = 0
        for artifact in required_artifacts:
            artifact_id = artifact.artifact_id
            if artifact_id not in final_artifacts:
                penalty += 25
                deductions.append(f"missing required artifact: {artifact_id} (-25)")
                failure_labels.add("goal_drift")
                continue

            actual_artifact = final_artifacts[artifact_id]
            completed_fields = set(actual_artifact.fields_completed)

            for field_name in artifact.required_fields:
                if field_name not in completed_fields:
                    penalty += 5
                    deductions.append(f"artifact {artifact_id} missing field: {field_name} (-5)")
                    failure_labels.add("artifact_inconsistency")

            if not self._has_meaningful_artifact_content(actual_artifact.field_values):
                penalty += 10
                deductions.append(f"artifact {artifact_id} has no meaningful content (-10)")
                failure_labels.add("artifact_inconsistency")

            for binding in artifact.semantic_checks:
                expected_value = self._get_nested_value(final_state, binding.state_path)
                actual_value = actual_artifact.field_values.get(binding.artifact_field)
                if actual_value != expected_value:
                    penalty += 10
                    deductions.append(
                        f"artifact {artifact_id} field {binding.artifact_field} "
                        f"does not reflect {binding.state_path} (-10)"
                    )
                    failure_labels.add("artifact_inconsistency")
        return penalty

    def _score_task_dependencies(
        self,
        dependencies: list[Any],
        task_order: dict[str, int],
        deductions: list[str],
        failure_labels: set[str],
    ) -> int:
        """完了タスクが依存関係どおりの順序を守っているか確認する。"""

        penalty = 0
        for dependency in dependencies:
            after_task = dependency.after
            before_task = dependency.before
            if after_task not in task_order:
                continue
            if before_task not in task_order:
                penalty += 20
                deductions.append(
                    f"dependency violated: {before_task} must happen before {after_task} (-20)"
                )
                failure_labels.add("invalid_dependency")
                continue
            if task_order[before_task] > task_order[after_task]:
                penalty += 20
                deductions.append(
                    f"dependency order violated: {before_task} after {after_task} (-20)"
                )
                failure_labels.add("invalid_dependency")
        return penalty

    def _score_task_breakdown(
        self,
        task_breakdown: list[TaskBreakdownItem],
        completed_tasks: list[str],
        deductions: list[str],
        failure_labels: set[str],
    ) -> int:
        """エージェントが実用的なタスク分解を記録しているか確認する。"""

        if not completed_tasks:
            return 0
        if not task_breakdown:
            deductions.append("task breakdown missing for completed work (-10)")
            failure_labels.add("poor_task_decomposition")
            return 10

        penalty = 0
        breakdown_ids = {item.task_id for item in task_breakdown}
        for task_name in completed_tasks:
            if task_name not in breakdown_ids:
                penalty += 5
                deductions.append(f"completed task not present in task breakdown: {task_name} (-5)")
                failure_labels.add("poor_task_decomposition")

        for item in task_breakdown:
            if not item.description.strip():
                penalty += 5
                deductions.append(f"task breakdown item {item.task_id} has empty description (-5)")
                failure_labels.add("poor_task_decomposition")
        return penalty

    def _score_questions(
        self,
        questions_asked: list[Any],
        task_order: dict[str, int],
        deductions: list[str],
        failure_labels: set[str],
    ) -> int:
        """確認質問が、該当作業より前に適切に記録されているか評価する。"""

        penalty = 0
        for question in questions_asked:
            if not question.question.strip():
                penalty += 5
                deductions.append(f"clarification question at turn {question.turn} is empty (-5)")
                failure_labels.add("question_handling")

            for blocked_task in question.blocks_tasks:
                blocked_task_order = task_order.get(blocked_task)
                if blocked_task_order is None:
                    continue
                if blocked_task_order < question.turn:
                    penalty += 10
                    deductions.append(
                        f"clarification for {blocked_task} was recorded after task execution (-10)"
                    )
                    failure_labels.add("question_handling")
        return penalty

    def _score_events(
        self,
        events: list[BenchmarkEvent],
        actions: list[ActionLog],
        required_artifact_ids: set[str],
        must_acknowledge_changes: bool,
        process_score: int,
        recovery_score: int,
        deductions: list[str],
        failure_labels: set[str],
    ) -> tuple[int, int]:
        """各イベント発生後の再計画品質を採点する。"""

        for event in events:
            responded = False
            updated_artifacts = False
            acknowledged = False
            expected_artifact_updates = set(event.expected_artifact_updates)
            observed_artifact_updates: set[str] = set()

            for action in actions:
                if action.turn < event.turn:
                    continue
                if action.turn > event.turn + event.expected_replan_within_turns:
                    continue

                if action.action_type in {"update_plan", "confirm_state"}:
                    responded = True
                if event.turn in action.acknowledged_event_turns:
                    acknowledged = True
                observed_artifact_updates.update(action.artifact_updates)

            if expected_artifact_updates:
                updated_artifacts = expected_artifact_updates.issubset(observed_artifact_updates)
            elif (
                observed_artifact_updates
                and observed_artifact_updates.issubset(required_artifact_ids)
            ):
                updated_artifacts = True

            if not responded:
                recovery_score -= 15
                process_score -= 15
                deductions.append(
                    f"no timely replan after event turn {event.turn} (-15 recovery, -15 process)"
                )
                failure_labels.add("missing_replan")
            if not acknowledged and must_acknowledge_changes:
                recovery_score -= 10
                deductions.append(f"event turn {event.turn} was not explicitly acknowledged (-10)")
                failure_labels.add("state_staleness")
            if not updated_artifacts:
                recovery_score -= 10
                deductions.append(
                    f"no artifact update observed after event turn {event.turn} (-10)"
                )
                failure_labels.add("partial_replan")

        return process_score, recovery_score

    def _task_order_map(self, completed_tasks: list[str]) -> dict[str, int]:
        """各完了タスクについて最初に観測された実行順を返す。"""

        order_map: dict[str, int] = {}
        for index, task_name in enumerate(completed_tasks):
            if task_name not in order_map:
                order_map[task_name] = index
        return order_map

    def _build_tracked_expected_state(
        self,
        initial_state: dict[str, Any],
        events: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """evaluator が追跡する state 項目に対してイベント差分を反映する。"""

        tracked_state: dict[str, Any] = {}
        for key, value in initial_state.items():
            if key in self._tracked_state_keys:
                if isinstance(value, dict):
                    tracked_state[key] = dict(value)
                elif isinstance(value, list):
                    tracked_state[key] = list(value)
                else:
                    tracked_state[key] = value

        for event in events:
            delta = event.get("delta", {})
            if isinstance(delta, dict):
                self._merge_state_delta(tracked_state, delta)
        return tracked_state

    def _merge_state_delta(self, expected_state: dict[str, Any], delta: dict[str, Any]) -> None:
        """1 件のイベント差分を追跡対象 state にマージする。"""

        for key, value in delta.items():
            if key == "participants_added":
                participants = self._safe_list(expected_state.get("participants"))
                expected_state["participants"] = participants + [
                    participant
                    for participant in self._safe_list(value)
                    if participant not in participants
                ]
                continue

            if isinstance(value, dict):
                current_value = expected_state.get(key, {})
                if not isinstance(current_value, dict):
                    current_value = {}
                merged_value = dict(current_value)
                self._merge_state_delta(merged_value, value)
                expected_state[key] = merged_value
                continue

            expected_state[key] = value

    def _compare_expected_state(
        self,
        expected_state: dict[str, Any],
        actual_state: dict[str, Any],
        deductions: list[str],
        failure_labels: set[str],
        path_prefix: str = "",
    ) -> int:
        """最終 state が期待される最新 state と一致するか比較する。"""

        penalty = 0
        for key, expected_value in expected_state.items():
            if key not in actual_state:
                penalty += 10
                deductions.append(f"final state missing key: {path_prefix}{key} (-10)")
                failure_labels.add("state_staleness")
                continue

            actual_value = actual_state[key]
            if isinstance(expected_value, dict):
                if not isinstance(actual_value, dict):
                    penalty += 10
                    deductions.append(f"final state type mismatch for {path_prefix}{key} (-10)")
                    failure_labels.add("state_staleness")
                    continue
                penalty += self._compare_expected_state(
                    expected_state=expected_value,
                    actual_state=actual_value,
                    deductions=deductions,
                    failure_labels=failure_labels,
                    path_prefix=f"{path_prefix}{key}.",
                )
                continue

            if isinstance(expected_value, list):
                if f"{path_prefix}{key}" == "participants":
                    if set(self._safe_list(actual_value)) != set(expected_value):
                        penalty += 10
                        deductions.append(f"final state mismatch for {path_prefix}{key} (-10)")
                        failure_labels.add("state_staleness")
                    continue
                if actual_value != expected_value:
                    penalty += 10
                    deductions.append(f"final state mismatch for {path_prefix}{key} (-10)")
                    failure_labels.add("state_staleness")
                continue

            if actual_value != expected_value:
                penalty += 10
                deductions.append(f"final state mismatch for {path_prefix}{key} (-10)")
                failure_labels.add("state_staleness")

        return penalty

    def _get_nested_value(self, payload: dict[str, Any], path: str) -> Any:
        """ドット区切りの path をたどってネストした値を取得する。"""

        current: Any = payload
        for key in path.split("."):
            if not isinstance(current, dict) or key not in current:
                return None
            current = current[key]
        return current

    def _has_meaningful_artifact_content(self, field_values: dict[str, Any]) -> bool:
        """成果物に 1 つでも意味のある非空値があるかを返す。"""

        for value in field_values.values():
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            if isinstance(value, list | dict) and not value:
                continue
            return True
        return False

    def _safe_list(self, value: Any) -> list[Any]:
        """評価処理を安全にするため任意値を list として正規化する。"""

        return value if isinstance(value, list) else []


def evaluate_case(
    case: Case,
    execution_log: ExecutionLog,
) -> EvaluationResult:
    """クラスベース evaluator を呼ぶ後方互換用ラッパー。"""

    return RuleBasedEvaluator().evaluate(case=case, execution_log=execution_log)
