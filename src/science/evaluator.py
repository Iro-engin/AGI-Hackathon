"""science の危険対処 execution log を採点して保存する evaluator。"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.models import Case, ExecutionLog, dump_json
from src.rule_evaluator import RuleBasedEvaluator


@dataclass
class HazardQuestionCheckpoint:
    """hazard ごとの質問 checkpoint 指標。

    Attributes:
        hazard_id: 対象 hazard。
        turn_limit: hazard 開始から何 turn 以内か。
        required_target_count: 聞くべき項目数。
        resolved_target_count: その turn までに回収した項目数。
        total_question_count: その turn までの質問数。
        redundant_question_count: 過剰質問数。
        redundant_question_ratio: 過剰質問割合。
    """

    hazard_id: str
    turn_limit: int
    required_target_count: int
    resolved_target_count: int
    total_question_count: int
    redundant_question_count: int
    redundant_question_ratio: float


@dataclass
class AggregateCheckpointMetric:
    """全 hazard をまとめた checkpoint 指標。

    Attributes:
        turn_limit: 集約対象の turn 上限。
        mean_resolved_ratio: 質問回収率の平均。
        mean_redundant_ratio: 過剰質問割合の平均。
    """

    turn_limit: int
    mean_resolved_ratio: float
    mean_redundant_ratio: float


@dataclass
class HazardEpisodeEvaluation:
    """1 hazard への対応エピソードを評価した結果。

    Attributes:
        hazard_id: 対象 hazard。
        source_formula: この hazard に対応する化学式。
        start_turn: 対応開始 turn。
        answer_turn: 最初に 1 手法へ絞った応答を返した turn。
        answered_within_30_turns: 30 turn 以内に回答できたか。
        required_question_target_count: 質問すべき項目数。
        resolved_question_target_count_before_answer: 回答までに回収できた項目数。
        redundant_question_ratio_before_answer: 回答までの過剰質問割合。
        ambiguous_execution: 未解決項目が残るのに 1 手法へ絞ったか。
        direct_execution_without_extra_questions: 余計な質問なしで実行できたか。
        selected_procedure_id: 実際に選んだ procedure_id。
        expected_procedure_id: 正解側 procedure_id。
        plan_match_score: 手順番号一致率。
        feasible_under_constraints: 道具制約下で実行可能か。
        checkpoints: 1/3/5 turn の質問指標。
    """

    hazard_id: str
    source_formula: str
    start_turn: int
    answer_turn: int | None
    answered_within_30_turns: bool
    required_question_target_count: int
    resolved_question_target_count_before_answer: int
    redundant_question_ratio_before_answer: float
    ambiguous_execution: bool
    direct_execution_without_extra_questions: bool
    selected_procedure_id: str | None
    expected_procedure_id: str
    plan_match_score: float = 0.0
    feasible_under_constraints: bool = False
    checkpoints: list[HazardQuestionCheckpoint] = field(default_factory=list)


@dataclass
class ScienceEvaluationResult:
    """science ケース全体の評価結果。

    Attributes:
        base_total_score: 汎用 evaluator の総合点。
        science_score: science 固有指標の総合点。
        blended_total_score: 両者を合成した最終点。
        answered_within_30_rate: 30 turn 以内に回答できた hazard の割合。
        plan_match_rate: procedure と手順の一致率平均。
        feasibility_rate: 制約下で実行可能だった割合。
        ambiguous_execution_rate: 曖昧なまま絞ってしまった割合。
        direct_execution_rate: 余計な質問なしに実行できた割合。
        question_resolution_rate_before_answer: 回答までの質問回収率。
        redundant_question_ratio_before_answer: 回答までの過剰質問割合。
        aggregated_checkpoints: 1/3/5 turn 集約指標。
        hazard_metrics: hazard ごとの詳細。
        deductions: 減点理由一覧。
        failure_labels: 失敗ラベル一覧。
    """

    base_total_score: int
    science_score: int
    blended_total_score: int
    answered_within_30_rate: float
    plan_match_rate: float
    feasibility_rate: float
    ambiguous_execution_rate: float
    direct_execution_rate: float
    question_resolution_rate_before_answer: float
    redundant_question_ratio_before_answer: float
    aggregated_checkpoints: list[dict[str, Any]] = field(default_factory=list)
    hazard_metrics: list[dict[str, Any]] = field(default_factory=list)
    deductions: list[str] = field(default_factory=list)
    failure_labels: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON 出力しやすい dict に変換する。"""

        return asdict(self)


def parse_args() -> argparse.Namespace:
    """CLI 引数を解釈する。"""

    parser = argparse.ArgumentParser(
        description="science hazard-response execution log を評価する"
    )
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument("--case", dest="case_path", type=Path)
    target_group.add_argument("--case-dir", dest="case_dir", type=Path)
    parser.add_argument("--execution-log", dest="execution_log_path", type=Path)
    parser.add_argument("--execution-log-dir", dest="execution_log_dir", type=Path)
    parser.add_argument("--output", dest="output_path", type=Path)
    parser.add_argument("--output-dir", dest="output_dir", type=Path)
    args = parser.parse_args()

    if args.case_path and args.execution_log_path is None:
        parser.error("--execution-log is required when using --case")
    if args.case_dir and args.execution_log_dir is None:
        parser.error("--execution-log-dir is required when using --case-dir")
    if args.case_path and args.execution_log_dir is not None:
        parser.error("--execution-log-dir cannot be used with --case")
    if args.case_dir and args.execution_log_path is not None:
        parser.error("--execution-log cannot be used with --case-dir")
    if args.case_path and args.output_dir is not None:
        parser.error("--output-dir cannot be used with --case")
    if args.case_dir and args.output_path is not None:
        parser.error("--output cannot be used with --case-dir")
    return args


def evaluate_science_execution(
    case: Case,
    execution_log: ExecutionLog,
) -> ScienceEvaluationResult:
    """science ケース向けの追加観点で execution log を採点する。"""

    base_result = RuleBasedEvaluator().evaluate(case, execution_log)
    deductions = list(base_result.deductions)
    state_by_turn = build_state_by_turn(case)
    hazard_episodes = build_hazard_episodes(case, execution_log, state_by_turn)
    hazard_metrics = [
        evaluate_hazard_episode(
            case=case,
            execution_log=execution_log,
            episode=episode,
            state_by_turn=state_by_turn,
            deductions=deductions,
        )
        for episode in hazard_episodes
    ]
    if not hazard_metrics:
        raise ValueError("no hazard episodes could be derived from the case")

    answered_within_30_rate = average_bool(
        metric.answered_within_30_turns for metric in hazard_metrics
    )
    plan_match_rate = average_float(metric.plan_match_score for metric in hazard_metrics)
    feasibility_rate = average_bool(metric.feasible_under_constraints for metric in hazard_metrics)
    ambiguous_execution_rate = average_bool(metric.ambiguous_execution for metric in hazard_metrics)
    direct_execution_rate = average_bool(
        metric.direct_execution_without_extra_questions for metric in hazard_metrics
    )
    resolved_total = sum(
        metric.resolved_question_target_count_before_answer for metric in hazard_metrics
    )
    required_total = sum(metric.required_question_target_count for metric in hazard_metrics)
    question_resolution_rate_before_answer = (
        resolved_total / required_total if required_total else 1.0
    )
    redundant_question_ratio_before_answer = average_float(
        metric.redundant_question_ratio_before_answer for metric in hazard_metrics
    )
    aggregated_checkpoints = aggregate_checkpoints(hazard_metrics)

    science_score = round(
        100
        * (
            answered_within_30_rate * 0.20
            + plan_match_rate * 0.25
            + feasibility_rate * 0.20
            + (1 - ambiguous_execution_rate) * 0.15
            + direct_execution_rate * 0.05
            + question_resolution_rate_before_answer * 0.10
            + (1 - redundant_question_ratio_before_answer) * 0.05
        )
    )
    blended_total_score = round(base_result.total_score * 0.40 + science_score * 0.60)

    failure_labels = list(base_result.failure_labels)
    if answered_within_30_rate < 0.8:
        failure_labels.append("science_delivery")
    if plan_match_rate < 0.6:
        failure_labels.append("science_plan_match")
    if feasibility_rate < 0.8:
        failure_labels.append("science_feasibility")
    if ambiguous_execution_rate > 0.2:
        failure_labels.append("science_ambiguous_execution")
    if question_resolution_rate_before_answer < 0.8:
        failure_labels.append("science_missing_questions")

    return ScienceEvaluationResult(
        base_total_score=base_result.total_score,
        science_score=science_score,
        blended_total_score=blended_total_score,
        answered_within_30_rate=round(answered_within_30_rate, 4),
        plan_match_rate=round(plan_match_rate, 4),
        feasibility_rate=round(feasibility_rate, 4),
        ambiguous_execution_rate=round(ambiguous_execution_rate, 4),
        direct_execution_rate=round(direct_execution_rate, 4),
        question_resolution_rate_before_answer=round(question_resolution_rate_before_answer, 4),
        redundant_question_ratio_before_answer=round(redundant_question_ratio_before_answer, 4),
        aggregated_checkpoints=[asdict(item) for item in aggregated_checkpoints],
        hazard_metrics=[asdict(item) for item in hazard_metrics],
        deductions=deductions,
        failure_labels=sorted(set(failure_labels)),
    )


def build_hazard_episodes(
    case: Case,
    execution_log: ExecutionLog,
    state_by_turn: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    """case から hazard 対応エピソードの開始点を抽出する。"""

    episodes: list[dict[str, Any]] = []
    seen_hazards: set[str] = set()
    initial_hazards = [
        str(item)
        for item in case.initial_state.reference_data.get("active_hazard_ids", [])
    ]
    for hazard_id in initial_hazards:
        episodes.append({"hazard_id": hazard_id, "start_turn": 1})
        seen_hazards.add(hazard_id)

    max_turn = max(
        [0, *state_by_turn.keys(), *(action.turn for action in execution_log.actions)],
    )
    previous_active = set(initial_hazards)
    for turn in range(1, max_turn + 1):
        current_active = {
            str(item)
            for item in (
                state_by_turn.get(turn, {})
                .get("reference_data", {})
                .get("active_hazard_ids", [])
            )
        }
        new_hazards = [
            hazard_id for hazard_id in current_active if hazard_id not in previous_active
        ]
        for hazard_id in new_hazards:
            if hazard_id not in seen_hazards:
                episodes.append({"hazard_id": hazard_id, "start_turn": turn})
                seen_hazards.add(hazard_id)
        previous_active = current_active

    episodes.sort(key=lambda item: item["start_turn"])
    for index, episode in enumerate(episodes):
        next_start_turn = (
            episodes[index + 1]["start_turn"]
            if index + 1 < len(episodes)
            else max_turn + 30
        )
        episode["end_turn"] = next_start_turn - 1
    return episodes


def evaluate_hazard_episode(
    *,
    case: Case,
    execution_log: ExecutionLog,
    episode: dict[str, Any],
    state_by_turn: dict[int, dict[str, Any]],
    deductions: list[str],
) -> HazardEpisodeEvaluation:
    """1 つの hazard 対応エピソードを採点する。"""

    hazard_id = episode["hazard_id"]
    start_turn = int(episode["start_turn"])
    end_turn = int(episode["end_turn"])
    visible_state = state_by_turn.get(start_turn, case.initial_state.model_dump(mode="python"))
    target_count, question_targets = get_question_targets(visible_state, hazard_id)
    questions = [
        question
        for question in execution_log.questions_asked
        if start_turn <= question.turn <= end_turn
    ]
    answer_actions = [
        action
        for action in execution_log.actions
        if start_turn <= action.turn <= end_turn and action.action_type != "ask"
    ]
    answer_turn = answer_actions[0].turn if answer_actions else None
    answered_within_30_turns = answer_turn is not None and (answer_turn - start_turn) <= 30

    checkpoint_stats, resolved_before_answer, redundant_ratio_before_answer, total_questions = (
        evaluate_question_checkpoints(
            case=case,
            questions=questions,
            state_by_turn=state_by_turn,
            hazard_id=hazard_id,
            start_turn=start_turn,
            answer_turn=answer_turn,
            question_targets=question_targets,
        )
    )

    decision_fields = collect_decision_fields(answer_actions)
    manual = get_hazard_manual(visible_state, hazard_id)
    runtime_fact = get_runtime_fact(case.initial_state.reference_data, hazard_id)
    expected_procedure = select_expected_response(
        manual=manual,
        runtime_fact=runtime_fact,
        available_tools=get_available_tools(
            state_by_turn.get(answer_turn or start_turn, visible_state)
        ),
    )
    selected_procedure_id = decision_fields.get("procedure_id")
    plan_match_score = calculate_plan_match_score(
        selected_procedure_id=selected_procedure_id,
        expected_procedure_id=expected_procedure["procedure_id"],
    )
    feasible_under_constraints = evaluate_feasibility(
        manual=manual,
        selected_procedure_id=selected_procedure_id,
        available_tools=get_available_tools(
            state_by_turn.get(answer_turn or start_turn, visible_state)
        ),
    )
    ambiguous_execution = (
        answer_turn is not None
        and resolved_before_answer < target_count
        and not can_directly_resolve_from_visible_state(visible_state, hazard_id)
    )
    direct_execution_without_extra_questions = (
        answer_turn is not None
        and total_questions == 0
        and can_directly_resolve_from_visible_state(visible_state, hazard_id)
    )
    if not answered_within_30_turns:
        deductions.append(f"science: {hazard_id} was not answered within 30 turns")
    if ambiguous_execution:
        deductions.append(f"science: {hazard_id} was executed while still ambiguous")
    if redundant_ratio_before_answer > 0.3:
        deductions.append(
            f"science: {hazard_id} had a high redundant-question ratio before answer"
        )
    if plan_match_score < 0.6:
        deductions.append(f"science: {hazard_id} plan match was low")
    if not feasible_under_constraints:
        deductions.append(f"science: {hazard_id} response was not feasible under constraints")

    return HazardEpisodeEvaluation(
        hazard_id=hazard_id,
        source_formula=get_source_formula(visible_state, hazard_id) or "unknown_formula",
        start_turn=start_turn,
        answer_turn=answer_turn,
        answered_within_30_turns=answered_within_30_turns,
        required_question_target_count=target_count,
        resolved_question_target_count_before_answer=resolved_before_answer,
        redundant_question_ratio_before_answer=round(redundant_ratio_before_answer, 4),
        ambiguous_execution=ambiguous_execution,
        direct_execution_without_extra_questions=direct_execution_without_extra_questions,
        selected_procedure_id=selected_procedure_id,
        expected_procedure_id=expected_procedure["procedure_id"],
        plan_match_score=round(plan_match_score, 4),
        feasible_under_constraints=feasible_under_constraints,
        checkpoints=checkpoint_stats,
    )


def evaluate_question_checkpoints(
    *,
    case: Case,
    questions: list[Any],
    state_by_turn: dict[int, dict[str, Any]],
    hazard_id: str,
    start_turn: int,
    answer_turn: int | None,
    question_targets: list[str],
) -> tuple[list[HazardQuestionCheckpoint], int, float, int]:
    """1/3/5 turn と回答時点までの質問効率を計算する。"""

    resolved_targets: set[str] = set()
    normalized_targets = {normalize_question_text(target): target for target in question_targets}
    seen_questions: set[str] = set()
    question_records: list[dict[str, Any]] = []

    for question in sorted(questions, key=lambda item: item.turn):
        normalized_question = normalize_question_text(question.question)
        matched_targets = [
            target
            for normalized_target, target in normalized_targets.items()
            if normalized_question == normalized_target
        ]
        new_matches = [target for target in matched_targets if target not in resolved_targets]
        visible_state = state_by_turn.get(
            question.turn,
            case.initial_state.model_dump(mode="python"),
        )
        normalized_visible_facts = {
            normalize_question_text(fact) for fact in collect_visible_known_facts(visible_state)
        }
        redundant = False

        if new_matches:
            for target in new_matches:
                resolved_targets.add(target)
        elif matched_targets:
            redundant = True
        elif normalized_question in seen_questions:
            redundant = True
        elif normalized_question in normalized_visible_facts:
            redundant = True
        seen_questions.add(normalized_question)
        question_records.append(
            {
                "turn": question.turn,
                "resolved_count_after": len(resolved_targets),
                "is_redundant": redundant,
            }
        )

    checkpoints: list[HazardQuestionCheckpoint] = []
    for turn_limit in (1, 3, 5):
        relevant = [
            record
            for record in question_records
            if (record["turn"] - start_turn + 1) <= turn_limit
        ]
        total_questions = len(relevant)
        redundant_questions = sum(record["is_redundant"] for record in relevant)
        resolved_count = max((record["resolved_count_after"] for record in relevant), default=0)
        redundant_ratio = redundant_questions / total_questions if total_questions else 0.0
        checkpoints.append(
            HazardQuestionCheckpoint(
                hazard_id=hazard_id,
                turn_limit=turn_limit,
                required_target_count=len(question_targets),
                resolved_target_count=resolved_count,
                total_question_count=total_questions,
                redundant_question_count=redundant_questions,
                redundant_question_ratio=round(redundant_ratio, 4),
            )
        )

    answer_records = (
        question_records
        if answer_turn is None
        else [record for record in question_records if record["turn"] < answer_turn]
    )
    resolved_before_answer = max(
        (record["resolved_count_after"] for record in answer_records),
        default=0,
    )
    redundant_before_answer = sum(record["is_redundant"] for record in answer_records)
    total_before_answer = len(answer_records)
    redundant_ratio_before_answer = (
        redundant_before_answer / total_before_answer if total_before_answer else 0.0
    )
    return checkpoints, resolved_before_answer, redundant_ratio_before_answer, total_before_answer


def collect_decision_fields(actions: list[Any]) -> dict[str, str]:
    """対応エピソード中の action notes から意思決定項目を集める。"""

    decision_fields: dict[str, str] = {}
    for action in actions:
        payload = parse_action_note(action.notes)
        if "procedure_id" in payload:
            decision_fields["procedure_id"] = str(payload["procedure_id"])
    return decision_fields


def parse_action_note(note: str | None) -> dict[str, Any]:
    """`ActionLog.notes` に埋めた JSON を復元する。"""

    if not note:
        return {}
    try:
        payload = json.loads(note)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def get_question_targets(state: dict[str, Any], hazard_id: str) -> tuple[int, list[str]]:
    """対象 hazard の質問ターゲット一覧を返す。"""

    for entry in state.get("reference_data", {}).get("hazard_question_targets", []):
        if isinstance(entry, dict) and entry.get("hazard_id") == hazard_id:
            targets = [str(item) for item in entry.get("targets", [])]
            return len(targets), targets
    return 0, []


def get_hazard_manual(state: dict[str, Any], hazard_id: str) -> dict[str, Any]:
    """対象 hazard のマニュアルを返す。"""

    for manual in state.get("reference_data", {}).get("hazard_manuals", []):
        if isinstance(manual, dict) and manual.get("hazard_id") == hazard_id:
            return manual
    raise ValueError(f"manual not found for {hazard_id}")


def get_runtime_fact(reference_data: dict[str, Any], hazard_id: str) -> dict[str, Any]:
    """対象 hazard の隠し事実を返す。"""

    for fact in reference_data.get("hidden_hazard_facts", []):
        if isinstance(fact, dict) and fact.get("hazard_id") == hazard_id:
            return fact
    raise ValueError(f"runtime fact not found for {hazard_id}")


def get_available_tools(state: dict[str, Any]) -> list[str]:
    """その時点で利用可能な道具一覧を返す。"""

    reference_data = state.get("reference_data", {})
    return [str(item) for item in reference_data.get("initial_tool_constraints", [])]


def get_source_formula(state: dict[str, Any], hazard_id: str) -> str | None:
    """hazard に対応する化学式を返す。"""

    for link in state.get("reference_data", {}).get("formula_hazard_links", []):
        if isinstance(link, dict) and link.get("hazard_id") == hazard_id:
            return str(link.get("formula_id"))
    return None


def select_expected_response(
    *,
    manual: dict[str, Any],
    runtime_fact: dict[str, Any],
    available_tools: list[str],
) -> dict[str, Any]:
    """隠し事実から正解の procedure を選ぶ。"""

    for procedure in manual.get("procedures", []):
        if not isinstance(procedure, dict):
            continue
        if str(procedure.get("procedure_id")) == str(runtime_fact.get("expected_procedure_id")):
            return procedure
    raise ValueError(f"expected procedure not found for {manual.get('hazard_id')}")


def calculate_plan_match_score(
    *,
    selected_procedure_id: str | None,
    expected_procedure_id: str,
) -> float:
    """選んだ procedure_id と正解の一致率を返す。"""

    return 1.0 if selected_procedure_id == expected_procedure_id else 0.0


def evaluate_feasibility(
    *,
    manual: dict[str, Any],
    selected_procedure_id: str | None,
    available_tools: list[str],
) -> bool:
    """選んだ手順が道具制約下で実行可能かどうかを見る。"""

    for procedure in manual.get("procedures", []):
        if not isinstance(procedure, dict):
            continue
        if selected_procedure_id and selected_procedure_id != procedure.get("procedure_id"):
            continue
        required = {str(t) for t in procedure.get("required_tools", [])}
        if required.issubset(set(available_tools)):
            return True
    return False


def can_directly_resolve_from_visible_state(state: dict[str, Any], hazard_id: str) -> bool:
    """visible state だけで質問ターゲットが埋まっているかを見る。"""

    visible_text = normalize_question_text(" ".join(collect_visible_known_facts(state)))
    _, targets = get_question_targets(state, hazard_id)
    if not targets:
        return True
    return all(normalize_question_text(target) in visible_text for target in targets)


def collect_visible_known_facts(state: dict[str, Any]) -> list[str]:
    """その turn で visible state 上に明示されている事実を集める。"""

    reference_data = state.get("reference_data", {})
    facts: list[str] = []
    for key in [
        "running_formulas",
        "active_hazard_ids",
        "resolved_hazard_ids",
        "initial_tool_constraints",
        "selected_constraint_types",
    ]:
        facts.extend(collect_text_facts(reference_data.get(key, [])))
    for key in ["formula_hazard_links", "hazard_manuals", "hazard_question_targets"]:
        for item in reference_data.get(key, []):
            if isinstance(item, dict):
                facts.append(dump_json(item))
    return facts


def aggregate_checkpoints(
    hazard_metrics: list[HazardEpisodeEvaluation],
) -> list[AggregateCheckpointMetric]:
    """hazard ごとの 1/3/5 turn 指標を集約する。"""

    grouped: dict[int, list[HazardQuestionCheckpoint]] = {1: [], 3: [], 5: []}
    for metric in hazard_metrics:
        for checkpoint in metric.checkpoints:
            grouped[checkpoint.turn_limit].append(checkpoint)

    aggregated: list[AggregateCheckpointMetric] = []
    for turn_limit in (1, 3, 5):
        checkpoints = grouped[turn_limit]
        if not checkpoints:
            aggregated.append(
                AggregateCheckpointMetric(
                    turn_limit=turn_limit,
                    mean_resolved_ratio=0.0,
                    mean_redundant_ratio=0.0,
                )
            )
            continue
        mean_resolved_ratio = average_float(
            checkpoint.resolved_target_count / checkpoint.required_target_count
            if checkpoint.required_target_count
            else 1.0
            for checkpoint in checkpoints
        )
        mean_redundant_ratio = average_float(
            checkpoint.redundant_question_ratio for checkpoint in checkpoints
        )
        aggregated.append(
            AggregateCheckpointMetric(
                turn_limit=turn_limit,
                mean_resolved_ratio=round(mean_resolved_ratio, 4),
                mean_redundant_ratio=round(mean_redundant_ratio, 4),
            )
        )
    return aggregated


def average_bool(values: Any) -> float:
    """真偽値列の平均を返す。"""

    values_list = [1.0 if value else 0.0 for value in values]
    return sum(values_list) / len(values_list) if values_list else 0.0


def average_float(values: Any) -> float:
    """数値列の平均を返す。"""

    values_list = [float(value) for value in values]
    return sum(values_list) / len(values_list) if values_list else 0.0


def collect_text_facts(value: Any) -> list[str]:
    """`list[str]` または `str` を文字列配列へ正規化する。"""

    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def has_token_overlap(left: str, right: str) -> bool:
    """英数字と underscore を中心に大まかな token 重なりを見る。"""

    return bool(tokenize(left) & tokenize(right))


def tokenize(text: str) -> set[str]:
    """文字列を粗い token 集合へ分解する。"""

    return {token for token in re.findall(r"[A-Za-z0-9_]+", text.lower()) if token}


def normalize_question_text(text: str) -> str:
    """質問項目の比較用に文字列を正規化する。"""

    tokens = tokenize(text)
    return " ".join(sorted(tokens))


def build_state_by_turn(case: Case) -> dict[int, dict[str, Any]]:
    """各 turn まで event を反映した可視 state を再現する。"""

    current_state = deep_copy_dict(case.initial_state.model_dump(mode="python"))
    state_by_turn: dict[int, dict[str, Any]] = {0: current_state}
    events_by_turn: dict[int, list[Any]] = {}
    for event in case.events:
        events_by_turn.setdefault(event.turn, []).append(event)

    max_turn = max((event.turn for event in case.events), default=0)
    for turn in range(1, max_turn + 1):
        current_state = deep_copy_dict(current_state)
        for event in events_by_turn.get(turn, []):
            apply_event_delta(current_state, event.delta)
        state_by_turn[turn] = current_state
    return state_by_turn


def apply_event_delta(current_state: dict[str, Any], delta: dict[str, Any]) -> None:
    """event.delta を evaluator 内部の state に反映する。"""

    for key, value in delta.items():
        if key == "reference_data":
            current_state["reference_data"] = deep_merge_dict(
                dict(current_state.get("reference_data", {})),
                value,
            )
            continue
        current_state[key] = value


def deep_merge_dict(base: dict[str, Any], delta: dict[str, Any]) -> dict[str, Any]:
    """ネストした辞書を再帰的にマージする。"""

    merged = dict(base)
    for key, value in delta.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dict(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def deep_copy_dict(value: dict[str, Any]) -> dict[str, Any]:
    """単純な辞書・配列構造を深いコピーで複製する。"""

    copied: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, dict):
            copied[key] = deep_copy_dict(item)
        elif isinstance(item, list):
            copied[key] = deep_copy_list(item)
        else:
            copied[key] = item
    return copied


def deep_copy_list(values: list[Any]) -> list[Any]:
    """単純な配列構造を深いコピーで複製する。"""

    copied: list[Any] = []
    for item in values:
        if isinstance(item, dict):
            copied.append(deep_copy_dict(item))
        elif isinstance(item, list):
            copied.append(deep_copy_list(item))
        else:
            copied.append(item)
    return copied


def evaluate_case_file(
    *,
    case_path: Path,
    execution_log_path: Path,
) -> ScienceEvaluationResult:
    """1 件の case と execution log を評価する。"""

    case = Case.from_path(case_path)
    execution_log = ExecutionLog.from_path(execution_log_path)
    return evaluate_science_execution(case, execution_log)


def evaluate_execution_directory(
    *,
    case_dir: Path,
    execution_log_dir: Path,
) -> list[dict[str, Any]]:
    """ディレクトリ配下の case 群をまとめて評価する。"""

    summaries: list[dict[str, Any]] = []
    for case_path in sorted(case_dir.glob("*.json")):
        execution_log_path = execution_log_dir / build_execution_log_filename(case_path)
        if not execution_log_path.exists():
            summaries.append(
                {
                    "task_id": case_path.stem,
                    "case_path": str(case_path),
                    "execution_log_path": str(execution_log_path),
                    "error": "execution log not found",
                }
            )
            continue

        try:
            result = evaluate_case_file(
                case_path=case_path,
                execution_log_path=execution_log_path,
            )
        except Exception as exc:  # noqa: BLE001
            summaries.append(
                {
                    "task_id": case_path.stem,
                    "case_path": str(case_path),
                    "execution_log_path": str(execution_log_path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        summaries.append(
            {
                "task_id": case_path.stem,
                "case_path": str(case_path),
                "execution_log_path": str(execution_log_path),
                "evaluation": result.to_dict(),
            }
        )
    return summaries


def build_execution_log_filename(case_path: Path) -> str:
    """case ファイル名から execution log 名を組み立てる。"""

    stem = case_path.stem
    if stem.startswith("science_"):
        suffix = stem.removeprefix("science_")
        return f"science_result_{suffix}.json"
    return f"{stem}_result.json"


def build_evaluation_filename(case_path: Path) -> str:
    """case ファイル名から evaluation 名を組み立てる。"""

    stem = case_path.stem
    if stem.startswith("science_"):
        suffix = stem.removeprefix("science_")
        return f"science_eval_{suffix}.json"
    return f"{stem}_eval.json"


def write_evaluation_result(path: Path, payload: dict[str, Any]) -> None:
    """評価結果 1 件を JSON として保存する。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{dump_json(payload)}\n", encoding="utf-8")


def main() -> None:
    """CLI エントリポイント。"""

    args = parse_args()

    if args.case_path is not None:
        result = evaluate_case_file(
            case_path=args.case_path,
            execution_log_path=args.execution_log_path,
        )
        payload = {
            "task_id": args.case_path.stem,
            "case_path": str(args.case_path),
            "execution_log_path": str(args.execution_log_path),
            "evaluation": result.to_dict(),
        }
        if args.output_path is not None:
            write_evaluation_result(args.output_path, payload)
        print(dump_json(payload))
        return

    summaries = evaluate_execution_directory(
        case_dir=args.case_dir,
        execution_log_dir=args.execution_log_dir,
    )
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for summary in summaries:
            if "evaluation" not in summary:
                continue
            case_path = Path(summary["case_path"])
            write_evaluation_result(
                args.output_dir / build_evaluation_filename(case_path),
                summary,
            )
    print(dump_json(summaries))


if __name__ == "__main__":
    main()
