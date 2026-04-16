"""science の危険対処ケースを組み立てる関数群。"""

from __future__ import annotations

from src.models import (
    BenchmarkEvent,
    Case,
    GoalCondition,
    InitialState,
    RequiredArtifact,
    SemanticCheck,
)
from src.science.type import (
    DIFFICULTY_DRAFT_CLASS,
    ScienceBenchmarkEventDraft,
    ScienceCaseDraft,
    ScienceFormulaHazardLink,
    ScienceReferenceData,
    validate_difficulty_profile_data,
    validate_formula_hazard_bijection_data,
    validate_manual_constraint_coverage_data,
    validate_reference_hazard_sets,
)

DEFAULT_DOMAIN = "science_hazard_response"
DEFAULT_ALLOWED_ACTIONS = ["ask", "do"]
DEFAULT_ARTIFACT_IDS = ("hazard_response_plan", "hazard_execution_log")
DEFAULT_DEADLINE = "2099-12-31T23:59:59+09:00"
DEFAULT_PARTICIPANTS = ["science_operator", "science_supervisor"]
DEFAULT_BUDGET = 0


def build_required_artifacts() -> list[RequiredArtifact]:
    """science ドメインで共通に使う成果物定義を返す。"""

    return [
        RequiredArtifact(
            artifact_id="hazard_response_plan",
            artifact_type="document",
            required_fields=[
                "active_hazard",
                "source_formula",
                "required_questions",
                "selected_procedure_id",
                "procedure_steps",
                "stop_targets",
                "tool_feasibility",
            ],
            semantic_checks=[
                SemanticCheck(
                    artifact_field="running_formulas",
                    state_path="reference_data.running_formulas",
                ),
                SemanticCheck(
                    artifact_field="initial_tool_constraints",
                    state_path="reference_data.initial_tool_constraints",
                ),
                SemanticCheck(
                    artifact_field="active_hazard_ids",
                    state_path="reference_data.active_hazard_ids",
                ),
            ],
            latest_version_required=True,
        ),
        RequiredArtifact(
            artifact_id="hazard_execution_log",
            artifact_type="document",
            required_fields=[
                "active_hazard",
                "asked_questions",
                "received_answers",
                "chosen_actions",
                "stopped_formulas",
                "remaining_unknowns",
            ],
            semantic_checks=[
                SemanticCheck(
                    artifact_field="initial_tool_constraints",
                    state_path="reference_data.initial_tool_constraints",
                ),
                SemanticCheck(
                    artifact_field="resolved_hazard_ids",
                    state_path="reference_data.resolved_hazard_ids",
                ),
            ],
            latest_version_required=True,
        ),
    ]


def build_case(task_id: str, difficulty: str, draft: ScienceCaseDraft) -> Case:
    """モデル出力を repository 共通の `Case` 形式へ変換する。"""

    draft_class = DIFFICULTY_DRAFT_CLASS.get(difficulty, ScienceCaseDraft)
    validated_draft = draft_class.model_validate(
        draft.model_dump(mode="python"),
        context={"difficulty": difficulty},
    )
    validate_science_draft(validated_draft, difficulty)

    return Case(
        task_id=task_id,
        domain=DEFAULT_DOMAIN,
        difficulty=difficulty,
        initial_request=validated_draft.initial_request,
        initial_state=InitialState(
            timezone="Asia/Tokyo",
            deadline=DEFAULT_DEADLINE,
            participants=list(DEFAULT_PARTICIPANTS),
            budget=DEFAULT_BUDGET,
            required_artifacts=build_required_artifacts(),
            constraints=[],
            task_dependencies=[],
            reference_data=validated_draft.reference_data.model_dump(mode="python"),
        ),
        allowed_actions=DEFAULT_ALLOWED_ACTIONS,
        events=normalize_events(validated_draft.events),
        goal_condition=GoalCondition(
            must_satisfy_latest_state=True,
            required_artifacts=list(DEFAULT_ARTIFACT_IDS),
            no_constraint_violation=True,
            must_acknowledge_changes=True,
            must_use_ask_clarification_on_ambiguity=True,
            must_not_finalize_with_unresolved_scope=False,
        ),
        notes=None,
    )


def validate_science_draft(draft: ScienceCaseDraft, difficulty: str) -> None:
    """science draft が要件を満たすか検証する。"""

    if len(draft.events) != 3:
        raise ValueError("events must contain exactly 3 items")

    event_turns = [event.turn for event in draft.events]
    if event_turns != [1, 2, 3]:
        raise ValueError("event turns must be fixed to 1, 2, and 3")

    reference_data = draft.reference_data
    validate_difficulty_profile(
        reference_data=reference_data,
        difficulty=difficulty,
    )
    validate_formula_hazard_bijection(reference_data.formula_hazard_links)
    validate_runtime_fact_coverage(reference_data)
    validate_hazard_manuals(reference_data)
    validate_question_targets(reference_data)
    validate_later_hazard_event(draft)


def validate_difficulty_profile(
    *,
    reference_data: ScienceReferenceData,
    difficulty: str,
) -> None:
    """難易度ごとの hazard 数と質問数を検証する。"""

    validate_difficulty_profile_data(
        difficulty=difficulty,
        hazard_count=len(reference_data.formula_hazard_links),
        question_target_counts=[
            len(entry.targets) for entry in reference_data.hazard_question_targets
        ],
        selected_constraint_count=len(reference_data.selected_constraint_types),
    )


def validate_formula_hazard_bijection(links: list[ScienceFormulaHazardLink]) -> None:
    """化学式と危険の 1 対 1 対応を検証する。"""

    validate_formula_hazard_bijection_data(links)


def validate_runtime_fact_coverage(reference_data: ScienceReferenceData) -> None:
    """manual・question・runtime fact の hazard 集合一致を検証する。"""

    validate_reference_hazard_sets(
        hazard_links=reference_data.formula_hazard_links,
        hazard_manuals=reference_data.hazard_manuals,
        question_targets=reference_data.hazard_question_targets,
        runtime_facts=reference_data.hidden_hazard_facts,
    )
    validate_manual_constraint_coverage_data(
        selected_constraint_types=reference_data.selected_constraint_types,
        hazard_manuals=reference_data.hazard_manuals,
        question_targets=reference_data.hazard_question_targets,
        runtime_facts=reference_data.hidden_hazard_facts,
    )


def validate_hazard_manuals(reference_data: ScienceReferenceData) -> None:
    """各 hazard manual の内部整合性を検証する。"""

    formula_map = {
        link.hazard_id: link.formula_id
        for link in reference_data.formula_hazard_links
    }
    selected_constraint_types = set(reference_data.selected_constraint_types)
    for manual in reference_data.hazard_manuals:
        if formula_map.get(manual.hazard_id) != manual.source_formula:
            raise ValueError(f"{manual.hazard_id} source_formula is inconsistent")
        for procedure in manual.procedures:
            condition_types = {
                condition.constraint_type
                for condition in procedure.applicable_conditions
            }
            if not condition_types.issubset(selected_constraint_types):
                raise ValueError(
                    f"{manual.hazard_id} procedure uses unsupported constraint types"
                )


def validate_question_targets(reference_data: ScienceReferenceData) -> None:
    """質問文が selected_constraint_types をきちんと表しているかを見る。"""

    selected_constraint_types = set(reference_data.selected_constraint_types)
    for entry in reference_data.hazard_question_targets:
        if len(set(entry.targets)) != len(entry.targets):
            raise ValueError(f"{entry.hazard_id} question targets must be unique")
        lowered = " ".join(item.lower() for item in entry.targets)
        for constraint_type in selected_constraint_types:
            keyword = constraint_type_to_keyword(constraint_type)
            if keyword not in lowered:
                raise ValueError(
                    f"{entry.hazard_id} question targets must mention {constraint_type}"
                )


def validate_later_hazard_event(draft: ScienceCaseDraft) -> None:
    """turn 3 までに新しい hazard が active になることを検証する。"""

    initial_hazards = set(draft.reference_data.active_hazard_ids)
    for event in draft.events:
        if event.turn < 3:
            continue
        reference_delta = event.delta.reference_data
        if reference_delta is None or not reference_delta.active_hazard_ids:
            continue
        later_hazards = set(reference_delta.active_hazard_ids)
        if later_hazards - initial_hazards:
            return
    raise ValueError("a later event must activate a new hazard distinct from the initial one")


def normalize_events(events: list[ScienceBenchmarkEventDraft]) -> list[BenchmarkEvent]:
    """event draft を repository 共通の event 形式へ直す。"""

    normalized: list[BenchmarkEvent] = []
    for event in sorted(events, key=lambda item: item.turn):
        normalized.append(
            BenchmarkEvent(
                turn=event.turn,
                type=normalize_event_type(event.type),
                message=event.message,
                delta=event.delta.model_dump(mode="python", exclude_none=True),
                expected_artifact_updates=list(DEFAULT_ARTIFACT_IDS),
                expected_replan_within_turns=max(1, event.expected_replan_within_turns),
            )
        )
    return normalized


def normalize_event_type(event_type: str) -> str:
    """science で許可する event type だけを benchmark 側へ渡す。"""

    normalized = event_type.strip().lower()
    aliases = {
        "constraint_change": "new_constraint",
        "constraint_update": "new_constraint",
        "state_update": "state_change",
    }
    canonical = aliases.get(normalized, normalized)
    allowed_types = {"state_change", "new_constraint"}
    if canonical not in allowed_types:
        raise ValueError(f"unsupported event type after normalization: {event_type}")
    return canonical


def constraint_type_to_keyword(constraint_type: str) -> str:
    """constraint_type を質問文で探すための代表語へ変換する。"""

    mapping = {
        "pressure_pa": "pressure",
        "catalyst_present": "catalyst",
        "flammable_nearby": "flammable",
        "escape_easy": "escape",
        "water_usable": "water",
    }
    return mapping.get(constraint_type, constraint_type)
