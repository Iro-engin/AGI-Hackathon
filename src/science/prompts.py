"""science の危険対処ケース生成に使う blueprint と prompt。"""

from __future__ import annotations

from src.models import dump_json
from src.science.type import ScienceBlueprint, ScienceCaseSeedDraft, ScienceCountProfile

SCIENCE_BLUEPRINTS: tuple[ScienceBlueprint, ...] = (
    ScienceBlueprint(
        slug="pressure_split_alpha",
        label="Handle sequential hazards while several formulas are running",
        scenario_hook=(
            "Several fictional formulas are already running. One hazard is active now, "
            "and another hazard appears later even if the first response was imperfect."
        ),
        formula_examples=("formula_A", "formula_B", "formula_C"),
        hazard_examples=("hazard_X", "hazard_Y", "hazard_Z"),
        tool_examples=("goggles_G", "spanner_S", "shield_P"),
        hard_constraints=(
            "Use only fictional variable names",
            "The first turn should encourage bundled questioning",
            "A later hazard must appear by turn 3",
        ),
    ),
    ScienceBlueprint(
        slug="tool_shortage_beta",
        label="Choose feasible hazard procedures under tool limits",
        scenario_hook=(
            "The worker inventory is limited, so the agent must ask the right questions "
            "and then choose a procedure that is actually executable with the listed tools."
        ),
        formula_examples=("formula_M", "formula_N", "formula_P"),
        hazard_examples=("hazard_M", "hazard_N", "hazard_Q"),
        tool_examples=("mask_H", "wrench_R", "clamp_V"),
        hard_constraints=(
            "The selected procedure must stay feasible under initial_tool_constraints",
            "Question targets are judged only by checklist membership",
        ),
    ),
    ScienceBlueprint(
        slug="compound_shift_gamma",
        label="Re-handle a new hazard after the first response",
        scenario_hook=(
            "The environment treats the first response as completed and then activates a "
            "different hazard on another formula, forcing a second round of questioning."
        ),
        formula_examples=("formula_L", "formula_Y", "formula_Z"),
        hazard_examples=("hazard_K", "hazard_R", "hazard_T"),
        tool_examples=("mask_Q", "tray_B", "barrier_F"),
        hard_constraints=(
            "The second hazard must not reuse the initial active hazard",
            "The same selected_constraint_types must be used across every hazard",
        ),
    ),
)


def select_blueprint(index: int) -> ScienceBlueprint:
    """連番からケース生成用の blueprint を選ぶ。"""

    return SCIENCE_BLUEPRINTS[(index - 1) % len(SCIENCE_BLUEPRINTS)]


def build_seed_prompt(
    *,
    task_id: str,
    difficulty: str,
    count_profile: ScienceCountProfile,
    blueprint: ScienceBlueprint,
    previous_error: str | None,
) -> str:
    """Phase 1 の seed 生成 prompt を返す。"""

    sections = [
        "Create one science-domain hazard-response benchmark seed.",
        f"task_id: {task_id}",
        f"difficulty: {difficulty}",
        f"fixed_hazard_count: {count_profile.hazard_count}",
        f"fixed_selected_constraint_count: {count_profile.selected_constraint_count}",
        f"theme_label: {blueprint.label}",
        f"scenario_hook: {blueprint.scenario_hook}",
        f"formula_examples: {', '.join(blueprint.formula_examples)}",
        f"hazard_examples: {', '.join(blueprint.hazard_examples)}",
        f"tool_examples: {', '.join(blueprint.tool_examples)}",
        "hard_constraints:",
        *[f"- {constraint}" for constraint in blueprint.hard_constraints],
        "",
        "Requirements:",
        "- Output language: English for every field value.",
        "- Use only fictional names such as formula_A, hazard_X, goggles_G.",
        "- Do not use real compounds, real hazards, or real lab equipment names.",
        (
            "- Create exactly "
            f"{count_profile.hazard_count} formula_hazard_links, no more and no fewer."
        ),
        (
            "- Choose exactly "
            f"{count_profile.selected_constraint_count} selected_constraint_types."
        ),
        "- Never reuse formula_id across formula_hazard_links.",
        "- Never reuse hazard_id across formula_hazard_links.",
        "- Do not generate participants, deadline, or budget fields in this phase.",
        (
            "- selected_constraint_types must be chosen only from: pressure_pa, "
            "catalyst_present, flammable_nearby, escape_easy, water_usable."
        ),
        "- active_hazard_ids must contain exactly 1 hazard at the initial state.",
        "- resolved_hazard_ids should usually start empty.",
        "- initial_tool_constraints must contain the worker's currently available tools.",
        "- Create exactly 3 events with turns fixed to 1, 2, and 3.",
        (
            "- Every event must include type, and the only allowed values are "
            "state_change or new_constraint."
        ),
        "- By turn 3, a new hazard different from the initial one must become active.",
        "- The later hazard event must happen regardless of response quality.",
        (
            "- Do not generate hazard_manuals, hazard_question_targets, or hidden_hazard_facts "
            "yet. This phase only creates the seed."
        ),
    ]

    if previous_error:
        sections.extend(["", "Fix the following issue from the previous attempt:", previous_error])

    return "\n".join(sections)


def build_manual_prompt(
    *,
    task_id: str,
    difficulty: str,
    count_profile: ScienceCountProfile,
    blueprint: ScienceBlueprint,
    seed: ScienceCaseSeedDraft,
    previous_error: str | None,
) -> str:
    """Phase 2 の manual 群生成 prompt を返す。"""

    seed_json = dump_json(seed.model_dump(mode="python"))
    selected_constraint_types = seed.reference_data.selected_constraint_types

    sections = [
        "Create hazard manuals, question targets, and hidden runtime facts for this seed.",
        f"task_id: {task_id}",
        f"difficulty: {difficulty}",
        f"fixed_hazard_count: {count_profile.hazard_count}",
        f"fixed_question_target_count: {count_profile.question_target_count}",
        f"theme_label: {blueprint.label}",
        "",
        "Seed JSON:",
        seed_json,
        "",
        "Requirements:",
        "- Output language: English for every field value.",
        (
            "- Create exactly one hazard_manual, one hazard_question_targets entry, and one "
            "hidden_hazard_facts entry for every hazard_id in formula_hazard_links, so each "
            f"of those top-level arrays must contain exactly {count_profile.hazard_count} items."
        ),
        (
            "- Every hazard_question_targets.targets list must contain exactly one explicit "
            "question for each selected_constraint_type."
        ),
        (
            "- Questions are scored only by exact checklist membership, so write clear direct "
            "questions and do not include reasons."
        ),
        (
            "- Each hazard manual must contain multiple procedures. Use applicable_conditions "
            "instead of fixed pressure/catalyst fields."
        ),
        (
            f"- Each procedure's applicable_conditions must have between 1 and "
            f"{count_profile.selected_constraint_count} entries (= number of selected constraint "
            f"types). The schema enforces this upper bound; do not exceed it."
        ),
        (
            "- Every applicable_conditions entry must use only the selected_constraint_types from "
            f"the seed: {', '.join(selected_constraint_types)}."
        ),
        (
            "- If constraint_type is pressure_pa, applicable_conditions must use mode=range or "
            "mode=irrelevant, and hidden runtime values must use numeric_value."
        ),
        (
            "- If constraint_type is catalyst_present, flammable_nearby, escape_easy, or "
            "water_usable, applicable_conditions must use mode=bool or mode=irrelevant, and "
            "hidden runtime values must use bool_value."
        ),
        (
            "- hidden_hazard_facts.constraint_values must contain exactly one value for each "
            "selected_constraint_type."
        ),
        (
            "- hidden_hazard_facts.expected_procedure_id must match exactly one procedure in the "
            "same hazard manual."
        ),
        (
            "- At least one procedure per hazard should require stopping the source formula, and "
            "the expected procedure may also require stop_source_formula."
        ),
        (
            "- required_tools may use only tools listed in initial_tool_constraints or fictional "
            "subsets consistent with them."
        ),
        "- Make the expected procedure executable with the initial_tool_constraints.",
        "- Keep procedure steps short and imperative.",
    ]

    if previous_error:
        sections.extend(["", "Fix the following issue from the previous attempt:", previous_error])

    return "\n".join(sections)
