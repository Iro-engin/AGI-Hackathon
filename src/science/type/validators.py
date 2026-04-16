"""science 型定義で共通利用する検証関数群。"""

from __future__ import annotations

from typing import Any


def as_mapping(item: Any) -> dict[str, Any]:
    """Pydantic model または dict を dict へ正規化する。"""

    if isinstance(item, dict):
        return item
    if hasattr(item, "model_dump"):
        return item.model_dump(mode="python")
    return dict(vars(item))


def validate_selected_constraint_types_data(
    selected_constraint_types: list[Any],
) -> None:
    """selected_constraint_types の重複を禁止する。"""

    normalized = [str(item) for item in selected_constraint_types]
    if len(set(normalized)) != len(normalized):
        raise ValueError("selected_constraint_types must not contain duplicates")


def validate_formula_hazard_bijection_data(links: list[Any]) -> None:
    """formula-hazard 対応が 1 対 1 であることを検証する。"""

    normalized_links = [as_mapping(link) for link in links]
    formula_ids = [str(link["formula_id"]) for link in normalized_links]
    hazard_ids = [str(link["hazard_id"]) for link in normalized_links]
    if len(set(formula_ids)) != len(formula_ids):
        raise ValueError("formula_hazard_links must not reuse formula_id")
    if len(set(hazard_ids)) != len(hazard_ids):
        raise ValueError("formula_hazard_links must not reuse hazard_id")
    if len(formula_ids) != len(hazard_ids):
        raise ValueError("formula_hazard_links must have equal formula_id and hazard_id counts")


def validate_reference_hazard_sets(
    *,
    hazard_links: list[Any],
    hazard_manuals: list[Any],
    question_targets: list[Any],
    runtime_facts: list[Any],
) -> None:
    """hazard 関連配列の集合が一致することを検証する。"""

    link_hazards = {str(as_mapping(link)["hazard_id"]) for link in hazard_links}
    manual_hazards = {str(as_mapping(manual)["hazard_id"]) for manual in hazard_manuals}
    target_hazards = {str(as_mapping(entry)["hazard_id"]) for entry in question_targets}
    runtime_hazards = {str(as_mapping(fact)["hazard_id"]) for fact in runtime_facts}
    if not (link_hazards == manual_hazards == target_hazards == runtime_hazards):
        raise ValueError(
            "formula_hazard_links, hazard_manuals, question_targets, and runtime_facts "
            "must cover the same hazards"
        )


def validate_manual_constraint_coverage_data(
    *,
    selected_constraint_types: list[Any],
    hazard_manuals: list[Any],
    question_targets: list[Any],
    runtime_facts: list[Any],
) -> None:
    """selected_constraint_types と manual 群の対応を検証する。"""

    selected = {str(item) for item in selected_constraint_types}
    target_map = {
        str(as_mapping(entry)["hazard_id"]): list(as_mapping(entry)["targets"])
        for entry in question_targets
    }
    fact_map = {
        str(as_mapping(fact)["hazard_id"]): as_mapping(fact)
        for fact in runtime_facts
    }

    for manual_item in hazard_manuals:
        manual = as_mapping(manual_item)
        hazard_id = str(manual["hazard_id"])
        fact = fact_map.get(hazard_id)
        if fact is None:
            raise ValueError(f"runtime fact missing for {hazard_id}")
        fact_types = {
            str(as_mapping(value)["constraint_type"])
            for value in fact["constraint_values"]
        }
        if fact_types != selected:
            raise ValueError(
                f"{hazard_id} runtime fact must cover exactly selected_constraint_types"
            )
        if len(target_map.get(hazard_id, [])) != len(selected_constraint_types):
            raise ValueError(
                f"{hazard_id} question targets must match selected_constraint_types count"
            )
        procedure_ids = {
            str(as_mapping(procedure)["procedure_id"])
            for procedure in manual["procedures"]
        }
        if str(fact["expected_procedure_id"]) not in procedure_ids:
            raise ValueError(
                f"{hazard_id} expected_procedure_id must exist in procedures"
            )


def validate_difficulty_profile_data(
    *,
    difficulty: str,
    hazard_count: int,
    question_target_counts: list[int],
    selected_constraint_count: int,
    expected_hazard_count: int | None = None,
    expected_selected_constraint_count: int | None = None,
) -> None:
    """difficulty ごとの件数プロファイルを検証する。"""

    hazard_ranges = {
        "easy": (2, 3),
        "normal": (3, 4),
        "hard": (4, 5),
    }
    constraint_ranges = {
        "easy": (1, 2),
        "normal": (3, 4),
        "hard": (3, 4),
    }
    if difficulty not in hazard_ranges:
        raise ValueError(f"unsupported difficulty: {difficulty}")

    hazard_min, hazard_max = hazard_ranges[difficulty]
    constraint_min, constraint_max = constraint_ranges[difficulty]
    if expected_hazard_count is not None:
        if hazard_count != expected_hazard_count:
            raise ValueError(
                f"{difficulty} requires exactly {expected_hazard_count} hazards, "
                f"got {hazard_count}"
            )
    elif not (hazard_min <= hazard_count <= hazard_max):
        raise ValueError(
            f"{difficulty} requires {hazard_min}-{hazard_max} hazards, got {hazard_count}"
        )
    if expected_selected_constraint_count is not None:
        if selected_constraint_count != expected_selected_constraint_count:
            raise ValueError(
                f"{difficulty} requires exactly {expected_selected_constraint_count} "
                f"selected constraints, got {selected_constraint_count}"
            )
    elif not (constraint_min <= selected_constraint_count <= constraint_max):
        raise ValueError(
            f"{difficulty} requires {constraint_min}-{constraint_max} selected constraints, "
            f"got {selected_constraint_count}"
        )
    for target_count in question_target_counts:
        if expected_selected_constraint_count is not None:
            if target_count != expected_selected_constraint_count:
                raise ValueError(
                    f"{difficulty} requires exactly {expected_selected_constraint_count} "
                    f"question targets per hazard, got {target_count}"
                )
        elif not (constraint_min <= target_count <= constraint_max):
            raise ValueError(
                f"{difficulty} requires {constraint_min}-{constraint_max} question targets per "
                f"hazard, got {target_count}"
            )


def build_constraint_value_map(runtime_fact: Any) -> dict[str, dict[str, Any]]:
    """runtime fact から constraint_type ごとの値辞書を作る。"""

    fact = as_mapping(runtime_fact)
    value_map: dict[str, dict[str, Any]] = {}
    for value in fact.get("constraint_values", []):
        item = as_mapping(value)
        value_map[str(item["constraint_type"])] = item
    return value_map


def condition_matches_runtime(
    condition: Any,
    value_map: dict[str, dict[str, Any]],
) -> bool:
    """1 条件が runtime value と一致するかを返す。"""

    item = as_mapping(condition)
    constraint_type = str(item["constraint_type"])
    mode = str(item["mode"])
    value = value_map.get(constraint_type)
    if value is None:
        return False
    if mode == "irrelevant":
        return True
    if mode == "range":
        numeric_value = value.get("numeric_value")
        if numeric_value is None:
            return False
        min_numeric = item.get("min_numeric")
        max_numeric = item.get("max_numeric")
        if min_numeric is not None and numeric_value < int(min_numeric):
            return False
        if max_numeric is not None and numeric_value > int(max_numeric):
            return False
        return True
    bool_value = value.get("bool_value")
    expected_bool = item.get("expected_bool")
    return bool_value is not None and bool_value is bool(expected_bool)


def procedure_matches_runtime_data(
    procedure: Any,
    runtime_fact: Any,
) -> bool:
    """procedure が runtime fact に一致するかを返す。"""

    procedure_item = as_mapping(procedure)
    conditions = procedure_item.get("applicable_conditions", [])
    value_map = build_constraint_value_map(runtime_fact)
    return all(condition_matches_runtime(condition, value_map) for condition in conditions)
