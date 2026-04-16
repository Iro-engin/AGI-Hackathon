"""science 制約種別ごとの派生型定義。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator


class SciencePressureProcedureCondition(BaseModel):
    """`pressure_pa` 用の手順条件。"""

    model_config = ConfigDict(extra="forbid")

    constraint_type: Literal["pressure_pa"]
    mode: Literal["range", "irrelevant"]
    min_numeric: int | None = None
    max_numeric: int | None = None
    expected_bool: None = None

    @model_validator(mode="after")
    def validate_condition_shape(self) -> SciencePressureProcedureCondition:
        """pressure 条件の値の持ち方を検証する。"""

        if self.mode == "range":
            if self.min_numeric is None and self.max_numeric is None:
                raise ValueError("pressure_pa range mode requires min_numeric or max_numeric")
            return self
        if self.min_numeric is not None or self.max_numeric is not None:
            raise ValueError("pressure_pa irrelevant mode must not set numeric bounds")
        return self


class ScienceBooleanProcedureConditionBase(BaseModel):
    """bool 系制約で共通の手順条件。"""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["bool", "irrelevant"]
    min_numeric: None = None
    max_numeric: None = None
    expected_bool: bool | None = None

    @model_validator(mode="after")
    def validate_condition_shape(self) -> ScienceBooleanProcedureConditionBase:
        """bool 系条件の値の持ち方を検証する。"""

        if self.mode == "bool":
            if self.expected_bool is None:
                raise ValueError("bool mode requires expected_bool")
            return self
        if self.expected_bool is not None:
            raise ValueError("irrelevant mode must not set expected_bool")
        return self


class ScienceCatalystProcedureCondition(ScienceBooleanProcedureConditionBase):
    """`catalyst_present` 用の手順条件。"""

    constraint_type: Literal["catalyst_present"]


class ScienceFlammableProcedureCondition(ScienceBooleanProcedureConditionBase):
    """`flammable_nearby` 用の手順条件。"""

    constraint_type: Literal["flammable_nearby"]


class ScienceEscapeProcedureCondition(ScienceBooleanProcedureConditionBase):
    """`escape_easy` 用の手順条件。"""

    constraint_type: Literal["escape_easy"]


class ScienceWaterProcedureCondition(ScienceBooleanProcedureConditionBase):
    """`water_usable` 用の手順条件。"""

    constraint_type: Literal["water_usable"]


class SciencePressureConstraintValue(BaseModel):
    """`pressure_pa` 用の実測値。"""

    model_config = ConfigDict(extra="forbid")

    constraint_type: Literal["pressure_pa"]
    numeric_value: int
    bool_value: None = None


class ScienceBooleanConstraintValueBase(BaseModel):
    """bool 系制約で共通の実測値。"""

    model_config = ConfigDict(extra="forbid")

    numeric_value: None = None
    bool_value: bool


class ScienceCatalystConstraintValue(ScienceBooleanConstraintValueBase):
    """`catalyst_present` 用の実測値。"""

    constraint_type: Literal["catalyst_present"]


class ScienceFlammableConstraintValue(ScienceBooleanConstraintValueBase):
    """`flammable_nearby` 用の実測値。"""

    constraint_type: Literal["flammable_nearby"]


class ScienceEscapeConstraintValue(ScienceBooleanConstraintValueBase):
    """`escape_easy` 用の実測値。"""

    constraint_type: Literal["escape_easy"]


class ScienceWaterConstraintValue(ScienceBooleanConstraintValueBase):
    """`water_usable` 用の実測値。"""

    constraint_type: Literal["water_usable"]


def _as_dict(item: object) -> dict[str, object]:
    """Pydantic model か dict を普通の dict にそろえる。"""

    if isinstance(item, dict):
        return item
    if hasattr(item, "model_dump"):
        return item.model_dump(mode="python")
    return dict(vars(item))


def validate_procedure_condition_variant(item: object) -> None:
    """制約種別に応じた procedure condition の形を検証する。"""

    data = _as_dict(item)
    constraint_type = str(data.get("constraint_type"))
    if constraint_type == "pressure_pa":
        SciencePressureProcedureCondition.model_validate(data)
        return
    mapping = {
        "catalyst_present": ScienceCatalystProcedureCondition,
        "flammable_nearby": ScienceFlammableProcedureCondition,
        "escape_easy": ScienceEscapeProcedureCondition,
        "water_usable": ScienceWaterProcedureCondition,
    }
    model = mapping.get(constraint_type)
    if model is None:
        raise ValueError(f"unsupported constraint_type: {constraint_type}")
    model.model_validate(data)


def validate_constraint_value_variant(item: object) -> None:
    """制約種別に応じた runtime value の形を検証する。"""

    data = _as_dict(item)
    constraint_type = str(data.get("constraint_type"))
    if constraint_type == "pressure_pa":
        SciencePressureConstraintValue.model_validate(data)
        return
    mapping = {
        "catalyst_present": ScienceCatalystConstraintValue,
        "flammable_nearby": ScienceFlammableConstraintValue,
        "escape_easy": ScienceEscapeConstraintValue,
        "water_usable": ScienceWaterConstraintValue,
    }
    model = mapping.get(constraint_type)
    if model is None:
        raise ValueError(f"unsupported constraint_type: {constraint_type}")
    model.model_validate(data)
