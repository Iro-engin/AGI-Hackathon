"""science 型定義の共通部分。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from src.science.type.constraint_variants import (
    validate_constraint_value_variant,
    validate_procedure_condition_variant,
)
from src.science.type.validators import (
    procedure_matches_runtime_data,
    validate_difficulty_profile_data,
    validate_formula_hazard_bijection_data,
    validate_manual_constraint_coverage_data,
    validate_reference_hazard_sets,
    validate_selected_constraint_types_data,
)


@dataclass(frozen=True)
class ScienceBlueprint:
    """ケース生成時に使うテーマ雛形。"""

    slug: str
    label: str
    scenario_hook: str
    formula_examples: tuple[str, ...]
    hazard_examples: tuple[str, ...]
    tool_examples: tuple[str, ...]
    hard_constraints: tuple[str, ...]


ScienceConstraintType = Literal[
    "pressure_pa",
    "catalyst_present",
    "flammable_nearby",
    "escape_easy",
    "water_usable",
]

ScienceConditionMode = Literal["range", "bool", "irrelevant"]


class ScienceProcedureCondition(BaseModel):
    """OpenAI schema 互換の汎用 procedure condition。"""

    model_config = ConfigDict(extra="forbid")

    constraint_type: ScienceConstraintType
    mode: ScienceConditionMode
    min_numeric: int | None = None
    max_numeric: int | None = None
    expected_bool: bool | None = None

    @model_validator(mode="after")
    def validate_condition_shape(self) -> ScienceProcedureCondition:
        """制約種別ごとの厳密な shape を別定義で検証する。"""

        validate_procedure_condition_variant(self)
        return self


class ScienceHazardProcedure(BaseModel):
    """危険発生時に選択できる対処手順。"""

    model_config = ConfigDict(extra="forbid")

    procedure_id: str
    applicable_conditions: list[ScienceProcedureCondition] = Field(min_length=1, max_length=5)
    required_tools: list[str] = Field(default_factory=list)
    stop_source_formula: bool = True
    hazard_potential:float = Field(ge=0.0, le=100.0, default=50.0)

    @model_validator(mode="after")
    def validate_unique_condition_types(self) -> ScienceHazardProcedure:
        """1 手順内で同じ constraint_type を重複させない。"""

        condition_types = [condition.constraint_type for condition in self.applicable_conditions]
        if len(set(condition_types)) != len(condition_types):
            raise ValueError("applicable_conditions must not reuse constraint_type")
        return self


class ScienceHazardManual(BaseModel):
    """危険ごとの対処マニュアル。"""

    model_config = ConfigDict(extra="forbid")

    hazard_id: str
    source_formula: str
    procedures: list[ScienceHazardProcedure] = Field(min_length=2, max_length=10)

    @model_validator(mode="after")
    def validate_unique_procedure_ids(self) -> ScienceHazardManual:
        """procedure_id の重複を禁止する。"""

        procedure_ids = [procedure.procedure_id for procedure in self.procedures]
        if len(set(procedure_ids)) != len(procedure_ids):
            raise ValueError("procedures must not reuse procedure_id")
        return self


class ScienceFormulaHazardLink(BaseModel):
    """化学式と危険の対応 1 件を表す。"""

    model_config = ConfigDict(extra="forbid")

    formula_id: str
    hazard_id: str


class ScienceHazardQuestionTargets(BaseModel):
    """危険ごとの質問チェックリスト。"""

    model_config = ConfigDict(extra="forbid")

    hazard_id: str
    targets: list[str] = Field(min_length=1, max_length=4)


class ScienceConstraintValue(BaseModel):
    """OpenAI schema 互換の汎用 runtime value。"""

    model_config = ConfigDict(extra="forbid")

    constraint_type: ScienceConstraintType
    numeric_value: int | None = None
    bool_value: bool | None = None

    @model_validator(mode="after")
    def validate_value_shape(self) -> ScienceConstraintValue:
        """制約種別ごとの厳密な shape を別定義で検証する。"""

        validate_constraint_value_variant(self)
        return self


class ScienceHazardRuntimeFact(BaseModel):
    """質問回答時にだけ参照される隠し事実。"""

    model_config = ConfigDict(extra="forbid")

    hazard_id: str
    constraint_values: list[ScienceConstraintValue] = Field(min_length=1, max_length=5)
    expected_procedure_id: str

    @model_validator(mode="after")
    def validate_unique_constraint_types(self) -> ScienceHazardRuntimeFact:
        """constraint_values 内で同じ種類を重複させない。"""

        constraint_types = [value.constraint_type for value in self.constraint_values]
        if len(set(constraint_types)) != len(constraint_types):
            raise ValueError("constraint_values must not reuse constraint_type")
        return self


class ScienceReferenceDataSeedBase(BaseModel):
    """Phase 1 生成用の背景情報。"""

    model_config = ConfigDict(extra="forbid")

    running_formulas: list[str] = Field(min_length=3)
    formula_hazard_links: list[ScienceFormulaHazardLink] = Field(min_length=2, max_length=5)
    active_hazard_ids: list[str] = Field(min_length=1, max_length=1)
    resolved_hazard_ids: list[str] = Field(default_factory=list)
    initial_tool_constraints: list[str] = Field(min_length=1)
    selected_constraint_types: list[ScienceConstraintType] = Field(min_length=1, max_length=4)

    @field_validator("formula_hazard_links")
    @classmethod
    def validate_formula_hazard_bijection_field(
        cls,
        value: list[ScienceFormulaHazardLink],
    ) -> list[ScienceFormulaHazardLink]:
        """formula-hazard の全単射をフィールド段階で見る。"""

        validate_formula_hazard_bijection_data(value)
        return value

    @field_validator("selected_constraint_types")
    @classmethod
    def validate_selected_constraint_types_field(
        cls,
        value: list[ScienceConstraintType],
    ) -> list[ScienceConstraintType]:
        """constraint type の重複を禁止する。"""

        validate_selected_constraint_types_data(value)
        return value

    @model_validator(mode="after")
    def validate_seed_consistency(self) -> ScienceReferenceDataSeedBase:
        """seed 側の hazard 集合整合性を検証する。"""

        hazard_ids = {link.hazard_id for link in self.formula_hazard_links}
        if set(self.active_hazard_ids) - hazard_ids:
            raise ValueError("active_hazard_ids must be included in formula_hazard_links")
        if set(self.resolved_hazard_ids) - hazard_ids:
            raise ValueError("resolved_hazard_ids must be included in formula_hazard_links")
        if set(self.active_hazard_ids) & set(self.resolved_hazard_ids):
            raise ValueError("active_hazard_ids and resolved_hazard_ids must not overlap")
        return self


class ScienceReferenceData(ScienceReferenceDataSeedBase):
    """最終ケースで使う背景情報全体。"""

    hazard_manuals: list[ScienceHazardManual] = Field(min_length=2, max_length=5)
    hazard_question_targets: list[ScienceHazardQuestionTargets] = Field(min_length=2, max_length=5)
    hidden_hazard_facts: list[ScienceHazardRuntimeFact] = Field(min_length=2, max_length=5)

    @model_validator(mode="after")
    def validate_reference_consistency(self) -> ScienceReferenceData:
        """manual・question・runtime fact の整合性を検証する。"""

        validate_reference_hazard_sets(
            hazard_links=self.formula_hazard_links,
            hazard_manuals=self.hazard_manuals,
            question_targets=self.hazard_question_targets,
            runtime_facts=self.hidden_hazard_facts,
        )
        validate_manual_constraint_coverage_data(
            selected_constraint_types=self.selected_constraint_types,
            hazard_manuals=self.hazard_manuals,
            question_targets=self.hazard_question_targets,
            runtime_facts=self.hidden_hazard_facts,
        )
        manual_map = {manual.hazard_id: manual for manual in self.hazard_manuals}
        for runtime_fact in self.hidden_hazard_facts:
            manual = manual_map[runtime_fact.hazard_id]
            matches = [
                procedure.procedure_id
                for procedure in manual.procedures
                if procedure_matches_runtime_data(procedure, runtime_fact)
            ]
            if runtime_fact.expected_procedure_id not in matches:
                raise ValueError(
                    f"{runtime_fact.hazard_id} expected_procedure_id must match runtime values"
                )
            if len(matches) == 0:
                raise ValueError(
                    f"{runtime_fact.hazard_id} runtime values must match at least one procedure applicable_conditions"
                )
        return self


class ScienceReferenceDataDelta(BaseModel):
    """event 発生時に反映する reference_data 差分。"""

    model_config = ConfigDict(extra="forbid")

    active_hazard_ids: list[str] | None = None
    resolved_hazard_ids: list[str] | None = None
    initial_tool_constraints: list[str] | None = None


class ScienceEventDelta(BaseModel):
    """event で state に反映する差分。"""

    model_config = ConfigDict(extra="forbid")

    reference_data: ScienceReferenceDataDelta | None = None


class ScienceBenchmarkEventDraft(BaseModel):
    """ケース内で発生する event 定義。"""

    model_config = ConfigDict(extra="forbid")

    turn: int
    type: Literal["state_change", "new_constraint"]
    message: str
    delta: ScienceEventDelta
    expected_replan_within_turns: int = 5

    @field_validator("turn")
    @classmethod
    def validate_turn(cls, value: int) -> int:
        """turn は 1 以上のみ許可する。"""

        if value < 1:
            raise ValueError("turn must be >= 1")
        return value

    @field_validator("type", mode="before")
    @classmethod
    def normalize_type(cls, value: str) -> str:
        """event type を正規化し、許可値だけに絞る。"""

        normalized = value.strip().lower()
        if not normalized:
            raise ValueError("event type must be non-empty")
        aliases = {
            "constraint_change": "new_constraint",
            "constraint_update": "new_constraint",
            "state_update": "state_change",
        }
        return aliases.get(normalized, normalized)


class ScienceCaseSeedDraft(BaseModel):
    """Phase 1 生成で受け取るケース骨子。"""

    model_config = ConfigDict(extra="forbid")

    initial_request: str
    reference_data: ScienceReferenceDataSeedBase
    events: list[ScienceBenchmarkEventDraft] = Field(min_length=3, max_length=3)


class ScienceManualBundleDraft(BaseModel):
    """Phase 2 生成で受け取る manual 群。"""

    model_config = ConfigDict(extra="forbid")

    hazard_manuals: list[ScienceHazardManual] = Field(min_length=2, max_length=5)
    hazard_question_targets: list[ScienceHazardQuestionTargets] = Field(min_length=2, max_length=5)
    hidden_hazard_facts: list[ScienceHazardRuntimeFact] = Field(min_length=2, max_length=5)


class ScienceCaseDraft(BaseModel):
    """最終的に組み立てるケース下書き。"""

    model_config = ConfigDict(extra="forbid")

    initial_request: str
    reference_data: ScienceReferenceData
    events: list[ScienceBenchmarkEventDraft] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def validate_difficulty_profile(self, info: ValidationInfo) -> ScienceCaseDraft:
        """difficulty context がある場合だけ件数プロファイルを検証する。"""

        difficulty = None
        expected_hazard_count = None
        expected_selected_constraint_count = None
        if isinstance(info.context, dict):
            difficulty = info.context.get("difficulty")
            count_profile = info.context.get("count_profile")
            if count_profile is not None:
                expected_hazard_count = getattr(count_profile, "hazard_count", None)
                expected_selected_constraint_count = getattr(
                    count_profile,
                    "selected_constraint_count",
                    None,
                )
        if not difficulty:
            return self

        validate_difficulty_profile_data(
            difficulty=str(difficulty),
            hazard_count=len(self.reference_data.formula_hazard_links),
            question_target_counts=[
                len(entry.targets) for entry in self.reference_data.hazard_question_targets
            ],
            selected_constraint_count=len(self.reference_data.selected_constraint_types),
            expected_hazard_count=expected_hazard_count,
            expected_selected_constraint_count=expected_selected_constraint_count,
        )
        return self
