"""science 型定義の難易度別派生クラス。"""

from __future__ import annotations

import random
from dataclasses import dataclass
from functools import cache

from pydantic import Field
from pydantic.main import create_model

from src.science.type.base import (
    ScienceCaseDraft,
    ScienceCaseSeedDraft,
    ScienceConstraintType,
    ScienceFormulaHazardLink,
    ScienceHazardManual,
    ScienceHazardProcedure,
    ScienceHazardQuestionTargets,
    ScienceHazardRuntimeFact,
    ScienceManualBundleDraft,
    ScienceProcedureCondition,
    ScienceReferenceData,
    ScienceReferenceDataSeedBase,
)

COUNT_RANGES = {
    "easy": {
        "hazard_count": (2, 3),
        "selected_constraint_count": (1, 2),
    },
    "normal": {
        "hazard_count": (3, 4),
        "selected_constraint_count": (3, 4),
    },
    "hard": {
        "hazard_count": (4, 5),
        "selected_constraint_count": (3, 4),
    },
}


@dataclass(frozen=True)
class ScienceCountProfile:
    """1 ケース内で固定する件数プロファイル。"""

    difficulty: str
    hazard_count: int
    selected_constraint_count: int
    question_target_count: int


def choose_count_profile(difficulty: str, *, key: str) -> ScienceCountProfile:
    """難易度ごとの範囲から、ケース専用の固定件数を 1 回だけ選ぶ。"""

    if difficulty not in COUNT_RANGES:
        raise ValueError(f"unsupported difficulty: {difficulty}")

    rng = random.Random(key)
    range_config = COUNT_RANGES[difficulty]
    selected_constraint_count = rng.randint(*range_config["selected_constraint_count"])
    return ScienceCountProfile(
        difficulty=difficulty,
        hazard_count=rng.randint(*range_config["hazard_count"]),
        selected_constraint_count=selected_constraint_count,
        question_target_count=selected_constraint_count,
    )


@cache
def build_exact_generation_classes(
    difficulty: str,
    hazard_count: int,
    selected_constraint_count: int,
) -> tuple[type[ScienceCaseSeedDraft], type[ScienceManualBundleDraft], type[ScienceCaseDraft]]:
    """固定件数プロファイルに一致する生成専用 schema 群を返す。"""

    if difficulty not in COUNT_RANGES:
        raise ValueError(f"unsupported difficulty: {difficulty}")

    question_target_count = selected_constraint_count
    class_suffix = (
        f"{difficulty.title()}H{hazard_count}C{selected_constraint_count}Q{question_target_count}"
    )

    question_targets_class = create_model(
        f"{class_suffix}ScienceHazardQuestionTargets",
        __base__=ScienceHazardQuestionTargets,
        __module__=__name__,
        targets=(
            list[str],
            Field(
                min_length=question_target_count,
                max_length=question_target_count,
            ),
        ),
    )

    # applicable_conditions は selected_constraint_count と完全一致させる。
    # 各手順はすべての制約タイプを網羅する必要があるため min=max=N とする。
    procedure_class = create_model(
        f"{class_suffix}ScienceHazardProcedure",
        __base__=ScienceHazardProcedure,
        __module__=__name__,
        applicable_conditions=(
            list[ScienceProcedureCondition],
            Field(
                min_length=selected_constraint_count,
                max_length=selected_constraint_count,
            ),
        ),
    )

    manual_class = create_model(
        f"{class_suffix}ScienceHazardManual",
        __base__=ScienceHazardManual,
        __module__=__name__,
        procedures=(
            list[procedure_class],
            Field(min_length=2, max_length=10),
        ),
    )

    reference_seed_class = create_model(
        f"{class_suffix}ScienceReferenceDataSeed",
        __base__=ScienceReferenceDataSeedBase,
        __module__=__name__,
        formula_hazard_links=(
            list[ScienceFormulaHazardLink],
            Field(min_length=hazard_count, max_length=hazard_count),
        ),
        selected_constraint_types=(
            list[ScienceConstraintType],
            Field(
                min_length=selected_constraint_count,
                max_length=selected_constraint_count,
            ),
        ),
    )

    reference_class = create_model(
        f"{class_suffix}ScienceReferenceData",
        __base__=ScienceReferenceData,
        __module__=__name__,
        formula_hazard_links=(
            list[ScienceFormulaHazardLink],
            Field(min_length=hazard_count, max_length=hazard_count),
        ),
        selected_constraint_types=(
            list[ScienceConstraintType],
            Field(
                min_length=selected_constraint_count,
                max_length=selected_constraint_count,
            ),
        ),
        hazard_manuals=(
            list[manual_class],
            Field(min_length=hazard_count, max_length=hazard_count),
        ),
        hazard_question_targets=(
            list[question_targets_class],
            Field(min_length=hazard_count, max_length=hazard_count),
        ),
        hidden_hazard_facts=(
            list[ScienceHazardRuntimeFact],
            Field(min_length=hazard_count, max_length=hazard_count),
        ),
    )

    seed_class = create_model(
        f"{class_suffix}ScienceCaseSeedDraft",
        __base__=ScienceCaseSeedDraft,
        __module__=__name__,
        reference_data=(reference_seed_class, ...),
    )

    bundle_class = create_model(
        f"{class_suffix}ScienceManualBundleDraft",
        __base__=ScienceManualBundleDraft,
        __module__=__name__,
        hazard_manuals=(
            list[manual_class],
            Field(min_length=hazard_count, max_length=hazard_count),
        ),
        hazard_question_targets=(
            list[question_targets_class],
            Field(min_length=hazard_count, max_length=hazard_count),
        ),
        hidden_hazard_facts=(
            list[ScienceHazardRuntimeFact],
            Field(min_length=hazard_count, max_length=hazard_count),
        ),
    )

    draft_class = create_model(
        f"{class_suffix}ScienceCaseDraft",
        __base__=ScienceCaseDraft,
        __module__=__name__,
        reference_data=(reference_class, ...),
    )

    return seed_class, bundle_class, draft_class


class EasyScienceHazardQuestionTargets(ScienceHazardQuestionTargets):
    """easy 用の質問ターゲット。"""

    targets: list[str] = Field(min_length=1, max_length=2)


class NormalScienceHazardQuestionTargets(ScienceHazardQuestionTargets):
    """normal 用の質問ターゲット。"""

    targets: list[str] = Field(min_length=3, max_length=4)


class HardScienceHazardQuestionTargets(ScienceHazardQuestionTargets):
    """hard 用の質問ターゲット。"""

    targets: list[str] = Field(min_length=3, max_length=4)


class EasyScienceReferenceDataSeed(ScienceReferenceDataSeedBase):
    """easy 用の seed reference_data。"""

    formula_hazard_links: list[ScienceFormulaHazardLink] = Field(min_length=2, max_length=3)
    selected_constraint_types: list[ScienceConstraintType] = Field(min_length=1, max_length=2)


class NormalScienceReferenceDataSeed(ScienceReferenceDataSeedBase):
    """normal 用の seed reference_data。"""

    formula_hazard_links: list[ScienceFormulaHazardLink] = Field(min_length=3, max_length=4)
    selected_constraint_types: list[ScienceConstraintType] = Field(min_length=3, max_length=4)


class HardScienceReferenceDataSeed(ScienceReferenceDataSeedBase):
    """hard 用の seed reference_data。"""

    formula_hazard_links: list[ScienceFormulaHazardLink] = Field(min_length=4, max_length=5)
    selected_constraint_types: list[ScienceConstraintType] = Field(min_length=3, max_length=4)


class EasyScienceReferenceData(ScienceReferenceData):
    """easy 用の最終 reference_data。"""

    formula_hazard_links: list[ScienceFormulaHazardLink] = Field(min_length=2, max_length=3)
    selected_constraint_types: list[ScienceConstraintType] = Field(min_length=1, max_length=2)
    hazard_manuals: list[ScienceHazardManual] = Field(min_length=2, max_length=3)
    hazard_question_targets: list[EasyScienceHazardQuestionTargets] = Field(
        min_length=2,
        max_length=3,
    )
    hidden_hazard_facts: list[ScienceHazardRuntimeFact] = Field(min_length=2, max_length=3)


class NormalScienceReferenceData(ScienceReferenceData):
    """normal 用の最終 reference_data。"""

    formula_hazard_links: list[ScienceFormulaHazardLink] = Field(min_length=3, max_length=4)
    selected_constraint_types: list[ScienceConstraintType] = Field(min_length=3, max_length=4)
    hazard_manuals: list[ScienceHazardManual] = Field(min_length=3, max_length=4)
    hazard_question_targets: list[NormalScienceHazardQuestionTargets] = Field(
        min_length=3,
        max_length=4,
    )
    hidden_hazard_facts: list[ScienceHazardRuntimeFact] = Field(min_length=3, max_length=4)


class HardScienceReferenceData(ScienceReferenceData):
    """hard 用の最終 reference_data。"""

    formula_hazard_links: list[ScienceFormulaHazardLink] = Field(min_length=4, max_length=5)
    selected_constraint_types: list[ScienceConstraintType] = Field(min_length=3, max_length=4)
    hazard_manuals: list[ScienceHazardManual] = Field(min_length=4, max_length=5)
    hazard_question_targets: list[HardScienceHazardQuestionTargets] = Field(
        min_length=4,
        max_length=5,
    )
    hidden_hazard_facts: list[ScienceHazardRuntimeFact] = Field(min_length=4, max_length=5)


class EasyScienceCaseSeedDraft(ScienceCaseSeedDraft):
    """easy 用のケース骨子。"""

    reference_data: EasyScienceReferenceDataSeed


class NormalScienceCaseSeedDraft(ScienceCaseSeedDraft):
    """normal 用のケース骨子。"""

    reference_data: NormalScienceReferenceDataSeed


class HardScienceCaseSeedDraft(ScienceCaseSeedDraft):
    """hard 用のケース骨子。"""

    reference_data: HardScienceReferenceDataSeed


class EasyScienceManualBundleDraft(ScienceManualBundleDraft):
    """easy 用の manual bundle。"""

    hazard_manuals: list[ScienceHazardManual] = Field(min_length=2, max_length=3)
    hazard_question_targets: list[EasyScienceHazardQuestionTargets] = Field(
        min_length=2,
        max_length=3,
    )
    hidden_hazard_facts: list[ScienceHazardRuntimeFact] = Field(min_length=2, max_length=3)


class NormalScienceManualBundleDraft(ScienceManualBundleDraft):
    """normal 用の manual bundle。"""

    hazard_manuals: list[ScienceHazardManual] = Field(min_length=3, max_length=4)
    hazard_question_targets: list[NormalScienceHazardQuestionTargets] = Field(
        min_length=3,
        max_length=4,
    )
    hidden_hazard_facts: list[ScienceHazardRuntimeFact] = Field(min_length=3, max_length=4)


class HardScienceManualBundleDraft(ScienceManualBundleDraft):
    """hard 用の manual bundle。"""

    hazard_manuals: list[ScienceHazardManual] = Field(min_length=4, max_length=5)
    hazard_question_targets: list[HardScienceHazardQuestionTargets] = Field(
        min_length=4,
        max_length=5,
    )
    hidden_hazard_facts: list[ScienceHazardRuntimeFact] = Field(min_length=4, max_length=5)


class EasyScienceCaseDraft(ScienceCaseDraft):
    """easy 用の最終 draft。"""

    reference_data: EasyScienceReferenceData


class NormalScienceCaseDraft(ScienceCaseDraft):
    """normal 用の最終 draft。"""

    reference_data: NormalScienceReferenceData


class HardScienceCaseDraft(ScienceCaseDraft):
    """hard 用の最終 draft。"""

    reference_data: HardScienceReferenceData


DIFFICULTY_SEED_CLASS = {
    "easy": EasyScienceCaseSeedDraft,
    "normal": NormalScienceCaseSeedDraft,
    "hard": HardScienceCaseSeedDraft,
}

DIFFICULTY_BUNDLE_CLASS = {
    "easy": EasyScienceManualBundleDraft,
    "normal": NormalScienceManualBundleDraft,
    "hard": HardScienceManualBundleDraft,
}

DIFFICULTY_DRAFT_CLASS = {
    "easy": EasyScienceCaseDraft,
    "normal": NormalScienceCaseDraft,
    "hard": HardScienceCaseDraft,
}
