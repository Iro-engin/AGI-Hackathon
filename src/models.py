"""ベンチマークケースと実行ログの Pydantic モデル定義。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SemanticCheck(BaseModel):
    """成果物の項目と、反映すべき state の値を対応付ける。"""

    artifact_field: str
    state_path: str


class RequiredArtifact(BaseModel):
    """タスク完了時点でエージェントが出力すべき成果物を定義する。"""

    artifact_id: str
    artifact_type: str
    required_fields: list[str]
    semantic_checks: list[SemanticCheck] = Field(default_factory=list)
    latest_version_required: bool = True


class TaskDependency(BaseModel):
    """論理タスク間の実行順序制約を表す。"""

    before: str
    after: str
    reason: str | None = None


class InitialState(BaseModel):
    """エージェントが推論を始める初期状態を表す。"""

    model_config = ConfigDict(extra="allow")

    timezone: str
    deadline: str
    participants: list[str]
    budget: int | float
    required_artifacts: list[RequiredArtifact]
    constraints: list[str]
    task_dependencies: list[TaskDependency] = Field(default_factory=list)
    reference_data: dict[str, Any] = Field(default_factory=dict)


class BenchmarkEvent(BaseModel):
    """再計画を促す動的な状態変化イベントを表す。"""

    model_config = ConfigDict(extra="forbid")

    turn: int
    type: str
    message: str
    delta: dict[str, Any]
    expected_replan_within_turns: int = 0
    expected_artifact_updates: list[str] = Field(default_factory=list)
    expected_tasks: list[str] = Field(default_factory=list)


class GoalCondition(BaseModel):
    """成功条件となる最終状態とプロセス要件を表す。"""

    model_config = ConfigDict(extra="forbid")

    must_satisfy_latest_state: bool
    required_artifacts: list[str]
    no_constraint_violation: bool
    must_acknowledge_changes: bool = True
    must_use_ask_clarification_on_ambiguity: bool = False
    must_not_finalize_with_unresolved_scope: bool = False


class Rubric(BaseModel):
    """ケースが何を評価するかを説明するルーブリック。"""

    model_config = ConfigDict(extra="forbid")

    outcome: list[str]
    process: list[str]
    recovery: list[str]


class Case(BaseModel):
    """evaluator が受け取る case のトップレベル定義。"""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    domain: str
    difficulty: str
    initial_request: str
    initial_state: InitialState
    allowed_actions: list[str]
    events: list[BenchmarkEvent]
    goal_condition: GoalCondition
    rubric: Rubric
    notes: str | None = None

    @classmethod
    def from_path(cls, path: Path) -> Case:
        """JSON ファイルから case を読み込み、検証する。"""

        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def to_summary(self) -> CaseSummary:
        """case の概要だけを抜き出した読みやすい要約を返す。"""

        return CaseSummary(
            task_id=self.task_id,
            difficulty=self.difficulty,
            event_count=len(self.events),
            required_artifacts=[
                artifact.artifact_id for artifact in self.initial_state.required_artifacts
            ],
            dependency_count=len(self.initial_state.task_dependencies),
            constraint_count=len(self.initial_state.constraints),
        )


BenchmarkCase = Case


class ArtifactSnapshot(BaseModel):
    """最終的に提出された 1 成果物の状態を表す。"""

    model_config = ConfigDict(extra="forbid")

    fields_completed: list[str] = Field(default_factory=list)
    field_values: dict[str, Any] = Field(default_factory=dict)


class ClarificationQuestion(BaseModel):
    """エージェントが行った確認質問の構造化記録。"""

    model_config = ConfigDict(extra="forbid")

    turn: int
    question: str
    reason: str | None = None
    blocks_tasks: list[str] = Field(default_factory=list)


class TaskBreakdownItem(BaseModel):
    """エージェントの作業計画における 1 つの分解タスク。"""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    description: str
    depends_on: list[str] = Field(default_factory=list)
    status: str = "pending"


class ActionLog(BaseModel):
    """エージェントの 1 行動を表す構造化ログ。"""

    model_config = ConfigDict(extra="forbid")

    turn: int
    action_type: str
    acknowledged_event_turns: list[int] = Field(default_factory=list)
    artifact_updates: list[str] = Field(default_factory=list)
    notes: str | None = None


class ExecutionLog(BaseModel):
    """evaluator 入力として使う execution log のトップレベル定義。"""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    actions: list[ActionLog]
    completed_tasks: list[str]
    task_breakdown: list[TaskBreakdownItem] = Field(default_factory=list)
    questions_asked: list[ClarificationQuestion] = Field(default_factory=list)
    constraint_violations: list[str] = Field(default_factory=list)
    unsafe_commit: bool = False
    final_state: dict[str, Any] = Field(default_factory=dict)
    final_artifacts: dict[str, ArtifactSnapshot] = Field(default_factory=dict)

    @classmethod
    def from_path(cls, path: Path) -> ExecutionLog:
        """JSON ファイルから execution log を読み込み、検証する。"""

        return cls.model_validate_json(path.read_text(encoding="utf-8"))


class JsonFileRepository:
    """検証付き JSON 読み込みをまとめる小さなリポジトリヘルパー。"""

    def __init__(self, root: Path) -> None:
        self.root = root

    def load_cases(self, relative_dir: str) -> list[Case]:
        """ディレクトリ配下の case をファイル名順に読み込む。"""

        case_dir = self.root / relative_dir
        return [Case.from_path(path) for path in sorted(case_dir.glob("*.json"))]

    def load_execution_log(self, relative_path: str) -> ExecutionLog:
        """リポジトリルートからの相対パスで execution log を 1 件読み込む。"""

        return ExecutionLog.from_path(self.root / relative_path)


@dataclass(frozen=True)
class CaseSummary:
    """case 一覧を俯瞰するための簡易サマリー。"""

    task_id: str
    difficulty: str
    event_count: int
    required_artifacts: list[str]
    dependency_count: int
    constraint_count: int


class CaseCatalog:
    """複数 case を見やすく扱うためのカタログ。"""

    def __init__(self, cases: list[Case]) -> None:
        self.cases = cases

    @classmethod
    def from_repository(cls, repository: JsonFileRepository, relative_dir: str) -> CaseCatalog:
        """リポジトリから case 群を読み込んでカタログ化する。"""

        return cls(repository.load_cases(relative_dir))

    def summaries(self) -> list[CaseSummary]:
        """各 case の要約一覧を返す。"""

        return [case.to_summary() for case in self.cases]


def dump_json(data: Any) -> str:
    """notebook 埋め込み向けに安定した形式で JSON 文字列化する。"""

    return json.dumps(data, ensure_ascii=False, indent=2)
