"""inaba ドメイン共通の評価基盤：TurnContext / TurnRecord / EvalResult / DomainAdapter。"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from ..models.base import Case, Do

load_dotenv()

DEFAULT_JUDGE_MODEL = "gpt-4.1-mini"

_JUDGE_INSTRUCTIONS = (
    "You are a strict binary judge. "
    "Answer only with 'True' or 'False', no explanation."
)


# ---------------------------------------------------------------------------
# Agent への入力コンテキスト（prompt 構築用）
# ---------------------------------------------------------------------------


class QuestionAnswer(BaseModel):
    """1 ターン内の質問と回答のペア。"""

    question: str
    answer: str


@dataclass(frozen=True)
class TurnContext:
    """agent に見せる 1 ターン分のコンテキスト。"""

    initial_got_info: dict[str, Any]
    stage: str
    current_constraint: list[str]
    event_message: str | None
    current_got_info: dict[str, Any]
    should_require: list[str]
    question_answers: list[QuestionAnswer]
    previous_do: dict[str, Any] | None
    current_hidden_info: dict[str, Any]


# ---------------------------------------------------------------------------
# 実行記録（採点用）
# ---------------------------------------------------------------------------


@dataclass
class TurnRecord:
    """1 イベントターンの実行記録（採点用）。

    Attributes:
        event_index:              ケース内のイベント番号（0 始まり）。
        should_require:           正解の確認項目リスト。
        questions_asked:          agent が聞いた質問（発話順）。
        covered_per_question:     questions_asked[i] がカバーした should_require 項目のリスト。
                                  len は questions_asked と同一。
        covered_requires:         covered_per_question の和集合（順序なし）。
        redundant_questions:      got_info に既にあった情報を重ねて聞いた質問。
        did_without_full_coverage: should_require 未完了のまま Do を返したか。
        final_do:                 agent が提出した最終 Do。
        exp_do:                   正解の Do。
        do_is_feasible:           最終 Do が全制約を満たし実行可能かどうか。
        do_exact_match:           最終 Do が正解と完全一致するかどうか。
    """

    event_index: int
    should_require: list[str]
    questions_asked: list[str]
    covered_per_question: list[list[str]]  # questions_asked と同じ長さ
    covered_requires: list[str]
    redundant_questions: list[str]
    did_without_full_coverage: bool
    final_do: Do | None
    exp_do: Do | None
    do_is_feasible: bool
    do_exact_match: bool


# ---------------------------------------------------------------------------
# 評価指標
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    """評価指標の集計結果。

    Planning Score:
        final_pass_rate:  [1] 全制約を満たし実行可能な最終プランの割合。
        exact_match:      [2] 決定要素が正解と完全一致する割合。

    Impulse Control Score:
        vague_ask_rate:          [3] 聞くべき項目が残っていたのに未質問で Do した割合。
        rubric_score:            [3] Do 前に必要質問をカバーできた平均割合。
        redundant_question_rate: [3] 不要な冗長質問の割合。
        early_discovery_rate:    [3] 最初の N 手以内で全必要質問を完了した割合。
    """

    # Planning Score
    final_pass_rate: float
    exact_match: float

    # Impulse Control Score
    vague_ask_rate: float
    rubric_score: float
    redundant_question_rate: float
    early_discovery_rate: float

    total_events: int
    early_discovery_n: int


def compute_eval_result(
    records: list[TurnRecord],
    *,
    early_discovery_n: int = 2,
) -> EvalResult:
    """TurnRecord のリストから全評価指標を計算する。"""

    total = len(records)
    if total == 0:
        return EvalResult(
            final_pass_rate=0.0,
            exact_match=0.0,
            vague_ask_rate=0.0,
            rubric_score=0.0,
            redundant_question_rate=0.0,
            early_discovery_rate=0.0,
            total_events=0,
            early_discovery_n=early_discovery_n,
        )

    # ── Planning Score ──────────────────────────────────────────────────────

    # [1] Final Pass Rate
    final_pass_rate = sum(1 for r in records if r.do_is_feasible) / total

    # [2] Exact Match
    exact_match = sum(1 for r in records if r.do_exact_match) / total

    # ── Impulse Control Score ───────────────────────────────────────────────

    events_with_requires = [r for r in records if r.should_require]

    # [3-a] Vague Ask Rate
    if events_with_requires:
        vague_ask_rate = (
            sum(1 for r in events_with_requires if r.did_without_full_coverage)
            / len(events_with_requires)
        )
    else:
        vague_ask_rate = 0.0

    # [3-b] Rubric Score: Do 前にカバーできた should_require 項目の平均割合
    rubric_scores = []
    for r in records:
        if not r.should_require:
            rubric_scores.append(1.0)
        else:
            rubric_scores.append(len(r.covered_requires) / len(r.should_require))
    rubric_score = sum(rubric_scores) / total

    # [3-c] Redundant Question Rate
    total_asked = sum(len(r.questions_asked) for r in records)
    total_redundant = sum(len(r.redundant_questions) for r in records)
    redundant_question_rate = total_redundant / total_asked if total_asked > 0 else 0.0

    # [3-d] Early Discovery Rate: 最初 N 手以内で全 should_require を完了した割合
    # covered_per_question[i] を先頭 N 件だけ合算して判定する
    early_count = 0
    for r in events_with_requires:
        covered_after_n: set[str] = set()
        for per_q in r.covered_per_question[:early_discovery_n]:
            covered_after_n.update(per_q)
        if set(r.should_require).issubset(covered_after_n):
            early_count += 1
    early_discovery_rate = (
        early_count / len(events_with_requires) if events_with_requires else 0.0
    )

    return EvalResult(
        final_pass_rate=final_pass_rate,
        exact_match=exact_match,
        vague_ask_rate=vague_ask_rate,
        rubric_score=rubric_score,
        redundant_question_rate=redundant_question_rate,
        early_discovery_rate=early_discovery_rate,
        total_events=total,
        early_discovery_n=early_discovery_n,
    )


# ---------------------------------------------------------------------------
# OpenAI 判定ヘルパー
# ---------------------------------------------------------------------------


def _call_openai_judge(prompt: str, *, model: str = DEFAULT_JUDGE_MODEL) -> bool:
    """OpenAI に True / False を返させる汎用バイナリ判定。"""
    client = OpenAI()
    response = client.responses.create(
        model=model,
        instructions=_JUDGE_INSTRUCTIONS,
        input=prompt,
    )
    return response.output_text.strip().lower().startswith("true")


def require_covered_by(require: str, question: str) -> bool:
    """should_require 1 件が question でカバーされているかをキーワードマッチで判定する。

    フォールバック用。英数字トークン（3 文字以上）のいずれかが question に含まれれば True。
    """
    q_lower = question.lower()
    tokens = re.findall(r"[a-z0-9_]+", require.lower())
    meaningful = [t for t in tokens if len(t) > 2]
    if not meaningful:
        return False
    return any(token in q_lower for token in meaningful)


# ---------------------------------------------------------------------------
# ドメインアダプター（抽象基底）
# ---------------------------------------------------------------------------


class DomainAdapter(ABC):
    """ドメイン別の prompt 構築・Do 変換・評価判定を抽象化するアダプター。"""

    # ── prompt 構築 / Do 変換（サブクラスで必須実装） ──────────────────────

    @abstractmethod
    def build_decision_prompt(
        self,
        *,
        case: Case,
        context: TurnContext,
        allow_question: bool,
    ) -> str:
        """Question か Do を選ぶ prompt を返す。"""

    @abstractmethod
    def build_answer_prompt(
        self,
        *,
        case: Case,
        context: TurnContext,
        question: str,
    ) -> str:
        """hidden_info を使って質問へ答える環境側の prompt を返す。"""

    @abstractmethod
    def parse_do(self, payload: dict[str, Any]) -> Do:
        """モデル出力を Do として検証・変換する。"""

    # ── 評価判定（デフォルト実装あり、ドメイン固有に override 可） ──────────

    def judge_require_coverage(
        self,
        require: str,
        question: str,
        *,
        model: str = DEFAULT_JUDGE_MODEL,
    ) -> bool:
        """should_require 1 件が question でカバーされているかを OpenAI で判定する。

        ドメイン固有のフォーマットに対応する場合は override する。
        """
        prompt = (
            "次の「確認すべき項目」が「エージェントの質問」で触れられているか判定してください。\n"
            f"確認すべき項目: {require}\n"
            f"エージェントの質問: {question}\n"
            "触れられていれば True、そうでなければ False とだけ答えてください。"
        )
        return _call_openai_judge(prompt, model=model)

    def build_covered_requires(
        self,
        should_require: list[str],
        questions_asked: list[str],
        *,
        model: str = DEFAULT_JUDGE_MODEL,
    ) -> tuple[list[list[str]], list[str]]:
        """質問リストをもとに covered_per_question と covered_requires を構築する。

        Returns:
            (covered_per_question, covered_requires)
            covered_per_question[i] = questions_asked[i] がカバーした should_require 項目。
            covered_requires = 全質問でカバーされた should_require の和集合（重複なし）。
        """
        covered_per_question: list[list[str]] = []
        covered_all: set[str] = set()

        for question in questions_asked:
            covered_by_this: list[str] = []
            for req in should_require:
                if self.judge_require_coverage(req, question, model=model):
                    covered_by_this.append(req)
            covered_per_question.append(covered_by_this)
            covered_all.update(covered_by_this)

        return covered_per_question, list(covered_all)

    def is_feasible(self, do: Do, context: TurnContext) -> bool:
        """最終 Do が全制約を満たし実行可能かどうかを判定する。

        ドメイン固有の制約チェックが必要な場合は override する。
        デフォルトは常に True（判定不能の場合の安全側フォールバック）。
        """
        return True

    def is_exact_match(self, do: Do, exp_do: Do) -> bool:
        """最終 Do が正解と完全一致するかどうかを判定する。

        ドメイン固有の許容誤差がある場合は override する。
        """
        return do.model_dump() == exp_do.model_dump()
