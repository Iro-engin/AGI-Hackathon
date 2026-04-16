"""finance ドメインの DomainAdapter 実装。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..models.base import Case, Do
from ..models.finance import FinanceDo
from .base import DEFAULT_JUDGE_MODEL, DomainAdapter, TurnContext

# stock_rates の合計が [1 - _RATE_TOL, 1 + _RATE_TOL] に収まれば feasible とみなす
_RATE_TOL = 0.05
# is_exact_match: 各銘柄の rates 差がこれ以下なら一致とみなす
_MATCH_TOL = 0.05


@dataclass(frozen=True)
class FinanceAdapter(DomainAdapter):
    """finance 用の prompt / Do 変換・評価をまとめる adapter。"""

    model: str = DEFAULT_JUDGE_MODEL

    def build_decision_prompt(
        self,
        *,
        case: Case,
        context: TurnContext,
        allow_question: bool,
    ) -> str:
        """finance の Question / Do 選択 prompt を作る。"""
        question_rule = (
            "You may choose to ask a question"
            if allow_question
            else "Do not ask a question at this point; return a Do"
        )
        return f"""
You are an agent working on a financial task.
In this turn, choose exactly one of: Question or Do.
{question_rule}.

Initial request:
{case.initial_request}

Visible information at the start:
{json.dumps(context.initial_got_info, ensure_ascii=False, indent=2)}

Current turn type:
{context.stage}

Current constraints:
{json.dumps(context.current_constraint, ensure_ascii=False, indent=2)}

Current event message:
{json.dumps(context.event_message, ensure_ascii=False)}

Currently visible information:
{json.dumps(context.current_got_info, ensure_ascii=False, indent=2)}

Questions and answers so far this turn:
{json.dumps([item.model_dump() for item in context.question_answers], ensure_ascii=False, indent=2)}

Previous turn Do:
{json.dumps(context.previous_do, ensure_ascii=False, indent=2)}

When returning a Do, return stock_rates as a dict covering all stocks.

Return exactly one of the following formats:
{{
  "action": "question",
  "question": "..."
}}

{{
  "action": "do",
  "do": {{
    "stock_rates": {{
      "stock_a": 0.20,
      "stock_b": 0.20,
      "stock_c": 0.20,
      "stock_d": 0.20,
      "stock_e": 0.20
    }}
  }}
}}
""".strip()

    def build_answer_prompt(
        self,
        *,
        case: Case,
        context: TurnContext,
        question: str,
    ) -> str:
        """finance で hidden_info を使って質問へ答える prompt を作る。"""
        return f"""
You are the environment side of a financial task.
Answer the question using only the current turn's hidden_info and visible information.
Do not speculate about things you cannot answer; state that they are unknown.

Initial request:
{case.initial_request}

Current event message:
{json.dumps(context.event_message, ensure_ascii=False)}

Currently visible information:
{json.dumps(context.current_got_info, ensure_ascii=False, indent=2)}

Current hidden_info:
{json.dumps(context.current_hidden_info, ensure_ascii=False, indent=2)}

Questions and answers so far this turn:
{json.dumps([item.model_dump() for item in context.question_answers], ensure_ascii=False, indent=2)}

Question:
{question}
""".strip()

    def parse_do(self, payload: dict[str, Any]) -> FinanceDo:
        """モデル出力を FinanceDo として検証し、比率を正規化する。"""
        do = FinanceDo.model_validate(payload)
        total = sum(do.stock_rates.values())
        if total <= 0:
            raise ValueError("finance do requires positive stock_rates total")
        do.stock_rates = {
            stock: round(value / total, 4)
            for stock, value in do.stock_rates.items()
        }
        return do

    # ── 評価判定 ────────────────────────────────────────────────────────────

    def is_feasible(self, do: Do, context: TurnContext) -> bool:
        """stock_rates が非負・全銘柄網羅・合計が 1.0 ± _RATE_TOL であれば feasible とみなす。

        finance の feasibility は数値制約のみで判定できるためルールベースで行う。
        """
        if not isinstance(do, FinanceDo):
            return False
        rates = do.stock_rates
        if not rates:
            return False
        if any(v < 0.0 for v in rates.values()):
            return False
        total = sum(rates.values())
        return abs(total - 1.0) <= _RATE_TOL

    def is_exact_match(self, do: Do, exp_do: Do) -> bool:
        """全銘柄の stock_rates 差が _MATCH_TOL 以内なら一致とみなす。

        浮動小数点の正規化誤差を吸収するため厳密等値比較を行わない。
        """
        if not isinstance(do, FinanceDo) or not isinstance(exp_do, FinanceDo):
            return False
        if set(do.stock_rates.keys()) != set(exp_do.stock_rates.keys()):
            return False
        return all(
            abs(do.stock_rates[k] - exp_do.stock_rates[k]) <= _MATCH_TOL
            for k in do.stock_rates
        )


FINANCE_ADAPTER = FinanceAdapter()
