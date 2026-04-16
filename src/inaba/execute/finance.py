from __future__ import annotations

import json
from dataclasses import dataclass

from ..models.base import Case
from ..models.finance import FinanceDo
from .base import DomainAdapter, TurnContext


@dataclass(frozen=True)
class FinanceAdapter(DomainAdapter):
    """finance 用の prompt / Do 変換をまとめる adapter。"""

    def build_decision_prompt(
        self,
        *,
        case: Case,
        context: TurnContext,
        allow_question: bool,
    ) -> str:
        """finance の Question / Do 選択 prompt を作る。"""
        question_rule = (
            "質問を選んでもよい"
            if allow_question
            else "この時点では質問せず、必ず Do を返す"
        )
        return f"""
あなたは金融タスクを進める agent です。
今回のターンでは、Question か Do のどちらか1つを選んでください。
{question_rule}。

初期依頼:
{case.initial_request}

初期状態で見えている情報:
{json.dumps(context.initial_got_info, ensure_ascii=False, indent=2)}

今回のターン種別:
{context.stage}

今回の制約:
{json.dumps(context.current_constraint, ensure_ascii=False, indent=2)}

今回の event message:
{json.dumps(context.event_message, ensure_ascii=False)}

今回見えている情報:
{json.dumps(context.current_got_info, ensure_ascii=False, indent=2)}

このターンのこれまでの質問と回答:
{json.dumps([item.model_dump() for item in context.question_answers], ensure_ascii=False, indent=2)}

直前ターンの Do:
{json.dumps(context.previous_do, ensure_ascii=False, indent=2)}

Do を返す場合は stock_rates を返してください。
stock_rates は全銘柄の比率を持つ辞書にしてください。
なお、stock_ratesの最適解となる比率は各株の株価の比率とほぼ等しいです。
返却形式は必ず次のどちらか:
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
あなたは金融タスクの環境側です。
質問に対して、現在ターンの hidden_info と visible 情報だけを使って日本語で答えてください。
答えられないことは推測せず、その旨を述べてください。

初期依頼:
{case.initial_request}

今回の event message:
{json.dumps(context.event_message, ensure_ascii=False)}

今回見えている情報:
{json.dumps(context.current_got_info, ensure_ascii=False, indent=2)}

今回の hidden_info:
{json.dumps(context.current_hidden_info, ensure_ascii=False, indent=2)}

このターンのこれまでの質問と回答:
{json.dumps([item.model_dump() for item in context.question_answers], ensure_ascii=False, indent=2)}

質問:
{question}
""".strip()

    def parse_do(self, payload: dict[str, object]) -> FinanceDo:
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


FINANCE_ADAPTER = FinanceAdapter()
