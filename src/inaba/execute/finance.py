"""finance ドメインの execute DomainAdapter 実装。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..models.base import Case
from ..models.finance import FinanceDo
from .base import DomainAdapter, TurnContext

_TERM_GUIDE = """
[Term Guide]
- stock_price_before: Each stock's price before the current news is reflected.
- inference_rate: The influence each past news item has on stock price movement (-1 to 1).
  Positive values signal upward movement; negative values signal downward.
- efficient_rate_change: A correction to inference_rate's effective strength (-0.25 to 0.25).
  Accumulated from past news and updated across turns.
- efficient_rate_now (hidden): The final effective price change rate, computed as
  inference_rate + efficient_rate_change after correction.
  Optimal stock_rates allocation strongly depends on this value.
- resource_mix_rate: The mixing ratio indicating how much each sector's past news
  (hidden_resource) is referenced by the current news.
  Note: efficient_rate_now changes with accumulated news history,
  so querying sector-specific related news improves its accuracy.
""".strip()


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
            "You may choose to ask a question"
            if allow_question
            else "Do not ask a question at this point; return a Do"
        )
        past_news_section = _format_past_news(context.past_event_messages)
        return f"""
You are an agent working on a financial task.
In this turn, choose exactly one of: Question or Do.
{question_rule}.

{_TERM_GUIDE}

Initial request:
{case.initial_request}

Visible information at the start:
{json.dumps(context.initial_got_info, ensure_ascii=False, indent=2)}

Current turn type:
{context.stage}

Current constraints:
{json.dumps(context.current_constraint, ensure_ascii=False, indent=2)}
{past_news_section}
Current event message (current news):
{json.dumps(context.event_message, ensure_ascii=False)}

Currently visible information:
{json.dumps(context.current_got_info, ensure_ascii=False, indent=2)}

Questions and answers so far this turn:
{json.dumps([item.model_dump() for item in context.question_answers], ensure_ascii=False, indent=2)}

Previous turn Do:
{json.dumps(context.previous_do, ensure_ascii=False, indent=2)}

When returning a Do, return stock_rates as a dict covering all stocks with values summing to 1.0.
Stocks with higher efficient_rate_now (hidden) should receive higher allocation.
Querying sector-specific related news improves the accuracy of efficient_rate_now estimates.

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
        past_news_section = _format_past_news(context.past_event_messages)
        return f"""
You are the environment side of a financial task.
Answer the question using only the current turn's hidden_info and visible information.
Do not speculate about things you cannot answer; state that they are unknown.

{_TERM_GUIDE}

Initial request:
{case.initial_request}
{past_news_section}
Current event message (current news):
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


def _format_past_news(past_event_messages: list[str]) -> str:
    """過去のニュース履歴をプロンプト用の文字列に整形する。空の場合は空文字を返す。"""
    if not past_event_messages:
        return ""
    lines = ["\nPast news history (already reflected in efficient_rate_now changes):"]
    for index, message in enumerate(past_event_messages, start=1):
        lines.append(f"  [{index}] {message}")
    lines.append("")
    return "\n".join(lines)


FINANCE_ADAPTER = FinanceAdapter()
