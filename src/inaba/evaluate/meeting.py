"""meeting ドメインの DomainAdapter 実装。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..models.base import Case, Do
from ..models.meeting import MeetingDo
from .base import DEFAULT_JUDGE_MODEL, DomainAdapter, TurnContext, _call_openai_judge


@dataclass(frozen=True)
class MeetingAdapter(DomainAdapter):
    """meeting 用の prompt / Do 変換・評価をまとめる adapter。"""

    model: str = DEFAULT_JUDGE_MODEL

    def build_decision_prompt(
        self,
        *,
        case: Case,
        context: TurnContext,
        allow_question: bool,
    ) -> str:
        """meeting の Question / Do 選択 prompt を作る。"""
        question_rule = (
            "You may choose to ask a question"
            if allow_question
            else "Do not ask a question at this point; return a Do"
        )
        return f"""
You are an agent working on a meeting scheduling task.
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

When returning a Do, specify the meeting location and start/end time.
Return when_from and when_to as ISO 8601 strings.

Return exactly one of the following formats:
{{
  "action": "question",
  "question": "..."
}}

{{
  "action": "do",
  "do": {{
    "where": "online",
    "when_from": "2026-04-20T10:00:00",
    "when_to": "2026-04-20T10:45:00"
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
        """meeting で hidden_info を使って質問へ答える prompt を作る。"""
        return f"""
You are the environment side of a meeting scheduling task.
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

    def parse_do(self, payload: dict[str, Any]) -> MeetingDo:
        """モデル出力を MeetingDo として検証する。"""
        return MeetingDo.model_validate(payload)

    # ── 評価判定 ────────────────────────────────────────────────────────────

    def is_feasible(self, do: Do, context: TurnContext) -> bool:
        """提案された会議が参加者の都合と制約を全て満たすか OpenAI で判定する。"""
        constraint_json = json.dumps(context.current_constraint, ensure_ascii=False, indent=2)
        hidden_json = json.dumps(context.current_hidden_info, ensure_ascii=False, indent=2)
        do_json = json.dumps(do.model_dump(), ensure_ascii=False, indent=2)
        prompt = (
            "Determine whether the following meeting proposal satisfies all participant"
            " availability and constraints.\n\n"
            f"Meeting proposal:\n{do_json}\n\n"
            f"Current constraints:\n{constraint_json}\n\n"
            f"Actual participant availability (hidden_info):\n{hidden_json}\n\n"
            "Answer only True if all participants can attend and the duration is met,"
            " or False otherwise."
        )
        return _call_openai_judge(prompt, model=self.model)

    def is_exact_match(self, do: Do, exp_do: Do) -> bool:
        """where / when_from / when_to がすべて一致するか判定する。"""
        if not isinstance(do, MeetingDo) or not isinstance(exp_do, MeetingDo):
            return False
        return (
            do.where == exp_do.where
            and do.when_from == exp_do.when_from
            and do.when_to == exp_do.when_to
        )


MEETING_ADAPTER = MeetingAdapter()
