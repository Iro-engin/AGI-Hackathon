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
            "質問を選んでもよい"
            if allow_question
            else "この時点では質問せず、必ず Do を返す"
        )
        return f"""
あなたは会議設定タスクを進める agent です。
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

Do を返す場合は、会議場所と開始・終了時刻を具体的に返してください。
when_from と when_to は ISO 8601 文字列で返してください。

返却形式は必ず次のどちらか:
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
あなたは会議設定タスクの環境側です。
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
            "次の会議提案が、参加者の都合と制約をすべて満たしているか判定してください。\n\n"
            f"会議提案:\n{do_json}\n\n"
            f"現在の制約:\n{constraint_json}\n\n"
            f"参加者の実際の都合（hidden_info）:\n{hidden_json}\n\n"
            "全参加者が参加でき、所要時間も満たしているなら True、"
            "そうでなければ False とだけ答えてください。"
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
