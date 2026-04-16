from __future__ import annotations

import json
import os
from typing import Any, Literal, Protocol

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

from ..models.base import Case

load_dotenv()
DEFAULT_DECISION_MODEL = "gpt-4.1-mini"
DEFAULT_ANSWER_MODEL = "gpt-4o-mini"

SELECTION_INSTRUCTIONS = """
あなたは与えられた状態だけを使って行動を選ぶエージェントです。
必ずJSONのみを返し、Markdownや説明文は返さないでください。
""".strip()

ANSWER_INSTRUCTIONS = """
あなたは隠れた情報も知っている環境側の応答役です。
質問に対して、日本語で簡潔かつ具体的に回答してください。
与えられた情報だけを使い、根拠のない補完はしないでください。
""".strip()


class SelectionResponse(BaseModel):
    """1回の選択で返す JSON 形式。"""

    action: Literal["question", "do"]
    question: str | None = None
    do: dict[str, Any] | None = None
    reason: str | None = None


class QuestionAnswerRecord(BaseModel):
    """1回の質問と回答の記録。"""

    question: str
    answer: str


class TurnExecution(BaseModel):
    """1ターン分の実行ログ。"""

    turn: int
    stage: Literal["initial", "event"]
    event_index: int | None = None
    event_message: str | None = None
    question_answers: list[QuestionAnswerRecord] = Field(default_factory=list)
    final_do: dict[str, Any]


class EvaluationRun(BaseModel):
    """case 1件に対する質問・実行の全履歴。"""

    task_id: str
    domain: str
    decision_model: str
    answer_model: str
    turns: list[TurnExecution] = Field(default_factory=list)


class TurnContext(BaseModel):
    """現在ターンで agent に見せる情報と hidden 情報をまとめる。"""

    turn: int
    stage: Literal["initial", "event"]
    event_index: int | None = None
    event_message: str | None = None
    initial_request: str
    initial_constraint: list[str] = Field(default_factory=list)
    initial_got_info: dict[str, Any] = Field(default_factory=dict)
    current_constraint: list[str] = Field(default_factory=list)
    current_got_info: dict[str, Any] = Field(default_factory=dict)
    current_hidden_info: dict[str, Any] = Field(default_factory=dict)
    should_require: list[str] = Field(default_factory=list)
    previous_do: dict[str, Any] | None = None
    question_answers: list[QuestionAnswerRecord] = Field(default_factory=list)


class DomainAdapter(Protocol):
    """domain ごとの prompt / Do 変換を抽象化する。"""

    def build_decision_prompt(
        self,
        *,
        case: Case,
        context: TurnContext,
        allow_question: bool,
    ) -> str:
        ...

    def build_answer_prompt(
        self,
        *,
        case: Case,
        context: TurnContext,
        question: str,
    ) -> str:
        ...

    def parse_do(self, payload: dict[str, Any]) -> BaseModel:
        ...


def run_case_with_openai(
    case: Case,
    *,
    decision_model: str = DEFAULT_DECISION_MODEL,
    answer_model: str = DEFAULT_ANSWER_MODEL,
    max_questions_per_turn: int = 4,
) -> EvaluationRun:
    """OpenAI を使って、各ターンで Question か Do を選ぶ実行を行う。"""
    adapter = _get_domain_adapter(case.domain)
    client = _build_openai_client()

    turns: list[TurnExecution] = []
    previous_do: dict[str, Any] | None = None

    for turn_index in range(len(case.events) + 1):
        context = _build_turn_context(
            case=case,
            turn_index=turn_index,
            previous_do=previous_do,
        )
        question_answers: list[QuestionAnswerRecord] = []
        final_do_payload: dict[str, Any] | None = None

        for question_count in range(max_questions_per_turn + 1):
            context.question_answers = list(question_answers)
            allow_question = question_count < max_questions_per_turn
            selection_prompt = adapter.build_decision_prompt(
                case=case,
                context=context,
                allow_question=allow_question,
            )
            selection = _call_openai_json(
                client=client,
                model=decision_model,
                instructions=SELECTION_INSTRUCTIONS,
                prompt=selection_prompt,
                response_model=SelectionResponse,
            )

            if selection.action == "question" and allow_question and selection.question:
                answer_prompt = adapter.build_answer_prompt(
                    case=case,
                    context=context,
                    question=selection.question,
                )
                answer = _call_openai_text(
                    client=client,
                    model=answer_model,
                    instructions=ANSWER_INSTRUCTIONS,
                    prompt=answer_prompt,
                )
                question_answers.append(
                    QuestionAnswerRecord(question=selection.question, answer=answer)
                )
                continue

            if selection.action == "do" and selection.do is not None:
                final_do_payload = adapter.parse_do(selection.do).model_dump(mode="json")
                break

            if not allow_question:
                forced_prompt = adapter.build_decision_prompt(
                    case=case,
                    context=context,
                    allow_question=False,
                )
                forced_selection = _call_openai_json(
                    client=client,
                    model=decision_model,
                    instructions=SELECTION_INSTRUCTIONS,
                    prompt=forced_prompt,
                    response_model=SelectionResponse,
                )
                if forced_selection.do is None:
                    raise RuntimeError("model did not return Do after question limit")
                final_do_payload = adapter.parse_do(
                    forced_selection.do
                ).model_dump(mode="json")
                break

            raise RuntimeError("model returned invalid selection payload")

        if final_do_payload is None:
            raise RuntimeError("turn finished without final Do")

        turns.append(
            TurnExecution(
                turn=turn_index + 1,
                stage=context.stage,
                event_index=context.event_index,
                event_message=context.event_message,
                question_answers=question_answers,
                final_do=final_do_payload,
            )
        )
        previous_do = final_do_payload

    return EvaluationRun(
        task_id=case.task_id,
        domain=case.domain,
        decision_model=decision_model,
        answer_model=answer_model,
        turns=turns,
    )


def _build_turn_context(
    *,
    case: Case,
    turn_index: int,
    previous_do: dict[str, Any] | None,
) -> TurnContext:
    """initial / event のどちらのターンかに応じて prompt 用の状態をまとめる。"""
    initial_state = case.initial_state
    if turn_index == 0:
        return TurnContext(
            turn=1,
            stage="initial",
            initial_request=case.initial_request,
            initial_constraint=list(initial_state.constraint),
            initial_got_info=initial_state.got_info.model_dump(mode="json"),
            current_constraint=list(initial_state.constraint),
            current_got_info=initial_state.got_info.model_dump(mode="json"),
            current_hidden_info=initial_state.hidden_info.model_dump(mode="json"),
            should_require=list(case.initial_requires),
            previous_do=previous_do,
        )

    event = case.events[turn_index - 1]
    state_before = event.state_before
    return TurnContext(
        turn=turn_index + 1,
        stage="event",
        event_index=turn_index - 1,
        event_message=event.message,
        initial_request=case.initial_request,
        initial_constraint=list(initial_state.constraint),
        initial_got_info=initial_state.got_info.model_dump(mode="json"),
        current_constraint=list(state_before.constraint),
        current_got_info=state_before.got_info.model_dump(mode="json"),
        current_hidden_info=state_before.hidden_info.model_dump(mode="json"),
        should_require=list(event.should_require),
        previous_do=previous_do,
    )


def _build_openai_client() -> OpenAI:
    """OPENAI_API_KEY を確認して OpenAI client を作る。"""
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return OpenAI()


def _call_openai_json(
    *,
    client: OpenAI,
    model: str,
    instructions: str,
    prompt: str,
    response_model: type[BaseModel],
) -> BaseModel:
    """OpenAI Responses API を呼び、JSON を Pydantic で検証する。"""
    response = client.responses.create(
        model=model,
        instructions=instructions,
        input=prompt,
    )
    payload = _parse_json_text(response.output_text)
    return response_model.model_validate(payload)


def _call_openai_text(
    *,
    client: OpenAI,
    model: str,
    instructions: str,
    prompt: str,
) -> str:
    """OpenAI Responses API を呼び、回答文をそのまま返す。"""
    response = client.responses.create(
        model=model,
        instructions=instructions,
        input=prompt,
    )
    return response.output_text.strip()


def _parse_json_text(text: str) -> dict[str, Any]:
    """Responses API の出力から JSON 部分だけを取り出す。"""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return json.loads(stripped)


def _get_domain_adapter(domain: str) -> DomainAdapter:
    """domain に応じた adapter を返す。"""
    if domain == "meeting":
        from .meeting import MEETING_ADAPTER

        return MEETING_ADAPTER
    if domain == "finance":
        from .finance import FINANCE_ADAPTER

        return FINANCE_ADAPTER
    raise ValueError(f"unsupported inaba evaluation domain: {domain}")
