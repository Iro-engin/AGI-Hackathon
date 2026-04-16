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
DEFAULT_DECISION_PROVIDER = "openai"
DEFAULT_ANSWER_PROVIDER = "openai"
DEFAULT_GEMINI_THINKING_LEVEL = "low"

SELECTION_INSTRUCTIONS = """
You are an agent that selects actions using only the given state.
Return only valid JSON. Do not include Markdown or explanatory text.
""".strip()

ANSWER_INSTRUCTIONS = """
You are the environment side that has access to hidden information.
Answer questions concisely and specifically in English.
Use only the provided information. Do not add unsupported inferences.
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
    past_event_messages: list[str] = Field(default_factory=list)
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


Provider = Literal["openai", "gemini"]


def run_case(
    case: Case,
    *,
    decision_provider: Provider = DEFAULT_DECISION_PROVIDER,
    decision_model: str = DEFAULT_DECISION_MODEL,
    answer_provider: Provider = DEFAULT_ANSWER_PROVIDER,
    answer_model: str = DEFAULT_ANSWER_MODEL,
    max_questions_per_turn: int = 4,
    gemini_thinking_level: str = DEFAULT_GEMINI_THINKING_LEVEL,
) -> EvaluationRun:
    """指定 provider を使い、各ターンで Question か Do を選ぶ実行を行う。"""
    adapter = _get_domain_adapter(case.domain)
    openai_client = _build_openai_client() if (
        decision_provider == "openai" or answer_provider == "openai"
    ) else None
    gemini_client = _build_gemini_client() if (
        decision_provider == "gemini" or answer_provider == "gemini"
    ) else None

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
                provider=decision_provider,
                openai_client=openai_client,
                gemini_client=gemini_client,
                model=decision_model,
                instructions=SELECTION_INSTRUCTIONS,
                prompt=selection_prompt,
                response_model=SelectionResponse,
                gemini_thinking_level=gemini_thinking_level,
            )

            if selection.action == "question" and allow_question and selection.question:
                answer_prompt = adapter.build_answer_prompt(
                    case=case,
                    context=context,
                    question=selection.question,
                )
                answer = _call_openai_text(
                    provider=answer_provider,
                    openai_client=openai_client,
                    gemini_client=gemini_client,
                    model=answer_model,
                    instructions=ANSWER_INSTRUCTIONS,
                    prompt=answer_prompt,
                    gemini_thinking_level=gemini_thinking_level,
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
                    provider=decision_provider,
                    openai_client=openai_client,
                    gemini_client=gemini_client,
                    model=decision_model,
                    instructions=SELECTION_INSTRUCTIONS,
                    prompt=forced_prompt,
                    response_model=SelectionResponse,
                    gemini_thinking_level=gemini_thinking_level,
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


def run_case_with_openai(
    case: Case,
    *,
    decision_model: str = DEFAULT_DECISION_MODEL,
    answer_model: str = DEFAULT_ANSWER_MODEL,
    max_questions_per_turn: int = 4,
) -> EvaluationRun:
    """後方互換のため、OpenAI 固定実行を維持する。"""
    return run_case(
        case=case,
        decision_provider="openai",
        decision_model=decision_model,
        answer_provider="openai",
        answer_model=answer_model,
        max_questions_per_turn=max_questions_per_turn,
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
    past_messages = [case.events[i].message for i in range(turn_index - 1)]
    return TurnContext(
        turn=turn_index + 1,
        stage="event",
        event_index=turn_index - 1,
        event_message=event.message,
        past_event_messages=past_messages,
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


def _build_gemini_client() -> Any:
    """GEMINI_API_KEY を確認して Gemini client を作る。"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")
    try:
        from google import genai
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "google-genai is not installed. Install it to use Gemini provider."
        ) from exc
    return genai.Client(api_key=api_key)


def _call_openai_json(
    *,
    provider: Provider,
    openai_client: OpenAI | None,
    gemini_client: Any | None,
    model: str,
    instructions: str,
    prompt: str,
    response_model: type[BaseModel],
    gemini_thinking_level: str,
) -> BaseModel:
    """provider に応じた API を呼び、JSON を Pydantic で検証する。"""
    if provider == "openai":
        if openai_client is None:
            raise RuntimeError("OpenAI client is not initialized")
        response = openai_client.responses.create(
            model=model,
            instructions=instructions,
            input=prompt,
        )
        raw_text = response.output_text
    elif provider == "gemini":
        if gemini_client is None:
            raise RuntimeError("Gemini client is not initialized")
        raw_text = _call_gemini_text(
            client=gemini_client,
            model=model,
            instructions=instructions,
            prompt=prompt,
            gemini_thinking_level=gemini_thinking_level,
        )
    else:  # pragma: no cover
        raise ValueError(f"unsupported provider: {provider}")

    payload = _parse_json_text(raw_text)
    return response_model.model_validate(payload)


def _call_openai_text(
    *,
    provider: Provider,
    openai_client: OpenAI | None,
    gemini_client: Any | None,
    model: str,
    instructions: str,
    prompt: str,
    gemini_thinking_level: str,
) -> str:
    """provider に応じた API を呼び、回答文をそのまま返す。"""
    if provider == "openai":
        if openai_client is None:
            raise RuntimeError("OpenAI client is not initialized")
        response = openai_client.responses.create(
            model=model,
            instructions=instructions,
            input=prompt,
        )
        return response.output_text.strip()

    if provider == "gemini":
        if gemini_client is None:
            raise RuntimeError("Gemini client is not initialized")
        return _call_gemini_text(
            client=gemini_client,
            model=model,
            instructions=instructions,
            prompt=prompt,
            gemini_thinking_level=gemini_thinking_level,
        )

    raise ValueError(f"unsupported provider: {provider}")


def _call_gemini_text(
    *,
    client: Any,
    model: str,
    instructions: str,
    prompt: str,
    gemini_thinking_level: str,
) -> str:
    """Gemini SDK を呼び、回答文を返す。"""
    from google.genai import types

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=instructions,
            thinking_config=types.ThinkingConfig(thinking_level=gemini_thinking_level),
        ),
    )
    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("Gemini response did not include text content")
    return text


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
