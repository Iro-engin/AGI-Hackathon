from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Do(BaseModel):
    pass


class Info(BaseModel):
    pass


class State(BaseModel):
    constraint: list[str] = Field(default_factory=list)
    hidden_info: Info
    info: Info
    got_info: Info


class Event(BaseModel):
    message: str
    state_before: State
    exp_do: Do
    should_require: list[str] = Field(default_factory=list)

    @property
    def requires(self) -> list[str]:
        return self.should_require


Events = Event


class Case(BaseModel):
    """
    1つのタスクケースを表すモデル。
    """
    task_id: str
    difficulty: Literal["easy", "medium", "hard"]
    domain: Literal["meeting", "conference", "finance", "science"]
    initial_request: str
    initial_state: State
    initial_do: Do | None = None
    events: list[Event] = Field(default_factory=list)
    initial_requires: list[str] = Field(default_factory=list)


class Question(BaseModel):
    """
    タスクケースに対する質問を表すモデル。
    """
    raw_question: str
    questions: list[str]
    cleared_before: list[bool]
    cleared_by: list[bool]
    not_needed_count: int
