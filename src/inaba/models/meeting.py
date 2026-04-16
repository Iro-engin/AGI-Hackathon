from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from .base import Case, Do, Event, Info, State


class Attendance(BaseModel):
    where: str | None = None
    when_from: datetime | None = None
    when_to: datetime | None = None


class PeopleInfo(BaseModel):
    name: str
    can_attend: list[Attendance] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    person_index: int | None = None
    hidden_attendance_indexes: list[int] = Field(default_factory=list)


class MeetingDo(Do):
    where: str
    when_from: datetime
    when_to: datetime


class MeetingInfo(Info):
    need_duration: int
    state_people: list[PeopleInfo] = Field(default_factory=list)


class MeetingState(State):
    constraint: list[str] = Field(default_factory=list)
    hidden_info: MeetingInfo
    info: MeetingInfo
    got_info: MeetingInfo


class MeetingEvent(Event):
    state_before: MeetingState
    exp_do: MeetingDo
    should_require: list[str] = Field(default_factory=list)


class MeetingCase(Case):
    domain: Literal["meeting"] = "meeting"
    initial_state: MeetingState
    initial_do: MeetingDo | None = None
    events: list[MeetingEvent] = Field(default_factory=list)
    initial_requires: list[str] = Field(default_factory=list)
