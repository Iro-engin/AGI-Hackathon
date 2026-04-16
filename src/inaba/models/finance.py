from __future__ import annotations

from typing import Literal

from pydantic import Field

from .base import Case, Do, Event, Info, State


class NewsInfo(Info):
    content: str
    inference_rate: dict[str, float] = Field(default_factory=dict)
    efficient_rate_change: dict[str, float] = Field(default_factory=dict)


class CurrentNewsInfo(NewsInfo):
    resource_mix_rate: dict[str, float] = Field(default_factory=dict)


class FinanceInfo(Info):
    news: CurrentNewsInfo | None = None
    stock_price_before: dict[str, float] = Field(default_factory=dict)
    hidden_resource: dict[str, NewsInfo] = Field(default_factory=dict)
    efficient_rate_now: dict[str, float] = Field(default_factory=dict)


class FinanceDo(Do):
    stock_rates: dict[str, float] = Field(default_factory=dict)


class FinanceState(State):
    constraint: list[str] = Field(default_factory=list)
    hidden_info: FinanceInfo
    info: FinanceInfo
    got_info: FinanceInfo


class FinanceEvent(Event):
    state_before: FinanceState
    exp_do: FinanceDo
    should_require: list[str] = Field(default_factory=list)


class FinanceCase(Case):
    domain: Literal["finance"] = "finance"
    initial_state: FinanceState
    events: list[FinanceEvent] = Field(default_factory=list)
    initial_requires: list[str] = Field(default_factory=list)
