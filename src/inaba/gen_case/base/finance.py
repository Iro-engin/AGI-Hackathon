from __future__ import annotations

import argparse
import random
from copy import deepcopy
from pathlib import Path
from typing import Literal

from ...models.finance import (
    CurrentNewsInfo,
    FinanceCase,
    FinanceDo,
    FinanceEvent,
    FinanceInfo,
    FinanceState,
    NewsInfo,
)

Difficulty = Literal["easy", "medium", "hard"]

EVENT_COUNT_BY_DIFFICULTY: dict[Difficulty, int] = {
    "easy": 2,
    "medium": 4,
    "hard": 4,
}

INITIAL_REQUIRE_COUNT_BY_DIFFICULTY: dict[Difficulty, int] = {
    "easy": 2,
    "medium": 4,
    "hard": 6,
}

THEME_LIBRARY = [
    {
        "name": "ai_rotation",
        "stocks": [
            "ai_server_supply_chain",
            "power_grid_equipment",
            "thermal_management",
            "semiconductor_equipment",
            "data_center_reits",
        ],
        "history": {
            "ai_server_supply_chain": [
                "Server stocks led the decline during a past hyperscaler investment slowdown.",
                "During GPU procurement restarts, the server supply chain led the rally.",
            ],
            "power_grid_equipment": [
                "During prolonged power shortages, grid equipment stocks gained relative strength.",
                "A delay in large-scale projects caused a temporary stall.",
            ],
            "thermal_management": [
                "Cooling-related stocks rose alongside high-density rack demand.",
                "Installation delays caused short-term underperformance.",
            ],
            "semiconductor_equipment": [
                "Manufacturing equipment lagged but rallied when advanced capex restarts resumed.",
                "Capex deferrals created short-term headwinds.",
            ],
            "data_center_reits": [
                "REITs with power-sufficient sites were previously preferred by the market.",
                "Development plan delays led to a re-rating.",
            ],
        },
    },
    {
        "name": "rates_rotation",
        "stocks": ["reit", "banks", "insurers", "utilities", "brokers"],
        "history": {
            "reit": [
                "REITs previously reacted first to rate-cut cycles.",
                "Funding concerns triggered heavy selling in REITs.",
            ],
            "banks": [
                "Banks led the market during yield curve steepening.",
                "Credit concerns caused a rapid reversal in bank stocks.",
            ],
            "insurers": [
                "Persistent rate rises boosted expectations for insurer unrealized gains.",
                "Valuation loss concerns eroded their relative advantage.",
            ],
            "utilities": [
                "Utilities provided stability when defensive flows dominated.",
                "Rate burden concerns pushed utilities lower in some periods.",
            ],
            "brokers": [
                "Rising trading volume previously lifted brokers in tandem.",
                "Market stagnation dampened earnings expectations for brokers.",
            ],
        },
    },
]


def generate_finance_case(
    task_id: str,
    difficulty: Difficulty = "medium",
    seed: int | None = None,
) -> FinanceCase:
    """1件分の finance ケースを生成する。"""
    rng = random.Random(seed)
    event_count = EVENT_COUNT_BY_DIFFICULTY[difficulty]
    initial_require_count = INITIAL_REQUIRE_COUNT_BY_DIFFICULTY[difficulty]

    theme = deepcopy(rng.choice(THEME_LIBRARY))
    stocks = list(theme["stocks"])
    current_stock_rates = _generate_stock_rates(rng=rng, stocks=stocks)
    current_stock_prices = _generate_stock_prices(rng=rng, stocks=stocks)
    hidden_inference_weights = _generate_hidden_inference_weights(rng=rng, stocks=stocks)
    hidden_resource = _build_hidden_resource(
        rng=rng,
        stocks=stocks,
        history=theme["history"],
    )
    initial_news = _compose_current_news(
        rng=rng,
        hidden_resource=hidden_resource,
        min_components=2,
        max_components=3,
    )
    efficient_rate_now = _compute_effective_rates(
        news=initial_news,
        hidden_inference_weights=hidden_inference_weights,
    )

    initial_state = _build_state(
        news=initial_news,
        stock_price_before=current_stock_prices,
        hidden_resource=hidden_resource,
        efficient_rate_now=efficient_rate_now,
    )

    initial_requires = _build_requires_from_state(initial_state)[:initial_require_count]
    initial_request = _build_initial_request(
        current_stock_rates=current_stock_rates,
        state=initial_state,
    )

    initial_do = FinanceDo(
        stock_rates=_recommend_stock_rates(
            current_stock_rates=current_stock_rates,
            efficient_rate_now=efficient_rate_now,
        )
    )

    events: list[FinanceEvent] = []
    now_rates = current_stock_rates
    now_prices = current_stock_prices
    weights = hidden_inference_weights

    for index in range(event_count):
        now_rates, now_prices, weights, event = _build_event(
            rng=rng,
            index=index,
            difficulty=difficulty,
            theme=theme,
            current_stock_rates=now_rates,
            current_stock_prices=now_prices,
            hidden_inference_weights=weights,
        )
        events.append(event)

    return FinanceCase(
        task_id=task_id,
        difficulty=difficulty,
        initial_request=initial_request,
        initial_state=initial_state,
        initial_do=initial_do,
        events=events,
        initial_requires=initial_requires,
    )


def generate_finance_cases(
    count: int,
    difficulty: Difficulty = "medium",
    seed: int | None = None,
    task_id_prefix: str = "finance",
) -> list[FinanceCase]:
    """複数の finance ケースをまとめて生成する。"""
    base_rng = random.Random(seed)
    return [
        generate_finance_case(
            task_id=f"{task_id_prefix}_{index + 1:03d}",
            difficulty=difficulty,
            seed=base_rng.randint(0, 10**9),
        )
        for index in range(count)
    ]


def write_finance_cases(
    output_dir: str | Path,
    count: int,
    difficulty: Difficulty = "medium",
    seed: int | None = None,
    task_id_prefix: str = "finance",
) -> list[Path]:
    """生成した finance ケース群を JSON ファイルとして保存する。"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    case_paths: list[Path] = []
    for case in generate_finance_cases(
        count=count,
        difficulty=difficulty,
        seed=seed,
        task_id_prefix=task_id_prefix,
    ):
        path = output_path / f"{case.task_id}.json"
        path.write_text(case.model_dump_json(indent=2), encoding="utf-8")
        case_paths.append(path)
    return case_paths


def parse_args() -> argparse.Namespace:
    """finance ケース生成 CLI 用の引数を解釈する。"""
    parser = argparse.ArgumentParser(description="Generate finance cases for the inaba models.")
    parser.add_argument("--count", type=int, default=5, help="Number of cases to generate.")
    parser.add_argument(
        "--difficulty",
        choices=("easy", "medium", "hard"),
        default="medium",
        help="Difficulty used for generated cases.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/generated/inaba/finance"),
        help="Directory where generated case JSON files are written.",
    )
    parser.add_argument(
        "--task-id-prefix",
        default="finance",
        help="Prefix used for generated task_id values.",
    )
    parser.add_argument(
        "--all-difficulties",
        action="store_true",
        help="Generate easy / medium / hard in sequence with difficulty suffix in task_id.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI から finance ケース生成を実行する。"""
    args = parse_args()
    difficulties: tuple[Difficulty, ...] = (
        ("easy", "medium", "hard") if args.all_difficulties else (args.difficulty,)
    )
    all_paths: list[Path] = []
    for difficulty in difficulties:
        prefix = (
            f"{args.task_id_prefix}_{difficulty}" if args.all_difficulties else args.task_id_prefix
        )
        paths = write_finance_cases(
            output_dir=args.output_dir,
            count=args.count,
            difficulty=difficulty,
            seed=args.seed,
            task_id_prefix=prefix,
        )
        all_paths.extend(paths)
    print(f"generated {len(all_paths)} finance cases into {args.output_dir}")
    for path in all_paths:
        print(path)


def _generate_stock_rates(
    rng: random.Random,
    stocks: list[str],
) -> dict[str, float]:
    """初期の持ち株比率を5銘柄で正規化して生成する。"""
    raw = {stock: rng.uniform(0.8, 1.2) for stock in stocks}
    return _normalize_rates(raw)


def _generate_hidden_inference_weights(
    rng: random.Random,
    stocks: list[str],
) -> dict[str, float]:
    """ニュースの inference_rate を内部でどれだけ強く効かせるかの補正係数を作る。"""
    return {stock: round(rng.uniform(0.6, 1.4), 4) for stock in stocks}


def _generate_stock_prices(
    rng: random.Random,
    stocks: list[str],
) -> dict[str, float]:
    """ニュース反映前の各株価を生成する。"""
    return {stock: round(rng.uniform(80.0, 220.0), 2) for stock in stocks}


def _normalize_rates(raw_rates: dict[str, float]) -> dict[str, float]:
    """比率の合計が1になるように正規化する。"""
    total = sum(raw_rates.values())
    return {stock: round(value / total, 4) for stock, value in raw_rates.items()}


def _build_hidden_resource(
    rng: random.Random,
    stocks: list[str],
    history: dict[str, list[str]],
) -> dict[str, NewsInfo]:
    """各分野について、隠し参照用の関連ニュース群を作る。"""
    hidden_resource: dict[str, NewsInfo] = {}
    for stock in stocks:
        hidden_resource[stock] = NewsInfo(
            content=str(rng.choice(history[stock])),
            inference_rate={
                name: round(rng.uniform(-1.0, 1.0), 4) for name in stocks
            },
            efficient_rate_change={
                name: round(rng.uniform(-0.25, 0.25), 4) for name in stocks
            },
        )
    return hidden_resource


def _compose_current_news(
    rng: random.Random,
    hidden_resource: dict[str, NewsInfo],
    min_components: int = 2,
    max_components: int = 3,
) -> CurrentNewsInfo:
    """複数分野の resource を何割ずつ混ぜるか決めて、現在ニュースを合成する。"""
    departments = list(hidden_resource.keys())
    component_count = min(len(departments), rng.randint(min_components, max_components))
    selected_departments = rng.sample(departments, k=component_count)

    raw_mix = {department: rng.uniform(0.2, 1.0) for department in selected_departments}
    raw_total = sum(raw_mix.values())
    normalized_abs_mix = {
        department: weight / raw_total
        for department, weight in raw_mix.items()
    }
    signed_mix_rate = {
        department: round(
            normalized_abs_mix[department] * rng.choice([-1.0, 1.0]),
            4,
        )
        for department in selected_departments
    }
    mix_rate = {
        department: signed_mix_rate.get(department, 0.0)
        for department in departments
    }

    stock_names = list(next(iter(hidden_resource.values())).inference_rate.keys())
    inference_rate = {
        stock: round(
            sum(
                hidden_resource[department].inference_rate.get(stock, 0.0) * mix_rate[department]
                for department in departments
            ),
            4,
        )
        for stock in stock_names
    }
    efficient_rate_change = {
        stock: round(
            sum(
                hidden_resource[department].efficient_rate_change.get(stock, 0.0)
                * mix_rate[department]
                for department in departments
            ),
            4,
        )
        for stock in stock_names
    }

    content = " / ".join(
        f"{department}: {hidden_resource[department].content}"
        for department in selected_departments
    )
    return CurrentNewsInfo(
        content=content,
        inference_rate=inference_rate,
        efficient_rate_change=efficient_rate_change,
        resource_mix_rate=mix_rate,
    )


def _compute_effective_rates(
    news: NewsInfo,
    hidden_inference_weights: dict[str, float],
) -> dict[str, float]:
    """見えない補正係数を使って、実際に効いている変動率を内部的に計算する。"""
    effective: dict[str, float] = {}
    for stock, inference in news.inference_rate.items():
        base_effect = news.efficient_rate_change.get(stock, 0.0)
        weight = hidden_inference_weights.get(stock, 1.0)
        effective[stock] = round(inference * weight + base_effect, 4)
    return effective


def _build_state(
    news: CurrentNewsInfo,
    stock_price_before: dict[str, float],
    hidden_resource: dict[str, NewsInfo],
    efficient_rate_now: dict[str, float],
    reveal_hidden: bool = False,
) -> FinanceState:
    """finance の完全情報・見えている情報・隠し情報を分けた State を作る。"""
    full_info = FinanceInfo(
        news=news,
        stock_price_before=stock_price_before,
        hidden_resource=hidden_resource,
        efficient_rate_now=efficient_rate_now,
    )
    return FinanceState(
        constraint=[
            "Only one news item is visible at a time",
            "Request related news for one sector at a time if needed",
            "Holdings are managed by ratio only",
        ],
        hidden_info=FinanceInfo(
            news=CurrentNewsInfo(
                content=news.content,
                inference_rate={} if reveal_hidden else news.inference_rate,
                efficient_rate_change={} if reveal_hidden else news.efficient_rate_change,
                resource_mix_rate={} if reveal_hidden else news.resource_mix_rate,
            ),
            stock_price_before={},
            hidden_resource={} if reveal_hidden else hidden_resource,
            efficient_rate_now={} if reveal_hidden else efficient_rate_now,
        ),
        info=full_info,
        got_info=FinanceInfo(
            news=CurrentNewsInfo(
                content=news.content,
                inference_rate=news.inference_rate if reveal_hidden else {},
                efficient_rate_change=news.efficient_rate_change if reveal_hidden else {},
                resource_mix_rate=news.resource_mix_rate if reveal_hidden else {},
            ),
            stock_price_before=stock_price_before,
            hidden_resource=hidden_resource if reveal_hidden else {},
            efficient_rate_now=efficient_rate_now if reveal_hidden else {},
        ),
    )


def _build_requires_from_state(state: FinanceState) -> list[str]:
    """hidden_resource から追加で確認すべき分野ニュース名を作る。"""
    return [
        f"Related news for {department} sector"
        for department in state.hidden_info.hidden_resource
    ]


def _build_initial_request(
    current_stock_rates: dict[str, float],
    state: FinanceState,
) -> str:
    """ユーザーに見せる初期依頼文を自然文で作る。"""
    rates_text = ", ".join(
        f"{stock}={rate:.2%}" for stock, rate in current_stock_rates.items()
    )
    prices_text = ", ".join(
        f"{stock}={price:.2f}" for stock, price in state.got_info.stock_price_before.items()
    )
    return (
        f"Current portfolio ratios: {rates_text}."
        f" Pre-news stock prices: {prices_text}."
        f" News: \"{state.got_info.news.content if state.got_info.news else ''}\"."
        " Decide which stock ratios to increase or decrease."
    )


def _build_event(
    rng: random.Random,
    index: int,
    difficulty: Difficulty,
    theme: dict[str, object],
    current_stock_rates: dict[str, float],
    current_stock_prices: dict[str, float],
    hidden_inference_weights: dict[str, float],
) -> tuple[dict[str, float], dict[str, float], dict[str, float], FinanceEvent]:
    """新しいニュースと内部前提の変化を含む event を1件生成する。"""
    stocks = list(theme["stocks"])
    focus_stock = stocks[index % len(stocks)]
    hidden_resource = _build_hidden_resource(
        rng=rng,
        stocks=stocks,
        history=theme["history"],
    )
    news = _compose_current_news(
        rng=rng,
        hidden_resource=hidden_resource,
        min_components=2,
        max_components=3,
    )

    updated_weights = deepcopy(hidden_inference_weights)
    impact_max = max(abs(rate) for rate in news.inference_rate.values())
    if impact_max >= 0.6:
        for stock in updated_weights:
            updated_weights[stock] = round(
                updated_weights[stock] * (1 + rng.uniform(-0.3, 0.3)),
                4,
            )

    efficient_rate_now = _compute_effective_rates(
        news=news,
        hidden_inference_weights=updated_weights,
    )
    next_stock_rates = _recommend_stock_rates(
        current_stock_rates=current_stock_rates,
        efficient_rate_now=efficient_rate_now,
    )
    state_before = _build_state(
        news=news,
        stock_price_before=current_stock_prices,
        hidden_resource=hidden_resource,
        efficient_rate_now=efficient_rate_now,
        reveal_hidden=False,
    )
    next_stock_prices = _apply_effective_rates_to_prices(
        stock_price_before=current_stock_prices,
        efficient_rate_now=efficient_rate_now,
    )

    message = (
        f"New news: {news.content}"
        f" Only material related to {focus_stock} is directly visible;"
        " related news for other sectors requires explicit inquiry."
    )

    return next_stock_rates, next_stock_prices, updated_weights, FinanceEvent(
        message=message,
        state_before=state_before,
        exp_do=FinanceDo(stock_rates=next_stock_rates),
        should_require=_build_requires_from_state(state_before)[: _question_count(difficulty)],
    )


def _recommend_stock_rates(
    current_stock_rates: dict[str, float],
    efficient_rate_now: dict[str, float],
) -> dict[str, float]:
    """内部の有効変動率をもとに次の持ち株比率を更新する。"""
    updated = {
        stock: max(
            0.01,
            current_stock_rates.get(stock, 0.0) * (1 + efficient_rate_now.get(stock, 0.0)),
        )
        for stock in current_stock_rates
    }
    return _normalize_rates(updated)


def _apply_effective_rates_to_prices(
    stock_price_before: dict[str, float],
    efficient_rate_now: dict[str, float],
) -> dict[str, float]:
    """現在ニュースの影響を反映して、次イベント開始時点の株価に更新する。"""
    return {
        stock: round(
            max(1.0, price * (1 + efficient_rate_now.get(stock, 0.0))),
            2,
        )
        for stock, price in stock_price_before.items()
    }


def _question_count(difficulty: Difficulty) -> int:
    """難易度ごとの追加確認項目数を返す。"""
    return INITIAL_REQUIRE_COUNT_BY_DIFFICULTY[difficulty]


if __name__ == "__main__":
    main()
