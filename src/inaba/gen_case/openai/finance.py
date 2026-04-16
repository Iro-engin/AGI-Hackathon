from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

from ...models.finance import (
    CurrentNewsInfo,
    FinanceCase,
    FinanceDo,
    FinanceEvent,
    FinanceInfo,
    FinanceState,
    NewsInfo,
)

load_dotenv()
Difficulty = Literal["easy", "medium", "hard"]

DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"

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

FINANCE_CONSTRAINTS = [
    "必要なら分野を1つずつ指定して関連ニュースを確認する",
    "持ち株は比率のみで判断する",
]

OPENAI_INSTRUCTIONS = """
あなたは金融ケース生成の補助エンジンです。
必ずJSONのみを返し、Markdownや説明文は一切出さないでください。
数値は実数で返し、指定した範囲を守ってください。
""".strip()


class ResourceDraft(BaseModel):
    """OpenAI が返す分野ごとの hidden_resource 下書き。"""

    department: str
    content: str
    inference_rate: dict[str, float] = Field(default_factory=dict)
    efficient_rate_change: dict[str, float] = Field(default_factory=dict)


class ResourcePlan(BaseModel):
    """分野一覧と hidden_resource の下書き。"""

    departments: list[str] = Field(default_factory=list)
    resources: list[ResourceDraft] = Field(default_factory=list)


class NewsDraft(BaseModel):
    """現在ニュースの本文と、内部で想定している関連分野。"""

    content: str
    related_departments: list[str] = Field(default_factory=list)


class NewsPlan(BaseModel):
    """initial と event 用ニュースの下書き集合。"""

    initial_news: NewsDraft
    event_news: list[NewsDraft] = Field(default_factory=list)


class MixRateDraft(BaseModel):
    """ニュース1件に対する分野ごとの混合率。"""

    resource_mix_rate: dict[str, float] = Field(default_factory=dict)


class MixRatePlan(BaseModel):
    """initial と event 用ニュースの混合率集合。"""

    initial_news: MixRateDraft
    event_news: list[MixRateDraft] = Field(default_factory=list)


def generate_finance_case(
    task_id: str,
    difficulty: Difficulty = "medium",
    seed: int | None = None,
    model: str = DEFAULT_OPENAI_MODEL,
) -> FinanceCase:
    """OpenAI を使って 1 件分の finance ケースを生成する。"""
    rng = random.Random(seed)
    event_count = EVENT_COUNT_BY_DIFFICULTY[difficulty]
    initial_require_count = INITIAL_REQUIRE_COUNT_BY_DIFFICULTY[difficulty]

    resource_plan = _generate_resource_plan(
        model=model,
        difficulty=difficulty,
        seed=seed,
    )
    departments = _normalize_departments(resource_plan.departments)
    hidden_resource = _build_hidden_resource_from_plan(
        departments=departments,
        resource_plan=resource_plan,
    )

    news_plan = _generate_news_plan(
        model=model,
        difficulty=difficulty,
        seed=seed,
        departments=departments,
        hidden_resource=hidden_resource,
        event_count=event_count,
    )
    news_plan = _normalize_news_plan(
        news_plan=news_plan,
        departments=departments,
        event_count=event_count,
    )

    mix_rate_plan = _generate_mix_rate_plan(
        model=model,
        difficulty=difficulty,
        seed=seed,
        departments=departments,
        hidden_resource=hidden_resource,
        news_plan=news_plan,
    )
    mix_rate_plan = _normalize_mix_rate_plan(
        mix_rate_plan=mix_rate_plan,
        departments=departments,
        event_count=event_count,
    )

    current_stock_rates = _generate_stock_rates(rng=rng, stocks=departments)
    current_stock_prices = _generate_stock_prices(rng=rng, stocks=departments)
    hidden_inference_weights = _generate_hidden_inference_weights(rng=rng, stocks=departments)

    initial_news = _build_current_news(
        news_draft=news_plan.initial_news,
        mix_rate=mix_rate_plan.initial_news.resource_mix_rate,
        hidden_resource=hidden_resource,
    )
    initial_effective_rate_now = _compute_effective_rates(
        news=initial_news,
        hidden_inference_weights=hidden_inference_weights,
    )
    initial_state = _build_state(
        news=initial_news,
        stock_price_before=current_stock_prices,
        hidden_resource=hidden_resource,
        efficient_rate_now=initial_effective_rate_now,
    )

    initial_requires = _build_requires_from_state(initial_state)[:initial_require_count]
    initial_request = _build_initial_request(
        current_stock_rates=current_stock_rates,
        state=initial_state,
    )

    events: list[FinanceEvent] = []
    now_rates = current_stock_rates
    now_prices = current_stock_prices
    now_weights = hidden_inference_weights

    for news_draft, mix_draft in zip(
        news_plan.event_news,
        mix_rate_plan.event_news,
        strict=True,
    ):
        event_news = _build_current_news(
            news_draft=news_draft,
            mix_rate=mix_draft.resource_mix_rate,
            hidden_resource=hidden_resource,
        )
        now_rates, now_prices, now_weights, event = _build_event(
            rng=rng,
            difficulty=difficulty,
            current_stock_rates=now_rates,
            current_stock_prices=now_prices,
            hidden_inference_weights=now_weights,
            hidden_resource=hidden_resource,
            news=event_news,
        )
        events.append(event)

    return FinanceCase(
        task_id=task_id,
        difficulty=difficulty,
        initial_request=initial_request,
        initial_state=initial_state,
        events=events,
        initial_requires=initial_requires,
    )


def generate_finance_cases(
    count: int,
    difficulty: Difficulty = "medium",
    seed: int | None = None,
    task_id_prefix: str = "finance_openai",
    model: str = DEFAULT_OPENAI_MODEL,
) -> list[FinanceCase]:
    """OpenAI を使って finance ケースを複数生成する。"""
    base_rng = random.Random(seed)
    return [
        generate_finance_case(
            task_id=f"{task_id_prefix}_{index + 1:03d}",
            difficulty=difficulty,
            seed=base_rng.randint(0, 10**9),
            model=model,
        )
        for index in range(count)
    ]


def write_finance_cases(
    output_dir: str | Path,
    count: int,
    difficulty: Difficulty = "medium",
    seed: int | None = None,
    task_id_prefix: str = "finance_openai",
    model: str = DEFAULT_OPENAI_MODEL,
) -> list[Path]:
    """OpenAI で生成した finance ケース群を JSON ファイルとして保存する。"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    case_paths: list[Path] = []
    for case in generate_finance_cases(
        count=count,
        difficulty=difficulty,
        seed=seed,
        task_id_prefix=task_id_prefix,
        model=model,
    ):
        path = output_path / f"{case.task_id}.json"
        path.write_text(case.model_dump_json(indent=2), encoding="utf-8")
        case_paths.append(path)
    return case_paths


def parse_args() -> argparse.Namespace:
    """OpenAI 版 finance ケース生成 CLI 用の引数を解釈する。"""
    parser = argparse.ArgumentParser(description="Generate finance cases with OpenAI.")
    parser.add_argument("--count", type=int, default=5, help="Number of cases to generate.")
    parser.add_argument(
        "--difficulty",
        choices=("easy", "medium", "hard"),
        default="medium",
        help="Difficulty used for generated cases.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed.")
    parser.add_argument(
        "--model",
        default=DEFAULT_OPENAI_MODEL,
        help="OpenAI model name used for staged generation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/generated/inaba/finance_openai"),
        help="Directory where generated case JSON files are written.",
    )
    parser.add_argument(
        "--task-id-prefix",
        default="finance_openai",
        help="Prefix used for generated task_id values.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI から OpenAI 版 finance ケース生成を実行する。"""
    args = parse_args()
    paths = write_finance_cases(
        output_dir=args.output_dir,
        count=args.count,
        difficulty=args.difficulty,
        seed=args.seed,
        task_id_prefix=args.task_id_prefix,
        model=args.model,
    )
    print(f"generated {len(paths)} finance cases into {args.output_dir}")
    for path in paths:
        print(path)


def _call_openai_json(model: str, prompt: str) -> dict[str, Any]:
    """OpenAI Responses API を呼び出し、JSON オブジェクトとして解釈する。"""
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    client = OpenAI()
    response = client.responses.create(
        model=model,
        instructions=OPENAI_INSTRUCTIONS,
        input=prompt,
    )
    return _parse_json_text(response.output_text)


def _parse_json_text(text: str) -> dict[str, Any]:
    """JSON 文字列だけを安全側に切り出して辞書へ変換する。"""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return json.loads(stripped)


def _generate_resource_plan(
    model: str,
    difficulty: Difficulty,
    seed: int | None,
) -> ResourcePlan:
    """Step1: hidden_resource と分野一覧を OpenAI に生成させる。"""
    prompt = f"""
難易度 {difficulty} の finance ケースを作ります。乱数シードの参照値は {seed} です。

以下を満たす JSON を返してください。
- 分野はちょうど5個
- 分野IDは英小文字の snake_case
- 分野どうしは近すぎないこと。たとえば半導体と半導体製造装置のような近接は避ける
- 各分野について hidden_resource 用の過去ニュースを1件ずつ作る
- content は日本語1文から2文
- inference_rate は各分野IDに対する影響度で、各値は -1 から 1
- efficient_rate_change は各分野IDに対する効き方の変化量で、各値は -0.25 から 0.25
- resources の各要素は department, content, inference_rate, efficient_rate_change を持つ
- inference_rate と efficient_rate_change のキーは5分野すべてを含める

返却形式:
{{
  "departments": ["...", "...", "...", "...", "..."],
  "resources": [
    {{
      "department": "...",
      "content": "...",
      "inference_rate": {{"d1": 0.1, "d2": -0.2, "d3": 0.0, "d4": 0.4, "d5": -0.1}},
      "efficient_rate_change": {{"d1": 0.02, "d2": -0.03, "d3": 0.0, "d4": 0.08, "d5": -0.01}}
    }}
  ]
}}
""".strip()
    return ResourcePlan.model_validate(_call_openai_json(model=model, prompt=prompt))


def _generate_news_plan(
    model: str,
    difficulty: Difficulty,
    seed: int | None,
    departments: list[str],
    hidden_resource: dict[str, NewsInfo],
    event_count: int,
) -> NewsPlan:
    """Step2: initial と event 用の現在ニュース本文を OpenAI に生成させる。"""
    prompt = f"""
難易度 {difficulty} の finance ケースを作ります。乱数シードの参照値は {seed} です。
分野一覧は {json.dumps(departments, ensure_ascii=False)} です。

以下を満たす JSON を返してください。
- initial_news を1件、event_news を {event_count} 件
- すべて日本語の自然なニュース文にする
- ニュース文に分野IDをそのまま書かない
- 「○○分野では」のように答えを露骨に示さない
- news は「1分野のニュース文を2本または3本つないだ形式」にしてください
- つまり 1 本の長い複合ニュースではなく、
  1分野に対応する短文を並べた multi headline 形式にしてください
- 先に related_departments を2個または3個決めてください
- 本文は related_departments の各分野に対して、
  1文ずつ対応する短いニュースを書く形にしてください
- 各文は基本的に1分野だけを自然に示す内容にしてください
- related_departments に入れた分野は、
  その分野に対応する文から直接わかるようにしてください
- related_departments に含めない分野は、
  本文から連想しにくいようにしてください
- 3個にすると不自然なら2個にしてください。無理に3個選ばないでください
- related_departments はなるべく近すぎない組み合わせにしてください
- ただし「近すぎない」ことより「本文との整合性」を優先してください
- resources の文章表現や言い回しは参照しないでください
- resources に引っ張られず、ニュース本文は独立に新しく書いてください
- event_news 同士は話題が被りすぎないようにする

悪い例:
- 本文:
  国内の主要銀行は、安定した資金調達環境を背景に、
  法人向けローンの金利を据え置くことを決定しました。
  related_departments: ["banks", "reit"]
  理由: 1文しかなく、reit に対応する文がない

良い例:
- 本文:
  政府はIT事業への投資を拡大する方針を発表しました。
  一方で、木材需要はコロナ後の反動で弱含んでいます。
  related_departments: ["it_services", "forestry"]
  理由: 2つの文がそれぞれ1分野ずつ自然に示している

返却形式:
{{
  "initial_news": {{
    "content": "...",
    "related_departments": ["...", "..."]
  }},
  "event_news": [
    {{
      "content": "...",
      "related_departments": ["...", "...", "..."]
    }}
  ]
}}
""".strip()
    return NewsPlan.model_validate(_call_openai_json(model=model, prompt=prompt))


def _generate_mix_rate_plan(
    model: str,
    difficulty: Difficulty,
    seed: int | None,
    departments: list[str],
    hidden_resource: dict[str, NewsInfo],
    news_plan: NewsPlan,
) -> MixRatePlan:
    """Step3: 各ニュースと各分野の相関度を OpenAI に生成させる。"""
    resource_summary = {
        department: {
            "content": hidden_resource[department].content,
            "inference_rate": hidden_resource[department].inference_rate,
            "efficient_rate_change": hidden_resource[department].efficient_rate_change,
        }
        for department in departments
    }
    prompt = f"""
難易度 {difficulty} の finance ケースを作ります。乱数シードの参照値は {seed} です。
分野一覧は {json.dumps(departments, ensure_ascii=False)} です。
hidden_resource は {json.dumps(resource_summary, ensure_ascii=False)} です。
ニュース一覧は {json.dumps(news_plan.model_dump(), ensure_ascii=False)} です。

以下を満たす JSON を返してください。
- initial_news と event_news のそれぞれについて resource_mix_rate を返す
- resource_mix_rate は全5分野のキーを必ず含める
- 値は -1 から 1
- 無関係なら 0 でよい
- related_departments は強めの非ゼロにする
- hidden_resource の内容とニュースの向きが逆なら負の値を使ってよい
- 極端な1点集中より、関連度に応じて複数分野へ配分する

返却形式:
{{
  "initial_news": {{
    "resource_mix_rate": {{"d1": 0.7, "d2": -0.2, "d3": 0.0, "d4": 0.1, "d5": 0.0}}
  }},
  "event_news": [
    {{
      "resource_mix_rate": {{"d1": 0.0, "d2": 0.5, "d3": -0.4, "d4": 0.1, "d5": 0.0}}
    }}
  ]
}}
""".strip()
    return MixRatePlan.model_validate(_call_openai_json(model=model, prompt=prompt))


def _normalize_departments(raw_departments: list[str]) -> list[str]:
    """分野IDを5件の snake_case に正規化する。"""
    normalized: list[str] = []
    for department in raw_departments:
        cleaned = _normalize_department_id(department)
        if cleaned and cleaned not in normalized:
            normalized.append(cleaned)

    fallback_pool = [
        "defense_electronics",
        "luxury_hotels",
        "industrial_gases",
        "pharmacy_retail",
        "grain_shipping",
    ]
    for department in fallback_pool:
        if len(normalized) >= 5:
            break
        if department not in normalized:
            normalized.append(department)
    return normalized[:5]


def _normalize_department_id(value: str) -> str:
    """分野IDを snake_case 風の内部表現にそろえる。"""
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def _build_hidden_resource_from_plan(
    departments: list[str],
    resource_plan: ResourcePlan,
) -> dict[str, NewsInfo]:
    """OpenAI が返した resource 下書きを hidden_resource へ変換する。"""
    resources_by_department = {
        _normalize_department_id(resource.department): resource
        for resource in resource_plan.resources
    }
    hidden_resource: dict[str, NewsInfo] = {}

    for department in departments:
        resource = resources_by_department.get(department)
        if resource is None:
            hidden_resource[department] = NewsInfo(
                content=f"{department} に関する過去の材料。",
                inference_rate={name: 0.0 for name in departments},
                efficient_rate_change={name: 0.0 for name in departments},
            )
            continue

        hidden_resource[department] = NewsInfo(
            content=resource.content,
            inference_rate=_normalize_signed_map(
                raw_map=resource.inference_rate,
                keys=departments,
                lower=-1.0,
                upper=1.0,
            ),
            efficient_rate_change=_normalize_signed_map(
                raw_map=resource.efficient_rate_change,
                keys=departments,
                lower=-0.25,
                upper=0.25,
            ),
        )
    return hidden_resource


def _normalize_news_plan(
    news_plan: NewsPlan,
    departments: list[str],
    event_count: int,
) -> NewsPlan:
    """ニュース件数と related_departments をケース仕様に合わせて整える。"""
    initial_news = _normalize_news_draft(news_plan.initial_news, departments)
    event_news = [
        _normalize_news_draft(news_draft, departments)
        for news_draft in news_plan.event_news[:event_count]
    ]

    while len(event_news) < event_count:
        base = departments[len(event_news) % len(departments)]
        partner = departments[(len(event_news) + 2) % len(departments)]
        event_news.append(
            NewsDraft(
                content=(
                    "新しい投資判断を受けて市場参加者の見方が変わっています。"
                    " 一方で、別の業界では需要見通しの弱さが意識されています。"
                ),
                related_departments=[base, partner],
            )
        )

    return NewsPlan(initial_news=initial_news, event_news=event_news)


def _normalize_news_draft(
    news_draft: NewsDraft,
    departments: list[str],
) -> NewsDraft:
    """ニュース本文と関連分野の候補を正規化する。"""
    normalized_related = (
        _normalize_department_id(value) for value in news_draft.related_departments
    )
    related = [department for department in normalized_related if department in departments]
    if len(related) < 2:
        related = departments[:2]
    if len(related) > 3:
        related = related[:3]

    content = news_draft.content.strip() or "需給見通しに影響する新しい材料が出ている。"
    for department in departments:
        content = content.replace(f"{department}:", "")
        content = content.replace(department, "")
    return NewsDraft(content=content.strip(), related_departments=related)


def _normalize_mix_rate_plan(
    mix_rate_plan: MixRatePlan,
    departments: list[str],
    event_count: int,
) -> MixRatePlan:
    """resource_mix_rate を全分野・全ニュース分そろえて正規化する。"""
    initial_news = MixRateDraft(
        resource_mix_rate=_normalize_signed_map(
            raw_map=mix_rate_plan.initial_news.resource_mix_rate,
            keys=departments,
            lower=-1.0,
            upper=1.0,
        )
    )
    event_news = [
        MixRateDraft(
            resource_mix_rate=_normalize_signed_map(
                raw_map=mix_draft.resource_mix_rate,
                keys=departments,
                lower=-1.0,
                upper=1.0,
            )
        )
        for mix_draft in mix_rate_plan.event_news[:event_count]
    ]

    while len(event_news) < event_count:
        event_news.append(
            MixRateDraft(resource_mix_rate={department: 0.0 for department in departments})
        )
    return MixRatePlan(initial_news=initial_news, event_news=event_news)


def _build_current_news(
    news_draft: NewsDraft,
    mix_rate: dict[str, float],
    hidden_resource: dict[str, NewsInfo],
) -> CurrentNewsInfo:
    """Step4: resource_mix_rate と hidden_resource から現在ニュースの各 rate を計算する。"""
    departments = list(hidden_resource.keys())
    inference_rate = {
        stock: round(
            sum(
                hidden_resource[department].inference_rate.get(stock, 0.0)
                * mix_rate.get(department, 0.0)
                for department in departments
            ),
            4,
        )
        for stock in departments
    }
    efficient_rate_change = {
        stock: round(
            sum(
                hidden_resource[department].efficient_rate_change.get(stock, 0.0)
                * mix_rate.get(department, 0.0)
                for department in departments
            ),
            4,
        )
        for stock in departments
    }
    return CurrentNewsInfo(
        content=news_draft.content,
        inference_rate=_normalize_signed_map(
            raw_map=inference_rate,
            keys=departments,
            lower=-1.0,
            upper=1.0,
        ),
        efficient_rate_change=_normalize_signed_map(
            raw_map=efficient_rate_change,
            keys=departments,
            lower=-0.3,
            upper=0.3,
        ),
        resource_mix_rate=_normalize_signed_map(
            raw_map=mix_rate,
            keys=departments,
            lower=-1.0,
            upper=1.0,
        ),
    )


def _generate_stock_rates(
    rng: random.Random,
    stocks: list[str],
) -> dict[str, float]:
    """初期の持ち株比率を5銘柄で正規化して生成する。"""
    raw = {stock: rng.uniform(0.8, 1.2) for stock in stocks}
    return _normalize_rates(raw)


def _generate_stock_prices(
    rng: random.Random,
    stocks: list[str],
) -> dict[str, float]:
    """ニュース反映前の各株価を生成する。"""
    return {stock: round(rng.uniform(80.0, 220.0), 2) for stock in stocks}


def _generate_hidden_inference_weights(
    rng: random.Random,
    stocks: list[str],
) -> dict[str, float]:
    """inference_rate を内部でどれだけ効かせるかの補正係数を作る。"""
    return {stock: round(rng.uniform(0.6, 1.4), 4) for stock in stocks}


def _normalize_rates(raw_rates: dict[str, float]) -> dict[str, float]:
    """比率の合計が1になるように正規化する。"""
    total = sum(raw_rates.values()) or 1.0
    return {stock: round(value / total, 4) for stock, value in raw_rates.items()}


def _normalize_signed_map(
    raw_map: dict[str, float],
    keys: list[str],
    lower: float,
    upper: float,
) -> dict[str, float]:
    """符号付き実数マップをキー集合と値範囲に合わせてそろえる。"""
    normalized: dict[str, float] = {}
    for key in keys:
        value = float(raw_map.get(key, 0.0))
        normalized[key] = round(max(lower, min(upper, value)), 4)
    return normalized


def _compute_effective_rates(
    news: NewsInfo,
    hidden_inference_weights: dict[str, float],
) -> dict[str, float]:
    """見えない補正係数を使って、実際に効いている変動率を内部的に計算する。"""
    effective: dict[str, float] = {}
    for stock, inference in news.inference_rate.items():
        base_effect = news.efficient_rate_change.get(stock, 0.0)
        weight = hidden_inference_weights.get(stock, 1.0)
        effective[stock] = round(max(-0.6, min(0.6, inference * weight + base_effect)), 4)
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
        constraint=FINANCE_CONSTRAINTS,
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
    """resource_mix_rate != 0 の分野に限定して確認すべき分野ニュース名を作る。"""
    mix_rate = state.hidden_info.news.resource_mix_rate if state.hidden_info.news else {}
    return [
        f"{department}分野の関連ニュース"
        for department in state.hidden_info.hidden_resource
        if mix_rate.get(department, 0.0) != 0.0
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
        f"現在の持ち株比率は {rates_text} です。"
        f" ニュース反映前の株価は {prices_text} です。"
        f" ニュースは「{state.got_info.news.content if state.got_info.news else ''}」。"
        " どの株の比率を上げ下げすべきか判断してください。"
    )


def _build_event(
    rng: random.Random,
    difficulty: Difficulty,
    current_stock_rates: dict[str, float],
    current_stock_prices: dict[str, float],
    hidden_inference_weights: dict[str, float],
    hidden_resource: dict[str, NewsInfo],
    news: CurrentNewsInfo,
) -> tuple[dict[str, float], dict[str, float], dict[str, float], FinanceEvent]:
    """現在ニュースと内部前提の変化を使って event を1件組み立てる。"""
    updated_weights = dict(hidden_inference_weights)
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

    return next_stock_rates, next_stock_prices, updated_weights, FinanceEvent(
        message=f"新しいニュース: {news.content}",
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
        stock: round(max(1.0, price * (1 + efficient_rate_now.get(stock, 0.0))), 2)
        for stock, price in stock_price_before.items()
    }


def _question_count(difficulty: Difficulty) -> int:
    """難易度ごとの追加確認項目数を返す。"""
    return INITIAL_REQUIRE_COUNT_BY_DIFFICULTY[difficulty]


if __name__ == "__main__":
    main()
