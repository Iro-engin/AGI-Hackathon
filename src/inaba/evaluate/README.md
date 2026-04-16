# inaba evaluate

agent の実行ログ（`execute` フェーズの出力）を採点し、定量的な評価指標を算出するモジュール。

---

## 評価指標一覧

評価指標は **Planning Score**（計画の質）と **Impulse Control Score**（衝動制御の質）の 2 カテゴリに分かれる。

---

### Planning Score

#### Final Pass Rate（最終プラン実行可能率）

> 最終的に提出した Do が、全制約を満たし実行可能であった割合。

$$
\text{Final Pass Rate} = \frac{\text{feasible な Do を出したターン数}}{\text{全ターン数}}
$$

- **meeting**: 提案した (where, when\_from, when\_to) が全参加者の都合・所要時間制約を満たすか → OpenAI で判定
- **finance**: `stock_rates` が全銘柄非負・合計 1.0 ± 0.05 に収まるか → ルールベースで判定

---

#### Exact Match（正解完全一致率）

> 最終的に提出した Do が、正解の Do と完全一致（または許容誤差内で一致）した割合。

$$
\text{Exact Match} = \frac{\text{正解と一致した Do のターン数}}{\text{全ターン数}}
$$

- **meeting**: where / when\_from / when\_to がすべて等値一致
- **finance**: 各銘柄の `stock_rates` 差がすべて 0.05 以内

初期ターン（event なし）は正解 Do が存在しないため False 扱い。

---

### Impulse Control Score

#### Vague Ask Rate（曖昧即答率）

> `should_require`（聞くべき項目）が残っているにもかかわらず、質問せずに Do を返してしまったターンの割合。

$$
\text{Vague Ask Rate} = \frac{\text{未カバーのまま Do したターン数}}{\text{should\_require が非空なターン数}}
$$

高いほど「必要な確認を怠って即答してしまう」傾向が強い。

---

#### Rubric Score（必要質問カバレッジ）

> Do を返す前に `should_require` の項目をどれだけ質問でカバーできたかの平均割合。

$$
\text{Rubric Score} = \frac{1}{N} \sum_{i=1}^{N} \frac{|\text{covered\_requires}_i|}{|\text{should\_require}_i|}
$$

- `should_require` が空のターンは 1.0 として扱う
- 各質問が `should_require` のどの項目をカバーするかは **OpenAI で判定**（`judge_require_coverage`）

---

#### Redundant Question Rate（冗長質問率）

> 同一ターン内で、すでに他の質問がカバー済みの項目しかカバーしない質問（冗長な質問）の割合。

$$
\text{Redundant Question Rate} = \frac{\text{全ターン合計の冗長質問数}}{\text{全ターン合計の質問数}}
$$

冗長質問の定義：その質問がカバーする `should_require` 項目が、同ターン内の**それ以前の質問**によってすでに全てカバー済みであるもの。

---

#### Early Discovery Rate（早期発見率）

> `should_require` の全項目を、**最初の N 問以内**に完了できたターンの割合。

$$
\text{Early Discovery Rate} = \frac{\text{最初 N 問で全 should\_require をカバーしたターン数}}{\text{should\_require が非空なターン数}}
$$

- デフォルト N = 2（`--early-discovery-n` で変更可）
- 何問目の質問がどの `should_require` をカバーしたかは `covered_per_question`（質問ごとのカバレッジリスト）から計算する

---

## 計算フロー

```
cases_dir/*.json          executions_dir/*.evaluation.json
       │                              │
       ▼                              ▼
  task_id → Case マップ         EvaluationRun（実行ログ）
       │                              │
       └──────────── 突き合わせ ──────┘
                         │
                         ▼
              build_turn_records()
              ┌─────────────────────────────────────┐
              │ 各ターンで:                          │
              │  1. questions_asked を取り出す       │
              │  2. judge_require_coverage (OpenAI)  │
              │     → covered_per_question を構築    │
              │  3. 冗長質問を検出                   │
              │  4. is_feasible / is_exact_match     │
              └─────────────────────────────────────┘
                         │
                         ▼
              compute_eval_result()
              → EvalResult（6 指標）
                         │
                         ▼
         {task_id}.score.json  +  _summary.json
```

---

## 出力ファイル

### `{task_id}.score.json`

1 ケース分の詳細スコア。`turns` 配列に各ターンの質問・カバレッジ・feasibility が記録される。

### `_summary.json`

全ケースの平均スコアをまとめた集計ファイル。

```json
{
  "domain": "finance",
  "case_count": 5,
  "early_discovery_n": 2,
  "average_scores": {
    "final_pass_rate": 0.9,
    "exact_match": 0.2,
    "vague_ask_rate": 0.4,
    "rubric_score": 0.6,
    "redundant_question_rate": 0.1,
    "early_discovery_rate": 0.5
  }
}
```

---

## 実行方法

```bash
# meeting
make eval-inaba-meeting

# finance
make eval-inaba-finance

# パラメータ上書き例
make eval-inaba-finance \
  INABA_FINANCE_SCORE_EXEC_DIR=results/execute/inaba/finance_openai \
  INABA_SCORE_EARLY_DISCOVERY_N=3 \
  INABA_SCORE_LIMIT=10
```
