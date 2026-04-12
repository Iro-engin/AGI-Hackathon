# AGI Hackathon Benchmark

## これは何か

このリポジトリは、Kaggle の `Measuring AGI` を意識して、
「AI エージェントが途中の状況変化に対して、正しく質問し、正しくタスク分解し、
正しい順番で実行し、必要に応じて再計画できるか」を評価するための
最小ベンチマーク実装です。

題材としては、まず `meeting_prep` ドメインを使っており、
次の拡張先として `finance_analyst` ドメインも含めています。

想定している流れは次のようなものです。

1. 会議準備を依頼する
2. エージェントが質問やタスク分解を行う
3. 資料やアジェンダを作り始める
4. 途中で参加者や締切などの条件が変わる
5. エージェントが変更を認識して再計画する
6. 最終成果物が最新状態に追従しているかを評価する

## このリポジトリでやりたいこと

この実装でやりたいことは、単に最終成果物の良し悪しを採点することではありません。
途中の意思決定プロセスまで含めて、エージェントの挙動の正当性を見たい、というのが主目的です。

特に見たい観点は次のとおりです。

- とある出力構造に沿って、適切に計画を出力できているか(計画立案)
- 制限事項に沿っており、かつ順序立てられた計画を立案できるか(計画立案)
- 不確定な情報を理解し、質問できるか(計画立案)
- 制約事項の意味を確実に捉えての立案になっているか(衝動抑制＝「どうせこうやろ」で話を進めない力)
    - ex.「Tomは普段木曜日は空いている」という情報があったとして、AIは「Tomは今週木曜日の13:00~14:00は空いてる」と勝手に解釈せず、「Tomは今週木曜日の13:00~14:00は空いてる？」と聞ける
- 前提変更時に適切に再計画できるか(認知的柔軟性)

## 現状できること

現時点で、このリポジトリでできることは次のとおりです。

- `Case` を Pydantic で検証付き読み込みできる
- execution log を Pydantic で検証付き読み込みできる
- ルールベース evaluator で sample を採点できる
- OpenAI API で case / execution log を検査できる
- notebook を再生成できる
- `LOG_LEVEL` でログレベルを切り替えながら実行できる
- `make` でローカル実行できる

## ディレクトリ構成

- [`src/models.py`](src/models.py)
  case / execution log の型定義
- [`src/rule_evaluator.py`](src/rule_evaluator.py)
  ルールベース採点ロジック
- [`src/build_notebook.py`](src/build_notebook.py)
  Kaggle 提出用 notebook 生成
- [`src/logging_config.py`](src/logging_config.py)
  共通ログ設定
- [`scenarios/meeting/`](scenarios/meeting/)
  benchmark case の JSON
- [`scenarios/finance/`](scenarios/finance/)
  finance analyst 用 case の JSON
- [`results/sample_execution_meeting_001.json`](results/sample_execution_meeting_001.json)
  sample execution log
- [`schemas/benchmark_case.schema.json`](schemas/benchmark_case.schema.json)
  case 用 JSON Schema
- [`schemas/execution_log.schema.json`](schemas/execution_log.schema.json)
  execution log 用 JSON Schema
- [`Makefile`](Makefile)
  実行ショートカット

## 評価の考え方

このリポジトリの evaluator は、逐次対話の生ログを直接採点するのではなく、
`Case + ExecutionLog` を入力として採点します。

- `Case`
  問題設定、初期状態、イベント、成功条件を持つ
- `ExecutionLog`
  エージェントの行動を評価用に構造化した記録を持つ

そのため、現状の評価は
「会話生成そのもの」よりも、
「構造化ログ上で状態追跡・計画・再計画が整合しているか」
を見る設計になっています。

### 1. 質問

初期情報が足りないときに、確認が必要な事項を先に質問できているかを見ます。

execution log では `questions_asked` を使います。
ここには次を入れる想定です。

- 何を質問したか
- なぜ質問したか
- その質問によって停止しているタスクは何か

### 2. タスク分解

仕事を適切な粒度で分解できているかを見ます。

execution log では `task_breakdown` を使います。
ここには次を入れる想定です。

- タスク ID
- タスク説明
- 依存先
- ステータス

### 3. タスクの実行順番

分解しただけでなく、依存関係に従って正しい順番で進められているかを見ます。

case 側では `initial_state.task_dependencies` を使い、
execution log 側では `completed_tasks` を使います。

### 4. 状態変化への追従

途中イベントが起きたときに、変更を認識し、一定ターン以内で再計画できるかを見ます。

case 側では `events` に次のような情報を入れます。

- 何ターン目に変化したか
- どんな変化か
- state にどう効くか
- 何ターン以内に再計画すべきか
- どの成果物に影響するか

現状の turn は、厳密な会話ターンというより
`actions` の時系列ステップ番号として扱っています。

### 5. 最終成果物の整合性

最終的に出した成果物が、最新の state を正しく反映しているかを見ます。

case 側では `required_artifacts` と `semantic_checks` を使い、
execution log 側では `final_artifacts` を使います。

## 入力データの考え方

### Benchmark case

case は「問題設定」です。

主な要素:

- `task_id`
- `domain`
- `difficulty`
- `initial_request`
- `initial_state`
- `allowed_actions`
- `events`
- `goal_condition`
- `rubric`

特に重要なのは `initial_state` と `events` です。

- `initial_state`
  初期の参加者、締切、予算、制約、依存関係、必要成果物を持つ
- `events`
  途中で起きる変化を持つ

### Execution log

execution log は「エージェントがどう動いたか」の記録です。

主な要素:

- `actions`
- `questions_asked`
- `task_breakdown`
- `completed_tasks`
- `constraint_violations`
- `unsafe_commit`
- `final_state`
- `final_artifacts`

## 採点ロジックの考え方

[`src/rule_evaluator.py`](src/rule_evaluator.py) では、主に次の3軸で採点します。

- `outcome_score`
  最終成果物と最終 state の整合性
- `process_score`
  質問、タスク分解、依存順、許可 action などのプロセス品質
- `recovery_score`
  状態変化後の再計画と回復の品質

補助的に `failure_labels` と `deductions` も返します。

## 現状の保証範囲と未成熟な部分

現状の rule-based evaluator が比較的安定して見られるのは次です。

- 許可 action 違反
- タスク依存順序違反
- 必須成果物の欠落
- 成果物の必須項目欠落
- 成果物と最終 state の単純な整合性
- イベント後の状態確認と再計画の有無

一方で、次はまだ粗い、もしくは今後拡張したい領域です。

- 何を質問すべきだったかの厳密評価
- 冗長質問や曖昧質問の評価
- 再計画の質そのものの評価
- 成果物本文の意味内容評価
- 生の会話から execution log への標準変換

このため、現状は
「構造化ログに対する最小の整合性評価器」
として理解するのが適切です。

## notebook の位置づけ

[`src/kaggle_submission_benchmark.ipynb`](src/kaggle_submission_benchmark.ipynb) は、
Kaggle 提出や説明用の notebook です。

この notebook は次を含みます。

- helper
- case summary
- case 読み込み
- sample execution log 読み込み
- Pydantic models
- evaluator
- validation demo
- evaluation demo

注意点:

- notebook は巨大な `MEETING_CASES = [...]` を埋め込んでいません
- 実行時に `scenarios/meeting/*.json` と `results/sample_execution_meeting_001.json` を読み込みます

## ローカル実行方法

### まず環境を作る

```bash
make setup
```

これは内部的に `uv sync` を実行し、依存関係は `pyproject.toml` から解決します。

`requirements.txt` も必要な場合は、次で `uv` 定義から再生成できます。

```bash
make export-requirements
```

### notebook を再生成する

```bash
make build-notebook
```

### sample を評価する

```bash
make eval-sample
```

### ログレベルを指定して実行する

```bash
LOG_LEVEL=DEBUG make build-notebook
LOG_LEVEL=DEBUG make eval-sample
```

## Python から直接使う例

```python
from pathlib import Path

from src.logging_config import configure_logging
from src.models import Case, ExecutionLog
from src.rule_evaluator import RuleBasedEvaluator

configure_logging()

case = Case.from_path(Path("scenarios/meeting/meeting_001.json"))
log = ExecutionLog.from_path(Path("results/sample_execution_meeting_001.json"))
result = RuleBasedEvaluator().evaluate(case, log)

print(result)
```

## OpenAI API で検査する

`OPENAI_API_KEY` を設定したうえで、case 単体または execution log 付きで検査できます。

```bash
OPENAI_API_KEY=... make inspect-case
OPENAI_API_KEY=... uv run python -m src.openai_case_inspector \
  --case scenarios/meeting/meeting_001.json \
  --execution-log results/sample_execution_meeting_001.json
```

必要に応じて `OPENAI_MODEL` か `--model` で使用モデルを上書きできます。

## 現状の設計方針

- JSON はそのまま資産として残す
- Python 側では Pydantic で型安全に扱う
- `uv` を標準の環境管理として使い、必要なら `requirements.txt` へ書き出せるようにする
- 最小変更で notebook と evaluator の両方を維持する
- 最終成果物だけでなく、質問、順序、分解、再計画も評価する
- ローカルで試しやすいように `.venv` と `make` を前提にする

## 今後やりたいこと

今後やりたいことは次です。

- `meeting_prep` 以外のドメイン追加
- `finance_analyst` のような状態変化型の分析タスク拡充
- 「質問すべきケース」を case 側でも明示して、質問評価を厳密化
- rubric と evaluator の対応関係をさらに明確化
- 複数 sample log を用意して、failure pattern ごとのデモを増やす
- notebook を説明資料としてもっと見やすくする
- 評価結果を表や可視化で見られるようにする

## いま見るべきファイル

最初に把握するなら、次の順番が見やすいです。

1. [`README.md`](README.md)
2. [`scenarios/meeting/meeting_001.json`](scenarios/meeting/meeting_001.json)
3. [`results/sample_execution_meeting_001.json`](results/sample_execution_meeting_001.json)
4. [`src/models.py`](src/models.py)
5. [`src/rule_evaluator.py`](src/rule_evaluator.py)
6. [`src/build_notebook.py`](src/build_notebook.py)
