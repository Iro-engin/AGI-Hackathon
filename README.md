# AGI Hackathon Benchmark

Kaggle の `Measuring AGI` を意識しつつ、特に「AIエージェントが動的な状況変化に対して正しく再計画し、成果物を最新状態に合わせて更新できるか」を評価するための最小ベンチマーク実装です。

このリポジトリでは、会議準備ドメインを題材に以下を一体で管理します。

- ベンチマークケース定義
- 実行ログ定義
- ルールベース evaluator
- Kaggle 提出向け notebook 生成

## 何を評価するか

本ベンチマークの中心は、単なる最終成果物の出来だけではなく、途中の意思決定プロセスまで含めて評価することです。

- 質問
  初期情報が不足しているときに、必要な確認を先に行えているかを見ます。`questions_asked` で、何を、なぜ、どのタスクを止める質問として聞いたかを記録できます。
- タスクの実行順番
  `initial_state.task_dependencies` と `completed_tasks` を比較し、依存関係を壊していないかを評価します。
- 適切なタスク分解
  `task_breakdown` により、作業をどの粒度で分解し、どの依存を持つかを明示します。evaluator は、実際に完了したタスクがこの分解に現れているかを確認します。
- 状態変化への追従
  `events` の内容を認識し、指定ターン内に `confirm_state` または `update_plan` で再計画したかを見ます。
- 成果物の整合性
  `required_artifacts` と `semantic_checks` に基づき、最終成果物が最新 state を反映しているかを見ます。

## コード概要

### `src/models.py`

Pydantic による入出力モデル定義です。

- `BenchmarkCase`
  ケース入力全体のトップレベルモデル
- `ExecutionLog`
  実行ログ入力全体のトップレベルモデル
- `CaseCatalog`
  case 一覧を要約表示するための補助クラス
- `JsonFileRepository`
  ケースと execution log をファイルから読み込む小さなリポジトリクラス

これにより、`case` と `execution_log` は `dict[str, Any]` 前提ではなく、読み込み時点で検証済みオブジェクトとして扱えます。

### `src/rule_evaluator.py`

ルールベース評価器です。

- `RuleBasedEvaluator`
  評価ロジック本体を持つクラス
- `EvaluationResult`
  採点結果を返す dataclass
- `evaluate_case`
  既存コード互換のための薄いラッパー関数

評価スコアは以下の3軸です。

- `outcome_score`
  最終成果物と最終 state の整合性
- `process_score`
  許可アクション、質問、タスク分解、タスク順序などのプロセス品質
- `recovery_score`
  イベント発生後の再計画と回復の速さ

### `src/build_notebook.py`

Kaggle 提出用 notebook の生成器です。

- `NotebookCellFactory`
  notebook cell を安定した ID 付きで生成
- `BenchmarkNotebookBuilder`
  Pydantic models、evaluator、JSON 読込 helper をまとめて notebook 化

notebook では巨大な `MEETING_CASES = [...]` を埋め込まず、`scenarios/meeting/*.json` と
`results/sample_execution_meeting_001.json` を実行時に読み込みます。

### `src/logging_config.py`

リポジトリ共通のログ設定です。`LOG_LEVEL` 環境変数でログレベルを切り替えられます。

### `scenarios/meeting/*.json`

会議準備ドメインのベンチマークケース群です。ケースは以下を持ちます。

- 初期依頼
- 初期状態
- 許可アクション
- イベント
- ゴール条件
- ルーブリック

### `results/sample_execution_meeting_001.json`

評価デモ用のサンプル execution log です。  
質問、タスク分解、アクション列、完了タスク、最終 state、最終成果物を含める想定です。

### `schemas/*.json`

JSON Schema 定義です。

- `benchmark_case.schema.json`
  benchmark case のスキーマ
- `execution_log.schema.json`
  execution log のスキーマ

## 入力フォーマット

### Benchmark case

主な要素は以下です。

- `initial_state.required_artifacts`
  必須成果物と、その必須項目、state との semantic binding
- `initial_state.task_dependencies`
  タスクの順序制約
- `events`
  状態変化や新しい制約
- `goal_condition`
  最新 state を満たすべきか、制約違反を許すかなど

### Execution log

主な要素は以下です。

- `actions`
  ターンごとのエージェント行動
- `questions_asked`
  必要な確認質問の記録
- `task_breakdown`
  タスク分解の記録
- `completed_tasks`
  実際に完了したタスク列
- `final_state`
  最終的に採用した状態
- `final_artifacts`
  最終成果物

## 生成フロー

1. `scenarios/meeting/*.json` を Pydantic で読み込む
2. `results/sample_execution_meeting_001.json` を Pydantic で読み込む
3. `src/models.py` と `src/rule_evaluator.py` を notebook に埋め込む
4. 生成された notebook は実行時に JSON ファイルを読み込む
5. `src/build_notebook.py` で `src/kaggle_submission_benchmark.ipynb` を再生成する

## 使い方

環境構築:

```bash
make setup
```

直接スクリプトを使う場合:

```bash
./runner/bootstrap.sh
```

notebook 再生成:

```bash
make build-notebook
```

ログレベルを指定して実行:

```bash
LOG_LEVEL=DEBUG make build-notebook
```

sample 評価実行:

```bash
make eval-sample
```

ローカルで evaluator を使う例:

```python
from pathlib import Path

from src.models import BenchmarkCase, ExecutionLog
from src.rule_evaluator import RuleBasedEvaluator

case = BenchmarkCase.from_path(Path("scenarios/meeting/meeting_001.json"))
log = ExecutionLog.from_path(Path("results/sample_execution_meeting_001.json"))
result = RuleBasedEvaluator().evaluate(case, log)

print(result)
```

## 設計意図

- 最小変更で notebook 提出物を維持する
- JSON をそのまま資産として持ちつつ、Python 側では型安全に扱う
- 最終成果物だけではなく、質問、順序、分解、再計画まで評価対象に含める
- ケース追加時に `scenarios/meeting/*.json` を足すだけで notebook 再生成に乗る構造にする
