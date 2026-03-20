こちらのコンペティションに参加します
そこで今回は特にAIエージェントの挙動の正当性担保について特に詳しく発展させた評価系の設計をしたいと考えている
https://www.kaggle.com/competitions/kaggle-measuring-agi


例
会議の準備を依頼→タスク分解→数回の会話ラリー→資料作成→最終確認→会議の前提条件が変わる→適切なタスクの修正と順番の更新→資料修正→最終確認

というような一連のタスクの正当性を評価するベンチマークの設計案になる

## 現在の最小構成

- `src/kaggle_submission_benchmark.ipynb`
  Kaggle 提出向けの self-contained notebook 本体
- `src/build_notebook.py`
  JSON ケースと evaluator から notebook を再生成するスクリプト
- `src/rule_evaluator.py`
  ルールベース採点ロジックの単一実装
- `scenarios/meeting/*.json`
  会議準備ドメインのベンチマークケース
- `results/sample_execution_meeting_001.json`
  evaluator 動作確認用のサンプル execution log
- `schemas/benchmark_case.schema.json`
  ケース JSON のスキーマ
- `Dockerfile`, `.dockerignore`, `requirements.txt`, `pyproject.toml`
  Docker 実行と lint 用の環境定義

## 生成フロー

1. `scenarios/meeting/*.json` と `results/sample_execution_meeting_001.json` を読む
2. `src/rule_evaluator.py` を notebook に埋め込む
3. `src/build_notebook.py` で `src/kaggle_submission_benchmark.ipynb` を生成する
