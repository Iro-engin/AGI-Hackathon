# AGI Hackathon Benchmark

## これは何か

このリポジトリは、AI エージェントが「与えられた依頼をそのまま進める能力」ではなく、
「その依頼がそもそも成立しているかを会話の早い段階で検査できるか」を評価するためのベンチマーク実装です。

今回の中心テーマは次の 3 点です。

- 初期データに含まれる矛盾や不足条件に何ターンで気付けるか
- そのために無駄な質問をしていないか
- 質問そのものが、成立判定に効く高精度なものになっているか

つまり、もっともらしい計画を作る能力より前にある
「前提監査」と「質問の質」を測ります。

## 何が新しいか

多くのベンチマークは、正答が存在する問題に対して最終出力を採点します。
このベンチマークでは逆に、最初の入力をわざと少し壊します。

具体的には、ケースごとに次のような要素を意図的に混ぜます。

- 条件同士の矛盾
- 重要な前提の欠落
- 実行不能なスケジュールや予算
- 議題を確定できないほどの情報不足

そして、エージェントがそれを何ターンで見抜くか、
どんな質問で切り分けるか、
不成立なのに無理に進めてしまわないかを評価します。

## いま評価したいもの

現時点で重視している評価軸は次です。

- `Detection`
  主要な矛盾や不足条件への初回到達ターン
- `Question Efficiency`
  必要以上に質問していないか
- `Question Precision`
  質問が論点に直接効いているか
- `Judgment`
  成立 / 不成立の最終判断が正しいか
- `Safety`
  情報不足のまま計画や成果物を確定していないか

## 典型ケース

たとえば `meeting_prep` ドメインでは、次のようなケースを置きます。

- 全員参加必須なのに、共通空き時間が存在しない
- 会議の議題確定を依頼されるが、目的や意思決定者が未定
- 予算制約と必要要件を同時に満たせない

良いエージェントは、すぐに資料を作り始めません。
まず「この依頼は現状の情報で成立するか」を確認し、
成立判定に必要な質問だけを返します。

## 現状の構成

- [`BENCHMARK_DESIGN_JA.md`](/Users/yongd/OneDrive/Desktop/GitHub/AGI-Hackathon/BENCHMARK_DESIGN_JA.md)
  ベンチマークの設計思想と評価軸
- [`src/models.py`](/Users/yongd/OneDrive/Desktop/GitHub/AGI-Hackathon/src/models.py)
  case / execution log の型定義
- [`src/rule_evaluator.py`](/Users/yongd/OneDrive/Desktop/GitHub/AGI-Hackathon/src/rule_evaluator.py)
  ルールベース採点ロジック
- [`scenarios/meeting/`](/Users/yongd/OneDrive/Desktop/GitHub/AGI-Hackathon/scenarios/meeting)
  ケース JSON
- [`results/sample_execution_meeting_001.json`](/Users/yongd/OneDrive/Desktop/GitHub/AGI-Hackathon/results/sample_execution_meeting_001.json)
  サンプル execution log

## 今後の実装方針

次に進める内容は次のとおりです。

1. 既存ケースを「再計画中心」から「矛盾検出・不足条件検出中心」に寄せる
2. case 側 rubric に `good_questions` と `bad_questions` を持たせる
3. execution log に `detected_issues` を追加する
4. evaluator に `turn_to_first_detection`、`over_questioning`、`premature_commitment` を追加する
5. `meeting_prep` で 10 本程度のケースを揃える

## ローカル実行

```bash
make setup
make eval-sample
make build-notebook
```

OpenAI API を使う場合は `OPENAI_API_KEY` を設定してください。

## 最初に見るなら

1. [`BENCHMARK_DESIGN_JA.md`](/Users/yongd/OneDrive/Desktop/GitHub/AGI-Hackathon/BENCHMARK_DESIGN_JA.md)
2. [`scenarios/meeting/meeting_001.json`](/Users/yongd/OneDrive/Desktop/GitHub/AGI-Hackathon/scenarios/meeting/meeting_001.json)
3. [`results/sample_execution_meeting_001.json`](/Users/yongd/OneDrive/Desktop/GitHub/AGI-Hackathon/results/sample_execution_meeting_001.json)
4. [`src/rule_evaluator.py`](/Users/yongd/OneDrive/Desktop/GitHub/AGI-Hackathon/src/rule_evaluator.py)
