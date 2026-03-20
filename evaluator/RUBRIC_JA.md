# Evaluator Rubric

## 目的

この rubric は、Dynamic Agent Correctness Benchmark における採点観点を固定化するための文書である。

採点は以下の 3 カテゴリで行う。

- Outcome Score: 最終成果物の整合性
- Process Score: 各ターンでの行動の妥当性
- Recovery Score: 状態変化後の復帰力


## 1. Outcome Score

0 から 100 で採点する。

### 主要観点

- 最新の world state を反映した成果物になっているか
- 必須 artifact がすべて存在するか
- 各 artifact の required_fields が満たされているか
- 最新制約に違反していないか
- artifact 同士に矛盾がないか

### 減点ルールの例

- 最新参加者や最新締切を未反映: -20
- 必須 artifact の欠落: 1 件ごとに -25
- required_fields の欠落: 1 項目ごとに -5
- 明確な制約違反: 1 件ごとに -20
- 古い前提と新しい前提の混在: -15


## 2. Process Score

0 から 100 で採点する。

### 主要観点

- 初期依頼から妥当なタスク分解を行っているか
- 依存関係に沿った順序で動いているか
- 不確実性があるときに確認行動を取っているか
- 状態変更後に古い計画を継続していないか
- 更新が必要な artifact やタスクを特定しているか

### 減点ルールの例

- 依存関係違反: 1 件ごとに -20
- 明らかに必要な確認を飛ばした: -10
- 変更後も旧計画を継続: -15
- 不要アクションが多い: 最大 -10


## 3. Recovery Score

0 から 100 で採点する。

### 主要観点

- イベント発生後に所定ターン内で再計画したか
- 影響範囲を正しく認識したか
- 修正量が過不足ないか
- 再計画後に整合したルートへ戻れているか

### 減点ルールの例

- expected_replan_within_turns を超過: 1 ターン超過ごとに -15
- 影響範囲の見落とし: 1 件ごとに -15
- 変更と無関係な大幅手戻り: -10


## 4. 失敗ラベルとの対応

採点時には点数だけでなく失敗ラベルも付与する。

- `state_staleness`: 古い状態を保持したまま作業を続けた
- `missing_replan`: イベント後に再計画がない
- `partial_replan`: 一部のみ更新し整合性が崩れた
- `invalid_dependency`: 依存順序を破った
- `goal_drift`: ゴールから逸脱した
- `unsafe_commit`: 確認前に確定行動へ進んだ
- `constraint_violation`: 明示制約に反した
- `artifact_inconsistency`: 成果物間または成果物内で矛盾した


## 5. 実装向けメモ

- Outcome はルールベースを優先する
- Process は event 前後の action_type と state_assumptions を重点的に見る
- Recovery は event.turn と最初の `update_plan` または `confirm_state` の差分で測る
- LLM 補助採点を入れる場合も、最終スコアの過半はルールベースで決める
