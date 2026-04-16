# Science Generator

## 何を作るか

`src/science` は、稼働中の架空化学式に対して危険が発生したとき、
必要な質問をしながら 1 つの対処手順へ絞る benchmark case を
生成・実行・評価します。

見る能力は主に次です。

- 何を質問すべきかを 1 turn でまとめられるか
- 質問チェックリストをどこまで早く回収できるか
- すでに聞いたことや自明なことを再質問しないか
- 所持道具で実行可能な手順を選べるか
- 危険が切り替わったあとも同じ要領で再対応できるか

## ファイル分割

- `type/base.py`
  - 共通の型定義
- `type/constraint_variants.py`
  - 制約種別ごとの派生型
- `type/difficulty.py`
  - `easy / normal / hard` の派生クラス
- `type/validators.py`
  - 型定義と builder から使う共通検証関数
- `prompts.py`
  - Phase 1 / Phase 2 の生成 prompt
- `generate_cases.py`
  - OpenAI 呼び出しと 2 段階生成
- `case_builder.py`
  - 最終 `Case` への変換
- `executor.py`
  - ケース実行
- `evaluator.py`
  - 実行結果の評価

## 生成フロー

このドメインは 2 段階生成です。

1. Phase 1  
   `initial_request`, `formula_hazard_links`, `initial_tool_constraints`,
   `selected_constraint_types`, `events` を作る
2. Phase 2  
   Phase 1 の結果をもとに、各 hazard について
   `hazard_manuals`, `hazard_question_targets`,
   `hidden_hazard_facts` を作る

## 難易度

- `easy`
  - hazard 数: 2-3
  - selected constraint 数: 1-2
  - 各 hazard の質問項目数: 1-2
- `normal`
  - hazard 数: 3-4
  - selected constraint 数: 3-4
  - 各 hazard の質問項目数: 3-4
- `hard`
  - hazard 数: 4-5
  - selected constraint 数: 3-4
  - 各 hazard の質問項目数: 3-4

`events` は常に 3 件で、turn は `1, 2, 3` 固定です。

`participants`, `deadline`, `budget` は science 固有 schema では持ちません。
repository 共通の `Case` に変換するときだけ固定ダミー値を入れます。

## 主要型

### `reference_data.formula_hazard_links`

化学式と hazard の 1 対 1 対応です。

- `formula_id`
- `hazard_id`

### `reference_data.initial_tool_constraints`

作業開始時点で使える道具一覧です。

### `reference_data.selected_constraint_types`

このケースで質問対象にする制約種類です。

取りうる値:

- `pressure_pa`
- `catalyst_present`
- `flammable_nearby`
- `escape_easy`
- `water_usable`

### `hazard_manuals[].procedures[].applicable_conditions`

各手順がどの条件で適用されるかを表します。

各条件は

- `constraint_type`
- `mode`
- `min_numeric`
- `max_numeric`
- `expected_bool`

を持ち、`pressure_pa` は range、他は bool か irrelevant で表現します。

各 `hazard_manual` には

- `max_applicable_conditions_per_procedure`

もあり、その hazard 内の各 procedure は
`1..max_applicable_conditions_per_procedure` 件の
`applicable_conditions` を持てます。
すべての procedure が同じ件数である必要はありません。

### `reference_data.hazard_question_targets`

hazard ごとの質問チェックリストです。

evaluator は質問文を正規化したうえで、
このチェックリストに完全一致しているかだけで
「適切な質問か」を判定します。
質問理由は採点に使いません。

### `reference_data.hidden_hazard_facts`

質問されたときだけ返す隠し事実です。

- `constraint_values`
- `expected_procedure_id`

を持ちます。
`expected_procedure_id` は、その実測値で最終的に選ばれるべき手順です。

## 実行

executor は各 turn で次のどちらか 1 つを返します。

- `ask`
  - 質問をまとめて出す
- `do`
  - `procedure_id` を 1 つ選ぶ

## 評価指標

主な指標は次です。

### 1. 1/3/5 turn の質問回収率

各 hazard episode ごとに、
質問チェックリストのうち何件を回収できたかを見ます。

### 2. 過剰質問率

自明または既出の質問をどれだけしたかを見ます。

### 3. 30 turn 以内に回答できたか

各 hazard について、開始から 30 turn 以内に
最初の `do` を返せたかを見ます。

### 4. 計画一致率

選んだ `procedure_id` が
`hidden_hazard_facts.expected_procedure_id` と一致したかを見ます。

### 5. 実行可能性

選んだ手順の `required_tools` が
`initial_tool_constraints` に含まれているかを見ます。

### 6. 曖昧実行率

質問ターゲットを回収しきる前に
`do` へ進んでしまった割合です。

### 7. 直接実行率

追加質問なしでそのまま `do` できた割合です。

# 手動での制御項目
- conditionのうち、どんな順番で質問してもP個確定させないと必ず１つに絞れないよう、condition・procedureの条件を設定しなおす

## Makefile

よく使うターゲット:

- `make science-dry-run`
- `make science-generate`
- `make science-execute`
- `make science-execute-all`
- `make science-evaluate`
- `make science-evaluate-all`
