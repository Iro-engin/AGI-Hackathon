# scenarios/meeting

このディレクトリの JSON は、`meeting_prep` ドメインの benchmark case です。  
ここでは「会議をどう準備するか」ではなく、「エージェントに何をさせたいケースなのか」を JSON で表現します。

この README は、共通 README の補足として、`scenarios/meeting/*.json` の各項目が meeting ケースでは何を意味するかを具体的に説明します。

## 1. トップレベル項目

- `task_id`: ケースの一意 ID。通常は `meeting_001` のような連番です。
- `domain`: ドメイン名。meeting ケースでは基本的に `meeting_prep` を使います。
- `difficulty`: ケース難易度。`easy` / `medium` / `hard` など、ケース作者が意図した難しさを示します。
- `initial_request`: 最初にエージェントへ渡す依頼文。会議テーマ、参加者、欲しい成果物、決めたいことを自然文で書きます。
- `initial_state`: 初期状態。締切、参加者、制約、成果物定義、補助情報をまとめます。
- `allowed_actions`: execution log 側で許可する action 種別の一覧です。
- `events`: 途中で起こる状態変化。会議条件の変更、参加者追加、曖昧化、資料更新などを入れます。
- `goal_condition`: 最終的に満たすべき成功条件です。
- `rubric`: このケースで何を評価したいかを自然文で記述した採点観点です。
- `notes`: ケース作者向けの補足。運用メモや設計意図を書きます。

## 2. `initial_state` の意味

- `timezone`: 会議準備で使う基準タイムゾーンです。`deadline` の解釈基準になります。
- `deadline`: 最終成果物を揃えるべき期限です。会議の開催時刻そのものとは限らず、「その時刻までに準備完了すべき」という意味で使います。
- `participants`: 会議参加者またはレビュー参加者の一覧です。agenda や memo にこの前提が反映されているかを見ます。
- `budget`: 会議準備のコスト上限や運用上の予算枠です。meeting ケースでは厳密な金額計算より、「制約として存在する状態値」という意味合いが強めです。
- `meeting_duration_limit_minutes`: `initial_state` に追加で置かれる時間上限です。agenda の時間配分や timebox がこの値に収まる必要があります。
- `ops_approval_available`: Ops 承認が今この時点で取れるかどうかのフラグです。会議内承認に進めるか、非同期承認へ切り替えるかに効きます。
- `required_artifacts`: 最終的に必ず作る成果物の定義です。
- `constraints`: 守るべきルールです。会議時間上限、共有範囲、曖昧なら確認必須、といった禁止条件や前提条件を書きます。
- `task_dependencies`: 作業順序の依存関係です。「何を先に確定しないと次へ進めないか」を evaluator に伝えます。
- `reference_data`: 会議テーマ固有の補足情報です。比較候補、共有方針、資料版、論点の状態など、meeting ドメイン特有の判断材料を入れます。

### 2.1 `required_artifacts` の各項目

- `artifact_id`: 成果物の論理名。`agenda`、`briefing_doc`、`decision_memo` などです。
- `artifact_type`: 成果物の種類。`document` や `slides` など、出力の形式を表します。
- `required_fields`: その成果物に必須のセクション名です。`agenda_items` や `risks` など、execution log の `final_artifacts[*].fields_completed` / `field_values` に存在してほしい項目です。  
  ここに入る文字列は evaluator の固定スキーマではなく、そのケースが要求する「成果物の見出し名」です。
- `semantic_checks`: 成果物中のある項目が、state のどの値を反映すべきかを対応付ける定義です。
- `latest_version_required`: `true` の場合、event 後に古い前提のまま残ってはいけません。最新版へ更新されている必要があります。

### 2.2 `semantic_checks` の各項目

- `artifact_field`: 成果物側で確認したい項目です。
- `state_path`: 反映元の state のパスです。`deadline`、`participants`、`reference_data.share_policy` のように書きます。

### 2.3 `task_dependencies` の各項目

- `before`: 先に終わっていなければならないタスク ID です。
- `after`: `before` の後でないと進めてはいけないタスク ID です。
- `reason`: その順序制約が必要な理由です。

## 3. `allowed_actions` の意味

現在の meeting ケースでは、次の action 名を使います。

- `ask_clarification`: 不足情報や曖昧さを解消する質問を行う。
- `propose_plan`: 初期の作業計画を提示する。
- `update_plan`: event 後などに計画を更新する。
- `create_artifact`: 成果物を新規作成する。
- `revise_artifact`: 既存成果物を修正する。
- `confirm_state`: 参加者追加、締切変更、共有方針変更などの state 変化を明示的に認識する。
- `finalize`: 成果物と state を揃えたうえで完了扱いにする。

## 4. `events` の意味

各 event は「途中で前提が変わること」を表します。

- `turn`: 何ターン目で event が発生するかです。厳密な会話ターンというより、action の時系列ステップ番号として使います。
- `type`: 変化の種類です。meeting ケースでは主に次を使います。  
  - `new_constraint`: 会議時間上限や必須記載事項のような新しい制約が追加される。  
  - `state_change`: 締切や参加者のような状態値が変わる。  
  - `stakeholder_change`: 意思決定者や関係者が増減し、見直しが必要になる。  
  - `artifact_update`: 参照デッキや資料パッケージの版が更新され、成果物も追随修正が必要になる。  
  - `ambiguity`: 比較軸、優先順位、共有範囲などが曖昧になり、確認が必要になる。
- `message`: エージェントに見せる event の自然文説明です。
- `delta`: state にどう反映されるかを機械可読で表した差分です。
- `expected_artifact_updates`: この event を受けて更新されるべき成果物 ID の一覧です。
- `expected_tasks`: この event 後に実行してほしいタスク ID の一覧です。
- `expected_replan_within_turns`: 何ターン以内に再計画または状態更新をしてほしいかです。

### 4.1 `delta` で meeting ケースに出る主な項目

- `delta_applies_to`: どこに差分を当てるかの補足です。既存ケースでは主に `initial_state` を指定します。
- `added_constraints`: 新しく追加される制約です。
- `overridden_constraints`: 以前の制約のうち、置き換え対象になるものです。
- `participants_added`: 追加された参加者です。agenda や memo の前提見直しが必要になります。
- `deadline`: 変更後の締切です。
- `meeting_duration_limit_minutes`: 変更後の会議時間上限です。
- `reference_data`: テーマ固有の補助情報の差し替えです。比較軸の曖昧化、資料版更新、共有ポリシー変更などをここに入れます。

## 5. `goal_condition` の意味

- `must_satisfy_latest_state`: 最終成果物が event 後の最新 state を満たしている必要があります。
- `required_artifacts`: 最終提出時に必須の成果物 ID 一覧です。
- `no_constraint_violation`: 制約違反がないことを要求します。
- `must_acknowledge_changes`: event による状態変化を明示的に認識している必要があります。
- `must_use_ask_clarification_on_ambiguity`: `ambiguity` 発生時に確認質問を要求します。
- `must_not_finalize_with_unresolved_scope`: 未解決の論点や scope が残ったまま finalize してはいけない、というフラグです。現状は `false` のケースもあります。

## 6. `rubric` の意味

- `outcome`: 最終成果物の質と整合性を見る観点です。
- `process`: 質問、依存順、再計画、成果物更新の進め方を見る観点です。
- `recovery`: event 後の追随速度と立て直し方を見る観点です。

## 7. meeting 固有の `reference_data` キー

`reference_data` はケース固有の辞書で、必須キーはケースごとに変わります。  
以下は、このディレクトリの既存ケース群で実際に使っているキーと意味です。

### 7.1 会議の基本情報

- `meeting_type`: 会議の種類。`executive review`、`migration review` のような会議の文脈です。
- `meeting_mode`: 会議の実施モード。オンライン、対面、ハイブリッドのような実施形態の前提です。
- `session_format`: セッション形式。`in-person workshop` のように、進め方の型を表します。
- `event_format`: 対外イベントや顧客向け会の形式です。配信形態や開催形態の前提として使います。
- `venue_mode`: 開催場所の前提です。`offsite` など、移動や準備の条件に効きます。
- `room_type`: 必要な部屋の種類です。会議室種別や設備条件を表します。
- `logistics_mode`: 会場案内、部屋セットアップ、機材手配など、運営面で必要な準備モードです。
- `topic`: 会議テーマです。agenda や memo が何についての会議かを明示します。
- `session_goal`: その場で達成したいセッション目的です。顧客向け会や共同会議でよく使います。
- `workshop_goal`: workshop 形式の会で達成したい目的です。
- `decision_goal`: その会議で最終的に何を決めたいかです。
- `decision_mode`: どういう決め方をする会議かの前提です。合意形成中心か、承認型か、といった意思決定方式を補足します。
- `planning_horizon`: 何期間先までを議論対象にするかです。例: 次四半期、今後 2 四半期。
- `target_date`: 合意したい目標日、実施日、切替日などです。
- `launch_date`: ローンチ予定日です。発売準備や readout のケースで使います。
- `participant_availability_note`: 参加者の細かい出席条件です。例: 「CFO は前半 30 分のみ参加可能」。

### 7.2 比較対象・候補・判断軸

- `agenda_min_items`: agenda に最低限入れるべき議題数です。
- `known_options`: 比較対象として最初から分かっている選択肢です。
- `scope_options`: scope の候補一覧です。
- `candidates`: 優先順位づけや比較の対象候補一覧です。
- `candidate_themes`: 投資テーマや戦略テーマの候補です。
- `candidate_experiments`: 実験案の候補です。
- `candidate_count`: 候補数の目安です。何件に絞るかの前提に使います。
- `comparison_basis`: 候補比較の基準です。何を軸に案を並べるかを示します。
- `primary_comparison_axis`: 主たる比較軸です。例: 成長速度、粗利インパクト。
- `decision_axis_status`: 比較軸や判断軸が確定済みか曖昧かの状態です。
- `priority_axis_status`: 優先順位の付け方が確定済みか曖昧かの状態です。
- `scope_status`: 範囲が確定済みか、曖昧か、縮小が必要かを示す状態です。
- `scope_decision_status`: scope に関する意思決定の状態です。
- `external_scope_status`: 外部共有範囲や対外説明範囲が確定しているかの状態です。
- `migration_scope`: 移行対象の範囲そのものです。
- `migration_scope_status`: 移行範囲の情報が更新済みか、見直し必要かを示す状態です。
- `launch_readiness_status`: ローンチ準備状況の状態です。途中 event で変化する前提として使います。
- `priority_snapshot_status`: 優先順位付け資料の更新状態です。
- `roadmap_snapshot_status`: ロードマップ資料の更新状態です。
- `pricing_snapshot_status`: 価格改定や pricing 関連資料の更新状態です。
- `experiment_snapshot_status`: 実験計画や検証資料の更新状態です。

### 7.3 共有方針・承認・統制

- `share_policy`: 誰にどこまで共有してよいかの方針です。`internal only`、`customer-safe only`、`split internal and external versions` などを取ります。
- `internal_note_policy`: 内部向け資料にどこまで詳細を書いてよいかの方針です。
- `anonymize_customer_names`: 顧客名を匿名化する必要があるかです。
- `security_review_mode`: セキュリティ観点の確認をどう行うかです。例: 会議中に live review をする。
- `approval_mode`: 承認の取り方です。会議内承認か、別途承認かなどを示します。
- `qa_signoff`: QA の承認済みかどうか、または QA 承認が必要かの状態です。
- `support_input_mode`: Support チームの意見をどう取り込むかです。例: live discussion。

### 7.4 参照資料・版管理・内容サマリ

- `deck_version`: 参照しているデッキの版です。
- `artifact_version`: 参照パッケージや会議資料の版です。event の `artifact_update` で更新されやすい値です。
- `messaging_version`: 対外メッセージやトークトラックの版です。
- `release_candidate`: 対象となる RC 版や候補版です。
- `approved_storyline`: すでに承認済みのストーリーラインやトピック列挙です。
- `product_context`: 製品や案件の背景説明です。
- `feature_summary`: 対象機能の要約です。
- `tool_summary`: 対象ツールやプロダクトの要約です。
- `incident_summary`: 障害やインシデントの概要です。
- `incident_severity`: インシデント重大度です。
- `key_metrics`: 主要指標の一覧です。
- `success_metrics`: 成功判定に使う指標です。
- `forecast_hires`: 採用計画の見込み人数です。
- `sales_snapshot_date`: 売上資料のスナップショット日付です。

### 7.5 関係者・対象・周辺情報

- `roles_in_scope`: 議論対象となる役割・職種です。
- `vendors`: 比較対象ベンダー一覧です。
- `vendor_tier`: ベンダーの重要度や格付けです。例: strategic。
- `renewal_type`: 契約更新の種類です。例: annual。
- `joint_project`: 共同案件名や協業テーマです。
- `customer_segments`: 対象顧客セグメントです。

### 7.6 リスク・既知前提

- `known_issues`: 既知の問題点です。
- `known_risks`: 既知のリスクです。
- `known_tradeoff`: 既知のトレードオフです。

## 8. `required_fields` の読み方

meeting ケースでの `required_fields` は、「その成果物に最低限そろっていてほしい見出し名」です。  
たとえば次のように解釈します。

- `agenda_items`, `topics`, `decision_topics`: 話す議題一覧。
- `owner`, `owners`: 各議題やアクションの担当者。
- `timebox`, `timeboxes`, `time_allocation`: 時間配分。
- `summary`, `context`, `background`, `shared_context`: 前提共有。
- `decision_points`, `decision_needed`, `decision_goal`, `decision_criteria`, `decision_rules`: 何をどう決めるか。
- `risks`, `top_risks`, `known_risks`, `customer_risks`, `readiness_gaps`: 主要リスクや未解決点。
- `next_actions`, `next_steps`, `followups`, `action_items`: 会議後の具体アクション。
- `options`, `scope_options`, `internal_options`, `candidates`, `scores`, `criteria`: 比較・評価のための材料。
- `approved_answers`, `internal_only_answers`, `do_not_share`, `do_not_say`, `share_version_note`, `confidentiality_note`: 共有範囲や対外説明の境界。
- `approval_mode`, `approval_path`, `security_review_mode`: 承認経路やレビュー方式。

要するに、`required_fields` は meeting ケースごとの「成果物契約」であり、意味づけは `artifact_id`、`initial_request`、`constraints`、`reference_data`、`rubric` とセットで読むのが正解です。
