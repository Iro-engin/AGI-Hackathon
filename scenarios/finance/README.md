# scenarios/finance

このディレクトリの JSON は、`finance_analyst` ドメインの benchmark case です。  
ここでは「ニュース起点でセクタービューを更新し、比較候補と推奨配分を整理できるか」をケース化しています。

この README では、`scenarios/finance/*.json` に出てくる各項目が finance ケースで何を意味するかを具体的に説明します。

## 1. トップレベル項目

- `task_id`: ケースの一意 ID。通常は `finance_001` のような連番です。
- `domain`: ドメイン名。finance ケースでは基本的に `finance_analyst` を使います。
- `difficulty`: ケース難易度です。
- `initial_request`: 最初にエージェントへ渡す依頼文です。主導セクター、崩れうる前提、見たいローテーション先、必要成果物を書きます。
- `initial_state`: 分析開始時点の市場前提と成果物要件です。
- `allowed_actions`: execution log で許可する action 一覧です。
- `events`: ニュース確認、制約追加、ローテーション曖昧化などの途中変化です。
- `goal_condition`: 最終的に満たすべき成功条件です。
- `rubric`: outcome / process / recovery の観点を書きます。
- `notes`: ケース作者向けの補足です。

## 2. `initial_state` の意味

- `timezone`: 朝会やレビューの基準タイムゾーンです。
- `deadline`: メモや更新案を揃えるべき期限です。
- `participants`: レビュー相手や受け手の役割です。finance ケースでは `PM`、`AnalystLead`、`RiskManager` のような読み手を置くことが多いです。
- `budget`: 配分総量の粗い上限です。current corpus では多くのケースで `100` を使い、「100 がフル配分」という近似表現にしています。
- `required_artifacts`: 最終的に必ず作る成果物定義です。
- `constraints`: 守るべき分析ルールです。未確認ニュースの断定禁止、曖昧時の単一推奨禁止、比較セクター数、配分合計などを書きます。
- `task_dependencies`: どの分析タスクを先にやるべきかの順序制約です。
- `reference_data`: セクターローテーション判断に使う市場テーマ、現在配分、相関、監視銘柄などです。

### 2.1 `required_artifacts` の各項目

- `artifact_id`: 成果物の論理名です。finance ケースでは主に `news_reaction_note` と `sector_watch_update` を使います。
- `artifact_type`: 成果物の種類です。既存ケースでは `document` が中心です。
- `required_fields`: その成果物に最低限必要なセクション名です。
- `semantic_checks`: 成果物の特定項目が、state のどの値を反映すべきかを定義します。
- `latest_version_required`: `true` の場合、ニュース発生後や制約変更後に古い前提のまま残ってはいけません。

### 2.2 `semantic_checks` の各項目

- `artifact_field`: 成果物内で確認したい項目です。
- `state_path`: 反映元の state のパスです。`reference_data.losing_sector` などを指定します。

### 2.3 `task_dependencies` の各項目

- `before`: 先に終えるべき分析タスクです。
- `after`: その後で実施すべきタスクです。
- `reason`: この順序が必要な理由です。

## 3. `allowed_actions` の意味

現在の finance ケースでは次を使います。

- `ask_clarification`: ニュース確認不足や曖昧な回転先について追加確認を求める。
- `propose_plan`: 初期の分析計画を提示する。
- `update_plan`: event 後に分析観点や優先順位を更新する。
- `create_artifact`: 監視メモや更新メモを作る。
- `revise_artifact`: 既存メモを最新 state に合わせて修正する。
- `confirm_state`: ニュース状態、ローテーション状態、制約追加を明示的に認識する。
- `finalize`: 最新前提に揃えたうえで完了にする。

## 4. `events` の意味

- `turn`: event が起こるターンです。
- `type`: 変化の種類です。finance ケースでは主に次を使います。  
  - `new_constraint`: 配分上限や比較条件のような新しい制約が追加される。  
  - `state_change`: 主導セクターを支えていた前提が崩れる、`losing_sector` が確定する、などの状態変化。  
  - `stakeholder_change`: 受け手やレビュー参加者が追加される。  
  - `ambiguity`: 回転先が複数あり、単一セクターへ断定できない状態になる。
- `message`: event の自然文説明です。
- `delta`: state に加える差分です。
- `expected_artifact_updates`: この event を受けて更新されるべき成果物 ID です。
- `expected_tasks`: この event 後に進めるべきタスク ID です。
- `expected_replan_within_turns`: 再計画や状態更新を求める期限です。

### 4.1 `delta` で finance ケースに出る主な項目

- `delta_applies_to`: 差分の適用先です。既存ケースでは主に `initial_state` です。
- `added_constraints`: 新しく追加される分析制約です。
- `participants_added`: 新たに加わるレビュー相手です。
- `reference_data`: 市場テーマ、ニュース状態、配分テンプレート、勝ち候補などの更新差分です。

## 5. `goal_condition` の意味

- `must_satisfy_latest_state`: 最終成果物が event 後の最新 state を反映している必要があります。
- `required_artifacts`: 必須成果物 ID 一覧です。
- `no_constraint_violation`: 制約違反がないことを要求します。
- `must_acknowledge_changes`: ニュースや制約の変化を明示的に認識している必要があります。
- `must_use_ask_clarification_on_ambiguity`: 回転先が曖昧なときに確認要求を入れる必要があります。
- `must_not_finalize_with_unresolved_scope`: 未解決論点を抱えたまま finalize してよいかのフラグです。現行ケースでは `false` のことがあります。

## 6. `rubric` の意味

- `outcome`: 最終メモが最新前提、比較対象、推奨配分を正しく反映しているかを見る観点です。
- `process`: ニュース確認、依存順、比較セクター数、配分変更理由の説明などを見る観点です。
- `recovery`: 変化後の再計画速度と、曖昧な勝ち筋を断定しない回復行動を見る観点です。

## 7. finance 固有の `reference_data` キー

### 7.1 市場テーマと状態

- `market_theme`: 今回の分析テーマです。例: `AI capex`、`higher-for-longer rates`、`oil shock`。
- `headline_summary`: 現時点の材料要約です。平常時 view でも、news event 後の新 headline でも使います。
- `news_state`: ニュース確認状態です。既存ケースでは主に次を使います。  
  - `unconfirmed`: まだ材料確認が不十分。断定推奨に進むべきではない。  
  - `confirmed_material_change`: view を更新すべき十分な変化が確認された。  
  - `partially_confirmed`: 一部は確認できたが、勝ち筋や解釈にまだ不確実性が残る。
- `current_leader_sector`: 現時点で主導していると見ているセクターです。
- `losing_sector`: 崩れた、または崩れつつあるセクターです。平常時は `null` のままにできます。
- `baseline_stability_reason`: なぜ今の主導セクターがまだ強いと見えるのかを、ニュース前提込みで具体的に説明する文章です。
- `rotation_status`: ローテーション判断の状態です。既存ケースでは主に次を使います。  
  - `rotation_under_review`: 回転の可能性は見ているが、まだ検証中。  
  - `ambiguous`: 勝ち筋が複数あり、単一セクターへ断定できない。
- `decision_goal`: 最終的に何を判断したいかです。例: AI インフラ比率を下げて電力設備を増やすべきか。
- `forecast_horizon`: どの期間の見通しで議論しているかです。例: `1w`、`1m`、`quarter`。

### 7.2 比較対象と因果リンク

- `winning_sector_candidates`: 資金が移りうる候補セクター一覧です。
- `sector_links`: 主導セクターが崩れたときに、なぜ別セクターへ資金が回るのかを説明する因果リンクです。
- `sector_universe`: 今回の比較対象として明示的に見るセクター集合です。最低比較数の制約と対応します。
- `watchlist_names`: 監視対象銘柄やバスケット名です。sector より一段具体的な観測対象を置きます。

### 7.3 配分関連

- `sector_weights`: 現在のセクター配分です。キーはセクター名、値は通常 0 から 1 の weight です。
- `target_weight_template`: event がなければ目標にしたい推奨配分の初期案です。新制約が入ったらこれを書き換える前提で使います。
- `budget_unit`: `budget` が何を意味するかの補足です。既存テンプレートでは `portfolio weight points (100 = full sector allocation budget)` を使います。

### 7.4 `correlation_map` の各項目

`correlation_map` は、セクター間の関係性をオブジェクト配列で持ちます。

- `pair`: 関係を見る 2 セクターです。
- `relation`: 関係の種類です。既存ケースでは主に `positive`、`negative`、`rotation` を使います。  
  - `positive`: 同方向に動きやすい。  
  - `negative`: 逆方向に動きやすい。  
  - `rotation`: 資金の移り先として比較しやすい関係。
- `strength`: 関係の強さです。`low` / `medium` / `high` を想定します。
- `note`: その関係をどう解釈するかの説明です。

## 8. セクター名そのものの扱い

`ai_server_supply_chain`、`power_grid_equipment`、`banks`、`utilities` のような文字列は、schema の固定キーではなく「ケース固有のセクター ID」です。  
これらは次の場所で同じ名前を一貫して使います。

- `current_leader_sector`
- `losing_sector`
- `winning_sector_candidates`
- `sector_universe`
- `sector_weights`
- `target_weight_template`
- `correlation_map[*].pair`

つまり finance ケースでは、「セクター名の一貫性」自体が重要な state です。

## 9. `required_fields` の読み方

finance ケースの `required_fields` はほぼ固定で、意味は次のとおりです。

### 9.1 `news_reaction_note`

- `baseline_view`: ニュース前の基本見通し。
- `headline_summary`: 今見ているニュース要約。
- `losing_sector`: 崩れたセクター。
- `winning_sector_candidates`: 回転先候補。
- `current_sector_weights`: 現在配分。
- `recommended_sector_weights`: 推奨配分。
- `weight_change_rationale`: なぜその配分変更にしたかの理由。
- `rotation_reasoning`: どの因果で資金移動を想定するか。
- `positioning_implication`: ポジショニング上の含意。

### 9.2 `sector_watch_update`

- `baseline_checks`: 平常時に確認すべき前提チェック。
- `news_checklist`: ニュース確認項目。
- `correlation_map`: セクター間関係。
- `rebalance_actions`: 実際に行う配分見直しアクション。
- `risk_scenarios`: 反証条件や downside scenario。
- `sectors_to_watch`: 継続監視すべきセクター。
- `next_checkpoints`: 次に確認すべき時点や材料。
