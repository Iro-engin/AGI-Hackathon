# Runner

## 目的

`run_benchmark.py` は、ケース JSON を読み込み、最低限の妥当性確認とサマリ表示を行うための最小 runner である。

現時点の役割は以下である。

- スキーマ準拠の前提となる必須フィールド確認
- シナリオファイルの一覧取得
- ケースごとの基本情報表示
- 今後 evaluator を接続するための入口提供


## 使い方

```bash
python runner/run_benchmark.py
python runner/run_benchmark.py --domain meeting
python runner/run_benchmark.py --case scenarios/meeting/meeting_001.json
```


## 今後の拡張

- JSON Schema による厳密検証
- エージェント実行ログの入力
- ルールベース採点
- 結果保存
