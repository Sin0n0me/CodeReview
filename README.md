# CodeReview

特定のブランチの最新commitと現在のブランチの最新commitの差分を取得し, GeminiAPIにレビューを依頼する簡易的なPythonコード

## 使い方

1. `GEMINI_API_KEY`という環境変数を作成しGeminiAPIを叩くための[APIキー]((https://ai.google.dev/gemini-api/docs/api-key))を渡しておく
2. `review_config.json`にレビュー対象のパスや任意のプロンプトを書き込む
3. `gemini_review.py`を実行
4. 実行に成功すると`review`配下に`gemini_review.py`実行時の日付(`YYYYMMDD_HHmmss`形式)名のディレクトリといくつかのファイルが生成される
5. レビューに成功しているとレビュー結果が`review.md`に記載されている

### 設定項目

* `project_path`
  * レビューを行う対象のプロジェクトのパス
* `targets`
  * `git diff`コマンドのファイル指定部分に渡す引数になる
    * ファイル指定をしない場合は空の配列にする
* `base_branch`
  * 比較対象元のブランチ
* `model`
  * GeminiAPIで使用可能なモデル名(モデルコード)を指定する
    * https://ai.google.dev/gemini-api/docs/models
* `prompt`
  * 任意のプロンプト
    * コード内でプロンプトを追記していないのでレビュー以外の出力も可能(ただしプロンプトの後に`git diff`による差分がくっ付く)

### 例

```json
{
  "project_path": "../your_project_directory",
  "targets": [
    "*.cpp",
    "*.hpp",
    "*.rs"
  ],
  "base_branch": "origin/master",
  "model": "gemini-2.5-pro",
  "prompt": [
    "以下はGitによる差分です. コードレビューを行ってください.",
    "## 情報",
    "* 自作言語のプロジェクト",
    "* 使用言語はRust",
    "## レビュー条件",
    "* レビュー結果はMarkdown記法",
    "* 必要に応じて公式ドキュメントなどを参照し正確な情報を参照し引用元のリンクを必ず記載",
    "* 変更はApproveできるか",
    "## チェック項目",
    "* 潜在的なバグや例外が含まれていないか",
    "* 命名規則が統一されているか",
    "* 複雑な条件式が作られていないか",
    "* 関数は適切な粒度か",
    "* Typoや不正確な命名がないか"
  ]
}
```
