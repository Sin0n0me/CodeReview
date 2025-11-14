import subprocess
import os
import sys
import json
import datetime
import time
from pathlib import Path
from google import genai


def run_cmd(cmd, cwd=None) -> str:
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
        encoding="utf-8",
    )

    return result.stdout.strip()


def get_current_branch(cwd) -> str:
    return run_cmd(["git", "branch", "--contains"], cwd=cwd)


def get_latest_commit(branch, cwd) -> str:
    return run_cmd(["git", "rev-parse", branch], cwd=cwd)


def get_diff(from_commit, to_commit, cwd) -> str:
    return run_cmd(
        ["git", "diff", "--minimal", f"{from_commit}..{to_commit}"], cwd=cwd
    )


def load_config(path="review_config.json"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)

            # promptが配列なら連結
            if isinstance(config.get("prompt"), list):
                config["prompt"] = "\n".join(config["prompt"])

            return config
    except FileNotFoundError:
        print(f"ERROR: 設定ファイル {path} が見つかりません", file=sys.stderr)
        sys.exit(1)


def call_gemini_api(model: str, prompt: str) -> str:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if GEMINI_API_KEY is None:
        print(
            "ERROR: 環境変数 GEMINI_API_KEY が設定されていません",
            file=sys.stderr,
        )
        sys.exit(1)

    retry_time = [3, 30, 120, 300, 600]
    i = 0
    while True:
        try:
            return try_call_gemini_api(model, prompt)
        except genai.errors.APIError as e:
            i += 1

            if not hasattr(e, "code"):
                break

            if i > len(retry_time) - 1:
                break

            if e.code == 429:
                print("レート制限に達しています")
                break

            if e.code == 503:
                print(
                    "過負荷によりリクエストが拒否されました\n"
                    f"リトライします 待機時間: {retry_time[i]}s"
                )
                time.sleep(retry_time[i])

    return None


def try_call_gemini_api(model: str, prompt: str):
    client = genai.Client()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )

    return response.text


def main():
    config = load_config()
    project_path = config.get("project_path", ".")
    base_branch = config.get("base_branch")
    model = config.get("model")
    prompt = config.get("prompt")

    # Git操作を指定ディレクトリで行う
    project_dir = Path(project_path)
    if not (project_dir / ".git").exists():
        print(
            f"ERROR: 指定されたパスに .git が見つかりません: {project_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"レビュー対象プロジェクト: {project_dir}")
    print(f"ベースブランチ: {base_branch}")
    print(f"使用モデル: {model}")

    current_branch = get_current_branch(project_dir)[2:]
    base_commit = get_latest_commit(base_branch, project_dir)
    head_commit = get_latest_commit(current_branch, project_dir)

    print(f"現在のブランチ: {current_branch}")
    print(f"{base_branch} の最新コミット: {base_commit}")
    print(f"{current_branch} の最新コミット: {head_commit}")

    if base_commit == head_commit:
        print("差分なし（レビュー対象なし）")
        return

    diff_text = get_diff(base_commit, head_commit, project_dir)
    if not diff_text.strip():
        print("差分が空です")
        return

    print("Geminiにレビューを依頼しています...(数分かかる場合があります)")

    review_result = call_gemini_api(model, f"{prompt}\n```{diff_text}```")
    if review_result is None:
        print("レビューに失敗しました")
        return

    # 結果を保存
    # 出力ディレクトリの作成
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"./review/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    diff_path = output_dir / "diff.txt"
    review_path = output_dir / "review.md"
    meta_path = output_dir / "meta.txt"

    # 差分をファイルに保存
    diff_path.write_text(diff_text, encoding="utf-8")
    print(f"差分を {diff_path} に保存しました")

    # メタ情報ファイルを作成
    meta_info = (
        f"date: {timestamp}\n"
        f"project path: {project_dir.resolve()}\n"
        f"base branch: {base_branch}\n"
        f"base commit: {base_commit}\n"
        f"review target branch: {current_branch}\n"
        f"review target commit: {head_commit}\n"
        f"model: {model}\n"
        "----\n"
        "prompt:\n"
        f"{prompt}\n"
    )
    meta_path.write_text(meta_info, encoding="utf-8")
    print(f"比較情報を {meta_path} に保存しました")

    # レビュー結果を保存
    review_path.write_text(review_result, encoding="utf-8")
    print(f"レビュー結果を {review_path} に保存しました")

    print("\n---- コードレビュー結果(抜粋) ----\n")
    print(review_result[:2000])


if __name__ == "__main__":
    main()
