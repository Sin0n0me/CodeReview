import subprocess
import os
import sys
import json
import datetime
import time
from pathlib import Path
from google import genai


JSON_KEY_PROJECT_PATH = "project_path"
JSON_KEY_BASE_BRANCH = "base_branch"
JSON_KEY_MODEL = "model"
JSON_KEY_PROMPT = "prompt"
JSON_KEY_TARGETS = "targets"
META_KEY_TIMESTAMP = "date"
META_KEY_PROJECT_PATH = "project path"
META_KEY_BASE_BRANCH = "base branch"
META_KEY_BASE_COMMIT = "base commit"
META_KEY_REVIEW_TARGET_BRANCH = "review target branch"
META_KEY_REVIEW_TARGET_COMMIT = "review target commit"
META_KEY_MODEL = "model"
META_KEY_PROMPT = "prompt"
META_KEY_TARGETS = "targets"


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


def get_latest_commit(branch: str, cwd) -> str:
    return run_cmd(["git", "rev-parse", branch], cwd=cwd)


def get_diff(from_commit: str, to_commit: str, targets: str, cwd) -> str:
    if targets == "":
        return run_cmd(
            [
                "git",
                "diff",
                "--minimal",
                f"{from_commit}..{to_commit}",
            ],
            cwd=cwd,
        )
    else:
        return run_cmd(
            [
                "git",
                "diff",
                "--minimal",
                f"{from_commit}..{to_commit}",
                targets,
            ],
            cwd=cwd,
        )


def load_config(path="review_config.json"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)

            # promptが配列なら連結
            if isinstance(config.get(JSON_KEY_PROMPT), list):
                config[JSON_KEY_PROMPT] = "\n".join(config[JSON_KEY_PROMPT])

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

            i += 1

    return None


def try_call_gemini_api(model: str, prompt: str):
    client = genai.Client()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )

    return response.text


def make_meta_data() -> dict[str, str]:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    config = load_config()
    project_path = config.get(JSON_KEY_PROJECT_PATH, ".")
    base_branch = config.get(JSON_KEY_BASE_BRANCH)
    model = config.get(JSON_KEY_MODEL)
    prompt = config.get(JSON_KEY_PROMPT)
    targets = config.get(JSON_KEY_TARGETS)

    # Git操作を指定ディレクトリで行う
    project_dir = Path(project_path)
    if not (project_dir / ".git").exists():
        print(
            f"ERROR: 指定されたパスに .git が見つかりません: {project_dir}",
            file=sys.stderr,
        )
        return None

    current_branch = get_current_branch(project_dir)[2:]
    base_commit = get_latest_commit(base_branch, project_dir)
    head_commit = get_latest_commit(current_branch, project_dir)

    # targetが配列なら連結
    if isinstance(targets, list):
        targets = " ".join(map(lambda target: ":/" + target, targets))

    return {
        META_KEY_TIMESTAMP: timestamp,
        META_KEY_PROJECT_PATH: str(project_dir.resolve()),
        META_KEY_TARGETS: targets,
        META_KEY_BASE_BRANCH: base_branch,
        META_KEY_BASE_COMMIT: base_commit,
        META_KEY_REVIEW_TARGET_BRANCH: current_branch,
        META_KEY_REVIEW_TARGET_COMMIT: head_commit,
        META_KEY_MODEL: model,
        META_KEY_PROMPT: prompt,
    }


def make_result(review_result: str, diff_text: str, meta_info: dict[str, str]):
    timestamp = meta_info[META_KEY_TIMESTAMP]

    # 結果を保存
    # 出力ディレクトリの作成
    output_dir = Path(f"./review/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    diff_path = output_dir / "diff.txt"
    review_path = output_dir / "review.md"
    meta_path = output_dir / "meta.txt"

    # 差分をファイルに保存
    diff_path.write_text(diff_text, encoding="utf-8")
    print(f"差分を {diff_path} に保存しました")

    meta_info_str = json.dumps(meta_info, indent=4, ensure_ascii=False)
    meta_path.write_text(
        meta_info_str,
        encoding="utf-8",
    )
    print(f"比較情報を {meta_path} に保存しました")

    # レビュー結果を保存
    if review_result is None:
        print("レビューに失敗しました")
    else:
        review_path.write_text(review_result, encoding="utf-8")
        print(f"レビュー結果を {review_path} に保存しました")
        print("\n---- コードレビュー結果(抜粋) ----\n")
        print(review_result[:2000])


def main():
    meta_info = make_meta_data()
    if meta_info is None:
        return

    project_dir = meta_info[META_KEY_PROJECT_PATH]
    targets = meta_info[META_KEY_TARGETS]
    base_branch = meta_info[META_KEY_BASE_BRANCH]
    current_branch = meta_info[META_KEY_REVIEW_TARGET_BRANCH]
    base_commit = meta_info[META_KEY_BASE_COMMIT]
    target_commit = meta_info[META_KEY_REVIEW_TARGET_COMMIT]
    model = meta_info[META_KEY_MODEL]
    prompt = meta_info[META_KEY_PROMPT]

    print(f"レビュー対象プロジェクト: {project_dir}")
    print(f"ベースブランチ: {base_branch}")
    print(f"使用モデル: {model}")
    print(f"現在のブランチ: {current_branch}")
    print(f"{base_branch} の最新コミット: {base_commit}")
    print(f"{current_branch} の最新コミット: {target_commit}")

    if base_commit == target_commit:
        print("差分なし（レビュー対象なし）")
        return

    diff_text = get_diff(base_commit, target_commit, targets, project_dir)
    if not diff_text.strip():
        print("差分が空です")
        return

    # レビュー依頼
    print("Geminiにレビューを依頼しています...(数分かかる場合があります)")
    review_result = None
    review_result = call_gemini_api(model, f"{prompt}\n```{diff_text}```")

    # レビュー結果
    make_result(review_result, diff_text, meta_info)


if __name__ == "__main__":
    main()
