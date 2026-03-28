#!/usr/bin/env python3
"""测试能否通过 OPENROUTER_API_KEY 与 OpenRouter 上的 bytedance-seed/seed-1.6 正常对话。"""

import argparse
import json
import os
import sys

import requests

CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "bytedance-seed/seed-1.6"
TIMEOUT_S = 120


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-m",
        "--message",
        default="你好，请用一句话介绍你自己。",
        help="发送给模型的用户消息（默认：简短中文自我介绍请求）",
    )
    args = parser.parse_args()

    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        print("错误: 未设置环境变量 OPENROUTER_API_KEY。", file=sys.stderr)
        return 1

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Referer": "https://github.com/TabletopGen",
        "X-OpenRouter-Title": "TabletopGen OpenRouter seed-1.6 chat test",
    }
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": args.message}],
    }

    try:
        r = requests.post(CHAT_URL, headers=headers, json=body, timeout=TIMEOUT_S)
    except requests.RequestException as e:
        print(f"错误: 请求失败（网络或超时）: {e}", file=sys.stderr)
        return 1

    try:
        data = r.json()
    except json.JSONDecodeError:
        print(
            f"错误: 响应不是合法 JSON。HTTP {r.status_code}, 正文: {r.text[:500]}",
            file=sys.stderr,
        )
        return 1

    if r.status_code != 200:
        err = data.get("error", {})
        msg = err.get("message", json.dumps(data, ensure_ascii=False)[:800])
        print(f"错误: OpenRouter 返回 HTTP {r.status_code}: {msg}", file=sys.stderr)
        return 1

    choices = data.get("choices") or []
    if not choices:
        print("错误: 响应中无 choices。", file=sys.stderr)
        print(json.dumps(data, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1

    msg = choices[0].get("message") or {}
    content = msg.get("content")
    if content is None or (isinstance(content, str) and not content.strip()):
        print("错误: 模型未返回有效文本内容。", file=sys.stderr)
        print(json.dumps(data, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1

    print(f"模型: {MODEL}")
    print("--- 助手回复 ---")
    print(content if isinstance(content, str) else json.dumps(content, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
