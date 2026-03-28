#!/usr/bin/env python3
"""检查本机能否通过 OPENROUTER_API_KEY 正常访问 OpenRouter API。"""

import json
import os
import sys

import requests

CREDITS_URL = "https://openrouter.ai/api/v1/credits"
TIMEOUT_S = 30


def main() -> int:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        print("错误: 未设置环境变量 OPENROUTER_API_KEY。", file=sys.stderr)
        return 1

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        # OpenRouter 文档中的 Referer / 标题（用于统计来源）
        "Referer": "https://github.com/TabletopGen",
        "X-OpenRouter-Title": "TabletopGen connectivity check",
    }

    try:
        r = requests.get(CREDITS_URL, headers=headers, timeout=TIMEOUT_S)
    except requests.RequestException as e:
        print(f"错误: 无法连接到 OpenRouter（网络或代理问题）: {e}", file=sys.stderr)
        return 1

    if r.status_code != 200:
        try:
            err = r.json()
            msg = err.get("error", {}).get("message", r.text)
        except json.JSONDecodeError:
            msg = r.text[:500] if r.text else "(empty body)"
        print(
            f"错误: OpenRouter 返回 HTTP {r.status_code}: {msg}",
            file=sys.stderr,
        )
        return 1

    try:
        data = r.json()
    except json.JSONDecodeError:
        print("错误: 响应不是合法 JSON。", file=sys.stderr)
        return 1

    print("OpenRouter 访问正常。")
    print(json.dumps(data, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
