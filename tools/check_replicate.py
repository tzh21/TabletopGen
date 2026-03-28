#!/usr/bin/env python3
"""检查本机能否通过 REPLICATE_API_TOKEN 正常访问 Replicate API。"""

import json
import os
import sys

import requests

ACCOUNT_URL = "https://api.replicate.com/v1/account"
TIMEOUT_S = 30


def main() -> int:
    token = os.environ.get("REPLICATE_API_TOKEN", "").strip()
    if not token:
        print("错误: 未设置环境变量 REPLICATE_API_TOKEN。", file=sys.stderr)
        return 1

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "User-Agent": "TabletopGen connectivity check",
    }

    try:
        r = requests.get(ACCOUNT_URL, headers=headers, timeout=TIMEOUT_S)
    except requests.RequestException as e:
        print(f"错误: 无法连接到 Replicate（网络或代理问题）: {e}", file=sys.stderr)
        return 1

    if r.status_code != 200:
        try:
            err = r.json()
            msg = err.get("detail") or err.get("title") or r.text
        except json.JSONDecodeError:
            msg = r.text[:500] if r.text else "(empty body)"
        print(
            f"错误: Replicate 返回 HTTP {r.status_code}: {msg}",
            file=sys.stderr,
        )
        return 1

    try:
        data = r.json()
    except json.JSONDecodeError:
        print("错误: 响应不是合法 JSON。", file=sys.stderr)
        return 1

    print("Replicate 访问正常。")
    print(json.dumps(data, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
