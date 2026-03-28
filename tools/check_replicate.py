#!/usr/bin/env python3
"""检查本机能否通过 REPLICATE_API_TOKEN 正常访问 Replicate API。"""

import json
import os
import sys

from replicate.client import Client
from replicate.exceptions import ReplicateError


def main() -> int:
    token = os.environ.get("REPLICATE_API_TOKEN", "").strip()
    if not token:
        print("错误: 未设置环境变量 REPLICATE_API_TOKEN。", file=sys.stderr)
        return 1

    try:
        client = Client(api_token=token)
        acc = client.accounts.current()
    except ReplicateError as e:
        print(f"错误: 无法访问 Replicate API: {e}", file=sys.stderr)
        return 1

    if hasattr(acc, "model_dump"):
        data = acc.model_dump()
    elif hasattr(acc, "dict"):
        data = acc.dict()
    else:
        data = {k: getattr(acc, k) for k in ("type", "username", "name", "github_url")}

    print("Replicate 访问正常。")
    print(json.dumps(data, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
