#!/usr/bin/env python3
"""通过 REPLICATE_API_TOKEN 调用 Replicate 上 tencent/hunyuan-3d-3.1 做文生 3D 测试。"""

import argparse
import json
import os
import sys
import time

import requests

MODEL_OWNER = "tencent"
MODEL_NAME = "hunyuan-3d-3.1"
PREDICTIONS_URL = f"https://api.replicate.com/v1/models/{MODEL_OWNER}/{MODEL_NAME}/predictions"
POLL_INTERVAL_S = 3.0
# 3D 生成耗时通常较长
POLL_TIMEOUT_S = 2400

DEFAULT_PROMPT = (
    "An elegant and opulent 3D chair shaped like an avocado, the outer shell forming a deep green "
    "velvet seat, the pit replaced with a polished golden cushion, luxurious materials, ornate "
    "carved gold trim, subtle botanical motifs, rich emerald and gold color palette, dramatic "
    "studio lighting, ultra-detailed textures, photorealistic 3D render, high-end interior "
    "design style, soft shadows, 4K, centered on a dark neutral background"
)


def _headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "User-Agent": "TabletopGen hunyuan-3d-3.1 test",
    }


def wait_prediction(get_url: str, token: str) -> dict:
    headers = _headers(token)
    del headers["Content-Type"]
    deadline = time.monotonic() + POLL_TIMEOUT_S
    while time.monotonic() < deadline:
        r = requests.get(get_url, headers=headers, timeout=120)
        if r.status_code != 200:
            print(f"错误: 轮询预测失败 HTTP {r.status_code}: {r.text[:500]}", file=sys.stderr)
            sys.exit(1)
        pred = r.json()
        status = pred.get("status")
        if status == "succeeded":
            return pred
        if status in ("failed", "canceled"):
            err = pred.get("error") or pred.get("logs") or pred
            print(f"错误: 预测 {status}: {err}", file=sys.stderr)
            sys.exit(1)
        time.sleep(POLL_INTERVAL_S)
    print("错误: 等待预测结果超时。", file=sys.stderr)
    sys.exit(1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="文本描述（与官方示例一致时可省略，使用默认长 prompt）",
    )
    parser.add_argument(
        "--face-count",
        type=int,
        default=500_000,
        help="输出网格面数（默认 500000，与 Replicate 示例一致）",
    )
    parser.add_argument(
        "--generate-type",
        choices=("Normal", "Geometry"),
        default="Normal",
        help="Normal=带纹理；Geometry=白模无纹理",
    )
    parser.add_argument(
        "--enable-pbr",
        action="store_true",
        help="启用 PBR 材质（默认关闭，与 Replicate 示例一致）",
    )
    args = parser.parse_args()

    token = os.environ.get("REPLICATE_API_TOKEN", "").strip()
    if not token:
        print("错误: 未设置环境变量 REPLICATE_API_TOKEN。", file=sys.stderr)
        return 1

    if len(args.prompt) > 1024:
        print("错误: prompt 超过模型上限 1024 字符。", file=sys.stderr)
        return 1

    payload = {
        "input": {
            "prompt": args.prompt,
            "enable_pbr": args.enable_pbr,
            "face_count": args.face_count,
            "generate_type": args.generate_type,
        }
    }

    try:
        r = requests.post(
            PREDICTIONS_URL,
            headers=_headers(token),
            json=payload,
            timeout=60,
        )
    except requests.RequestException as e:
        print(f"错误: 创建预测失败（网络）: {e}", file=sys.stderr)
        return 1

    try:
        created = r.json()
    except json.JSONDecodeError:
        print(
            f"错误: 创建预测响应不是 JSON。HTTP {r.status_code}, {r.text[:500]}",
            file=sys.stderr,
        )
        return 1

    if r.status_code not in (200, 201):
        err = created.get("detail") or created.get("title") or created
        print(f"错误: 创建预测 HTTP {r.status_code}: {err}", file=sys.stderr)
        return 1

    get_url = (created.get("urls") or {}).get("get")
    if not get_url:
        print("错误: 响应中无预测轮询 URL。", file=sys.stderr)
        print(json.dumps(created, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1

    pred = wait_prediction(get_url, token)
    out = pred.get("output")
    print(f"模型: {MODEL_OWNER}/{MODEL_NAME}")
    print(f"generate_type: {args.generate_type}, face_count: {args.face_count}, enable_pbr: {args.enable_pbr}")
    print("--- 输出（.glb 等资源的 URL）---")
    if isinstance(out, str):
        print(out)
    else:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
