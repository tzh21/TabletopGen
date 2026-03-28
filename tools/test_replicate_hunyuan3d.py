#!/usr/bin/env python3
"""通过 REPLICATE_API_TOKEN 调用 Replicate 上 tencent/hunyuan-3d-3.1 做文生 3D 测试。"""

import argparse
import json
import os
import sys

from replicate.client import Client
from replicate.exceptions import ModelError, ReplicateError

MODEL_REF = "tencent/hunyuan-3d-3.1"

DEFAULT_PROMPT = (
    "An elegant and opulent 3D chair shaped like an avocado, the outer shell forming a deep green "
    "velvet seat, the pit replaced with a polished golden cushion, luxurious materials, ornate "
    "carved gold trim, subtle botanical motifs, rich emerald and gold color palette, dramatic "
    "studio lighting, ultra-detailed textures, photorealistic 3D render, high-end interior "
    "design style, soft shadows, 4K, centered on a dark neutral background"
)


def _print_output(out: object) -> None:
    print("--- 输出（.glb 等资源的 URL）---")
    if isinstance(out, str):
        print(out)
        return
    if isinstance(out, list):
        for item in out:
            u = getattr(item, "url", None)
            print(u if u is not None else item)
        return
    url = getattr(out, "url", None)
    if url is not None:
        print(url)
        return
    print(json.dumps(out, ensure_ascii=False, default=str, indent=2))


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

    client = Client(api_token=token)
    try:
        out = client.run(
            MODEL_REF,
            input={
                "prompt": args.prompt,
                "enable_pbr": args.enable_pbr,
                "face_count": args.face_count,
                "generate_type": args.generate_type,
            },
            use_file_output=False,
        )
    except ModelError as e:
        pred = getattr(e, "prediction", None)
        detail = getattr(pred, "error", None) if pred else None
        logs = getattr(pred, "logs", None) if pred else None
        print(f"错误: 模型预测失败: {e}", file=sys.stderr)
        if detail:
            print(detail, file=sys.stderr)
        if logs:
            print(logs, file=sys.stderr)
        return 1
    except ReplicateError as e:
        print(f"错误: Replicate API: {e}", file=sys.stderr)
        return 1

    print(f"模型: {MODEL_REF}")
    print(
        f"generate_type: {args.generate_type}, face_count: {args.face_count}, "
        f"enable_pbr: {args.enable_pbr}"
    )
    _print_output(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
