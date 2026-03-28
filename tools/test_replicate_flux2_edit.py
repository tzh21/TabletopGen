#!/usr/bin/env python3
"""通过 REPLICATE_API_TOKEN 调用 Replicate 上 black-forest-labs/flux-2-dev 做图片编辑测试。"""

import argparse
import json
import os
import sys
from pathlib import Path

from replicate.client import Client
from replicate.exceptions import ModelError, ReplicateError

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = REPO_ROOT / "local/images/woman-by-car.jpg"
DEFAULT_PROMPT = "Replace the color of the car to blue."
MODEL_REF = "black-forest-labs/flux-2-dev"


def _print_output(out: object) -> None:
    print("--- 输出 ---")
    if isinstance(out, str):
        print(out)
        return
    if isinstance(out, list):
        for item in out:
            url = getattr(item, "url", None)
            print(url if url is not None else item)
        return
    url = getattr(out, "url", None)
    if url is not None:
        print(url)
        return
    print(json.dumps(out, ensure_ascii=False, default=str, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_IMAGE,
        help=f"输入图片路径（默认: {DEFAULT_IMAGE}）",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="编辑说明（英文 prompt）",
    )
    args = parser.parse_args()

    token = os.environ.get("REPLICATE_API_TOKEN", "").strip()
    if not token:
        print("错误: 未设置环境变量 REPLICATE_API_TOKEN。", file=sys.stderr)
        return 1

    image_path = args.image.resolve()
    if not image_path.is_file():
        print(f"错误: 找不到图片文件: {image_path}", file=sys.stderr)
        return 1

    client = Client(api_token=token)
    try:
        with image_path.open("rb") as img:
            out = client.run(
                MODEL_REF,
                input={
                    "prompt": args.prompt,
                    "input_images": [img],
                    "aspect_ratio": "match_input_image",
                    "go_fast": True,
                    "output_format": "jpg",
                    "output_quality": 80,
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
    except OSError as e:
        print(f"错误: 无法读取图片: {e}", file=sys.stderr)
        return 1

    print(f"模型: {MODEL_REF}")
    print(f"输入: {image_path}")
    print(f"prompt: {args.prompt}")
    _print_output(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
