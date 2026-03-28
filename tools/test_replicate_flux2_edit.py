#!/usr/bin/env python3
"""通过 REPLICATE_API_TOKEN 调用 Replicate 上 black-forest-labs/flux-2-dev 做图片编辑测试。"""

import argparse
import base64
import json
import mimetypes
import os
import sys
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = REPO_ROOT / "local/images/woman-by-car.jpg"
DEFAULT_PROMPT = "Replace the color of the car to blue."
MODEL_OWNER = "black-forest-labs"
MODEL_NAME = "flux-2-dev"
PREDICTIONS_URL = f"https://api.replicate.com/v1/models/{MODEL_OWNER}/{MODEL_NAME}/predictions"
FILES_URL = "https://api.replicate.com/v1/files"
# Replicate：小文件可直接用 data URL；见 predictions 文档
MAX_DATA_URI_BYTES = 256 * 1024
POLL_INTERVAL_S = 2.0
POLL_TIMEOUT_S = 600


def _headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "User-Agent": "TabletopGen flux-2-dev edit test",
    }


def _file_headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "User-Agent": "TabletopGen flux-2-dev edit test",
    }


def image_uri_for_input(path: Path, token: str) -> str:
    data = path.read_bytes()
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    if len(data) <= MAX_DATA_URI_BYTES:
        b64 = base64.standard_b64encode(data).decode("ascii")
        return f"data:{mime};base64,{b64}"

    with path.open("rb") as f:
        r = requests.post(
            FILES_URL,
            headers=_file_headers(token),
            files={"content": (path.name, f, mime)},
            timeout=120,
        )
    if r.status_code not in (200, 201):
        try:
            err = r.json()
            msg = err.get("detail") or err.get("title") or r.text
        except json.JSONDecodeError:
            msg = r.text[:500] if r.text else "(empty body)"
        raise RuntimeError(f"上传输入图失败 HTTP {r.status_code}: {msg}")

    body = r.json()
    url = (body.get("urls") or {}).get("get") or body.get("url")
    if not url:
        raise RuntimeError(f"上传响应中无可用 URL: {body}")
    return url


def wait_prediction(get_url: str, token: str) -> dict:
    headers = _headers(token)
    del headers["Content-Type"]
    deadline = time.monotonic() + POLL_TIMEOUT_S
    while time.monotonic() < deadline:
        r = requests.get(get_url, headers=headers, timeout=60)
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

    try:
        image_ref = image_uri_for_input(image_path, token)
    except OSError as e:
        print(f"错误: 无法读取图片: {e}", file=sys.stderr)
        return 1
    except RuntimeError as e:
        print(f"错误: {e}", file=sys.stderr)
        return 1

    payload = {
        "input": {
            "prompt": args.prompt,
            "input_images": [image_ref],
            "aspect_ratio": "match_input_image",
            "go_fast": True,
            "output_format": "jpg",
            "output_quality": 80,
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
    print(f"输入: {image_path}")
    print(f"prompt: {args.prompt}")
    print("--- 输出 ---")
    if isinstance(out, str):
        print(out)
    else:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
