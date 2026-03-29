"""
Replicate Python client wrappers for bytedance/seedream-4 and tencent/hunyuan-3d-3.1.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any, BinaryIO, List, Optional, Sequence, Union

import requests
from replicate.client import Client
from replicate.exceptions import ModelError, ReplicateError

MODEL_SEEDREAM = "bytedance/seedream-4"
MODEL_HUNYUAN = "tencent/hunyuan-3d-3.1"

# Hunyuan infers format from the image URI; Replicate file URLs often have no extension,
# which yields "Unsupported image format: ." — data URIs carry an explicit MIME type.
_MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}
_HUNYUAN_MAX_IMAGE_BYTES = 6 * 1024 * 1024


def resolve_replicate_token(replicate_api_token: Optional[str] = None) -> str:
    t = (replicate_api_token or os.environ.get("REPLICATE_API_TOKEN") or "").strip()
    if not t:
        raise ValueError("Missing api_keys.replicate_api_token or environment REPLICATE_API_TOKEN")
    return t


def _client(replicate_api_token: Optional[str]) -> Client:
    return Client(api_token=resolve_replicate_token(replicate_api_token))


def _extract_http_url(output: Any) -> str:
    if output is None:
        raise RuntimeError("Replicate returned empty output")
    if isinstance(output, str) and output.startswith("http"):
        return output
    if isinstance(output, list):
        if not output:
            raise RuntimeError("Replicate returned empty list")
        return _extract_http_url(output[0])
    url = getattr(output, "url", None)
    if url:
        return str(url)
    raise RuntimeError(f"Cannot extract HTTP URL from Replicate output: {output!r}")


def _run_model(
    client: Client,
    model_ref: str,
    input_dict: dict[str, Any],
) -> Any:
    try:
        return client.run(model_ref, input=input_dict, use_file_output=False)
    except ModelError as e:
        pred = getattr(e, "prediction", None)
        detail = getattr(pred, "error", None) if pred else None
        logs = getattr(pred, "logs", None) if pred else None
        msg = f"Replicate model error: {e}"
        if detail:
            msg += f"\n{detail}"
        if logs:
            msg += f"\n{logs}"
        raise RuntimeError(msg) from e
    except ReplicateError as e:
        raise RuntimeError(f"Replicate API error: {e}") from e


def image_uri_for_hunyuan3d(image_path: Union[str, Path]) -> str:
    """
    Build the `image` field for tencent/hunyuan-3d-3.1.

    Replicate-hosted file URLs often lack a file extension; Hunyuan then reports
    "Unsupported image format: .". A data URI embeds the MIME type explicitly.
    """
    path = Path(image_path)
    data = path.read_bytes()
    if len(data) > _HUNYUAN_MAX_IMAGE_BYTES:
        raise ValueError(
            f"Image too large for Hunyuan 3D (max {_HUNYUAN_MAX_IMAGE_BYTES} bytes): {path}"
        )
    mime = _MIME_BY_SUFFIX.get(path.suffix.lower()) or "image/png"
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def download_url_to_file(url: str, dest_path: Union[str, Path], timeout_s: int = 300) -> None:
    r = requests.get(url, timeout=timeout_s)
    r.raise_for_status()
    Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "wb") as f:
        f.write(r.content)


def run_seedream4(
    prompt: str,
    *,
    image_paths: Optional[Sequence[Union[str, Path]]] = None,
    replicate_api_token: Optional[str] = None,
    size: str = "2K",
    aspect_ratio: Optional[str] = None,
    sequential_image_generation: str = "disabled",
    max_images: int = 1,
    enhance_prompt: bool = True,
) -> str:
    """
    Run Seedream 4; returns HTTPS URL of the generated image.
    """
    client = _client(replicate_api_token)
    inp: dict[str, Any] = {
        "prompt": prompt,
        "size": size,
        "sequential_image_generation": sequential_image_generation,
        "max_images": max_images,
        "enhance_prompt": enhance_prompt,
    }
    paths = [Path(p) for p in (image_paths or []) if p]
    opened: List[BinaryIO] = []
    try:
        if paths:
            inp["aspect_ratio"] = aspect_ratio or "match_input_image"
            for p in paths:
                opened.append(open(p, "rb"))
            inp["image_input"] = opened
        else:
            inp["aspect_ratio"] = aspect_ratio or "4:3"
        out = _run_model(client, MODEL_SEEDREAM, inp)
    finally:
        for f in opened:
            try:
                f.close()
            except OSError:
                pass
    return _extract_http_url(out)


def run_seedream4_to_file(
    prompt: str,
    output_path: Union[str, Path],
    *,
    image_paths: Optional[Sequence[Union[str, Path]]] = None,
    replicate_api_token: Optional[str] = None,
    size: str = "2K",
    aspect_ratio: Optional[str] = None,
    sequential_image_generation: str = "disabled",
    max_images: int = 1,
    enhance_prompt: bool = True,
) -> str:
    """Generate image and save to disk; returns the image URL."""
    url = run_seedream4(
        prompt,
        image_paths=image_paths,
        replicate_api_token=replicate_api_token,
        size=size,
        aspect_ratio=aspect_ratio,
        sequential_image_generation=sequential_image_generation,
        max_images=max_images,
        enhance_prompt=enhance_prompt,
    )
    download_url_to_file(url, output_path)
    return url


def run_hunyuan3d_image_to_glb(
    image_path: Union[str, Path],
    output_glb_path: Union[str, Path],
    *,
    replicate_api_token: Optional[str] = None,
    enable_pbr: bool = True,
    face_count: int = 500_000,
    generate_type: str = "Normal",
) -> None:
    """
    Image-to-3D only (no prompt). Downloads GLB to output_glb_path.
    """
    client = _client(replicate_api_token)
    path = Path(image_path)
    inp = {
        "image": image_uri_for_hunyuan3d(path),
        "enable_pbr": enable_pbr,
        "face_count": face_count,
        "generate_type": generate_type,
    }
    out = _run_model(client, MODEL_HUNYUAN, inp)
    url = _extract_http_url(out)
    download_url_to_file(url, output_glb_path)
