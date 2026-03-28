"""
Single-object image-to-3D via Replicate tencent/hunyuan-3d-3.1 (image input only; no prompt).
"""

from pathlib import Path

from modules.replicate_client import run_hunyuan3d_image_to_glb


def gen_single_obj_hy3dapi(replicate_api_token, image_path, output_glb_path):
    """
    Generate a GLB mesh from a local RGBA/PNG object image using Hunyuan3D on Replicate.

    Args:
        replicate_api_token: Replicate API token (or None to use REPLICATE_API_TOKEN).
        image_path: Path to input image (Redraw crop).
        output_glb_path: Where to write the downloaded .glb file.
    """
    run_hunyuan3d_image_to_glb(
        Path(image_path),
        Path(output_glb_path),
        replicate_api_token=replicate_api_token,
        enable_pbr=True,
        face_count=500_000,
        generate_type="Normal",
    )
    print("Saved to:", output_glb_path)


if __name__ == "__main__":
    import os

    token = os.environ.get("REPLICATE_API_TOKEN", "").strip()
    image_path = "path/to/your/image.png"
    output_glb_path = "out.glb"
    if token and os.path.exists(image_path):
        gen_single_obj_hy3dapi(token, image_path, output_glb_path)
