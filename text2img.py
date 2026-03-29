"""
Generate scene images based on text input.
    1. Generate prompt for scene description.
    2. Call Seedream on Replicate to generate image.
    3. Save to scene_image directory and record in input_text.json.
"""

import argparse
import json
import os
from pathlib import Path

import yaml

from configs.pipeline_config import resolve_openrouter_api_key
from modules.replicate_client import run_seedream4_to_file
from modules.setup_openai_client import setup_openai_client


def _load_api_config():
    root = Path(__file__).resolve().parent
    cfg_path = root / "configs" / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    api = cfg.get("api_keys", {}) or {}
    proxy_cfg = cfg.get("proxy") or {}
    rep = (api.get("replicate_api_token") or "").strip()
    if not rep:
        rep = (os.environ.get("REPLICATE_API_TOKEN") or "").strip()
    return {
        "gpt_api_key": resolve_openrouter_api_key(api),
        "base_url": (api.get("base_url") or "https://openrouter.ai/api/v1").strip(),
        "replicate_api_token": rep,
        "http_proxy": (proxy_cfg.get("http") or "").strip() or None,
    }


def generate_scene_prompt(scene_description, client):
    """
    Generate asset list json based on scene description
    """
    prompt = f"""
You are a 3D scene layout prompt designer. Your task is to Infer the possible items based on the scene description and layout the items. 
**Rules**: 
- First of all, a complete and head-on(front) table is needed. Every parts of the table must be very easy to be seen, especially any of the table foot!
- The angle more than 50 degrees and less than 60 degrees above the table.
- The objects placed should preferably be individual objects, not combined objects.
- "completely white background", " a head-on(front) table and every parts of the table is needed" and "The longer side of the table must be level! The bottommost foot of the desk's front legs must be fully displayed" must be added in end of the prompt. 
- Make the layout of objects more natural, ensure semantic rationality, and conform to real-world logic. When two or more items are semantically related, arrange them according to real-world logic. For example, place the pen to the right of the notebook, the mouse to the right of the keyboard, and the laptop in front of the keyboard. Avoid lining up all objects side by side.
- Ensure object independence for segmentation: To facilitate later segmentation, avoid arrangements where one object is contained within another. For example, pens should be laid directly on the desk, NOT placed inside a cup or pen holder. 
- All items should be placed separately and adjacent to each other.But when there are two or more same items, they must not be placed too close to each other，avoiding being attached together.
- Make sure that all possible objects are on the table.And all items on the table can be clearly seen.
- Keep in mind that the full view of the table and all items can be seen completely.
- Make sure there are only table and all objects on the table in the picture, without chairs etc.
- Avoid items that are too thin, such as a piece of paper. Avoid the same thing. Avoid a single fragile label sticker.
- If no specific items are specified, the number of items should not exceed seven.
Only output a short final prompt. The scene description is :"{scene_description}" """

    response = client.chat.completions.create(
        model="openai/gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content


def get_next_filename(output_dir, specified_id=None):
    """
    Get the next available filename

    Args:
        output_dir (str): Output directory
        specified_id (int, optional): Specified ID, if provided, use this ID

    Returns:
        tuple: (filename, full file path)
    """
    os.makedirs(output_dir, exist_ok=True)

    if specified_id is not None:
        next_num = specified_id
    else:
        existing_files = [f for f in os.listdir(output_dir) if f.startswith("scene_image_") and f.endswith(".png")]

        if not existing_files:
            next_num = 1
        else:
            numbers = []
            for f in existing_files:
                try:
                    num = int(f.replace("scene_image_", "").replace(".png", ""))
                    numbers.append(num)
                except ValueError:
                    continue
            next_num = max(numbers) + 1 if numbers else 1

    filename = f"scene_image_{next_num}.png"
    filepath = os.path.join(output_dir, filename)
    return filename, filepath


def update_input_text_json(output_dir, filename, input_text):
    """
    Update input_text.json file, record filename and corresponding input text
    """
    json_path = os.path.join(output_dir, "input_text.json")

    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}

    data[filename] = input_text

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✓ Updated input_text.json: {filename} -> {input_text[:50]}...")


def generate_scene_image_with_seedream(
    prompt_text, output_dir, replicate_api_token, input_text, specified_id=None
):
    """
    Generate scene image using Seedream 4 on Replicate (text-to-image).
    """
    try:
        filename, output_path = get_next_filename(output_dir, specified_id)

        print(f"Generating image: {filename}")

        run_seedream4_to_file(
            prompt_text,
            output_path,
            image_paths=None,
            replicate_api_token=replicate_api_token,
            size="2K",
            aspect_ratio="4:3",
            sequential_image_generation="disabled",
            max_images=1,
            enhance_prompt=True,
        )
        print(f"✓ Image saved to: {output_path}")

        update_input_text_json(output_dir, filename, input_text)

        return output_path

    except Exception as e:
        print(f"✗ Failed to generate image: {e}")
        return None


def parse_args():
    parser = argparse.ArgumentParser(description="Generate scene images using Replicate Seedream 4")
    parser.add_argument(
        "--id",
        type=int,
        default=None,
        help="Specify generated image ID (optional, auto-increment if not specified)",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="A hobby desk with some model cars and tools.",
        help="Scene description text",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    conf = _load_api_config()
    if not conf["gpt_api_key"]:
        raise SystemExit(
            "Missing OpenRouter key: set api_keys.gpt_api_key in configs/config.yaml, "
            "or OPENROUTER_API_KEY (or GPT_API_KEY) in the environment"
        )
    if not conf["replicate_api_token"]:
        raise SystemExit(
            "Missing Replicate token: set api_keys.replicate_api_token in configs/config.yaml, "
            "or REPLICATE_API_TOKEN in the environment"
        )

    client = setup_openai_client(conf["gpt_api_key"], conf["http_proxy"], conf["base_url"])

    input_text = args.text
    print("Generating scene prompt...")
    scene_prompt = generate_scene_prompt(input_text, client)
    print(f"Scene prompt: {scene_prompt}")

    PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
    output_directory = os.path.join(PIPELINE_DIR, "scene_image")

    print("\nGenerating scene image...")
    generate_scene_image_with_seedream(
        scene_prompt,
        output_directory,
        conf["replicate_api_token"],
        input_text,
        args.id,
    )
