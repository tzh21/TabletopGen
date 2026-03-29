"""
Pipeline Configuration Module.
"""

import os
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

# OpenRouter model id for text / vision chat (replaces former GPT-4.1 / GPT-5 routes).
DEFAULT_OPENROUTER_CHAT_MODEL = "qwen/qwen3.5-plus-02-15"

# Project root directory (parent of configs); needed before YAML helpers.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_openrouter_api_key(api_keys_section: Optional[Mapping[str, Any]] = None) -> str:
    """
    Resolve the OpenRouter-compatible API key (used as Bearer token).

    Precedence: api_keys.gpt_api_key from YAML, then OPENROUTER_API_KEY, then GPT_API_KEY.
    """
    api = dict(api_keys_section or {})
    yaml_key = (api.get("gpt_api_key") or "").strip()
    if yaml_key:
        return yaml_key
    for env_name in ("OPENROUTER_API_KEY", "GPT_API_KEY"):
        v = os.environ.get(env_name, "").strip()
        if v:
            return v
    raise ValueError(
        "Missing OpenRouter API key: set api_keys.gpt_api_key in configs/config.yaml, "
        "or environment variable OPENROUTER_API_KEY (or GPT_API_KEY)"
    )


def _api_keys_from_default_config_yaml() -> dict[str, Any]:
    cfg_path = PROJECT_ROOT / "configs" / "config.yaml"
    if not cfg_path.is_file():
        return {}
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return dict(cfg.get("api_keys", {}) or {})
    except Exception:
        return {}


def resolve_openrouter_chat_model(api_keys_section: Optional[Mapping[str, Any]] = None) -> str:
    """
    OpenRouter chat model for GPT-compatible text/vision calls.

    Precedence: api_keys.chat_model from YAML, then OPENROUTER_CHAT_MODEL, then default.
    When api_keys_section is None, reads api_keys from configs/config.yaml if present.
    """
    api = (
        dict(api_keys_section)
        if api_keys_section is not None
        else _api_keys_from_default_config_yaml()
    )
    yaml_model = (api.get("chat_model") or "").strip()
    if yaml_model:
        return yaml_model
    env_model = os.environ.get("OPENROUTER_CHAT_MODEL", "").strip()
    if env_model:
        return env_model
    return DEFAULT_OPENROUTER_CHAT_MODEL


PIPELINE_DIR = PROJECT_ROOT


def get_pipeline_dir() -> str:
    """Get the pipeline root directory (project root)."""
    return str(PIPELINE_DIR)


def get_output_scene_dir() -> str:
    """Get the output scene root directory."""
    return str(PIPELINE_DIR / "output_scene")


def get_scene_dir(scene_id: int) -> str:
    """Get the directory for a specific scene."""
    return str(Path(get_output_scene_dir()) / f"scene_{scene_id}")


def get_next_scene_id() -> int:
    """Automatically get the next available scene ID."""
    output_scene_dir = Path(get_output_scene_dir())

    if not output_scene_dir.exists():
        return 1

    existing_scenes = []
    for item in output_scene_dir.iterdir():
        if item.is_dir() and item.name.startswith("scene_"):
            try:
                scene_num = int(item.name.split("_")[1])
                existing_scenes.append(scene_num)
            except (ValueError, IndexError):
                continue

    if not existing_scenes:
        return 1
    return max(existing_scenes) + 1


def get_comfy_image_dir(scene_dir: str) -> str:
    """Get the image output directory."""
    return str(Path(scene_dir) / "comfy_image")


def get_output_assets_dir(scene_dir: str) -> str:
    """Get the assets output directory."""
    return str(Path(scene_dir) / "output_assets")


def set_working_directory_to_scene(scene_dir: str) -> str:
    """Switch working directory to the scene directory and return the original working directory."""
    original_cwd = os.getcwd()
    os.chdir(scene_dir)
    return original_cwd


def restore_working_directory(original_cwd: str) -> None:
    """Restore the working directory."""
    os.chdir(original_cwd)


def get_configs_dir() -> str:
    """Get the configs directory."""
    return str(PROJECT_ROOT / "configs")
