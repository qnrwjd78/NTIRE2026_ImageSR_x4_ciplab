from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


TEAM_DIR = Path(__file__).resolve().parent
INFERENCE_DIR = TEAM_DIR / "inference"
DEFAULT_CONFIG_NAMES = ("inference_config.json", "config.json")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def _resolve_config_path(model_dir: str | None) -> Path:
    if not model_dir:
        for name in DEFAULT_CONFIG_NAMES:
            candidate = TEAM_DIR / name
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"No inference config found in {TEAM_DIR}. Expected one of: {', '.join(DEFAULT_CONFIG_NAMES)}"
        )

    path = Path(model_dir).expanduser().resolve()
    if path.is_file():
        return path

    if path.is_dir():
        for name in DEFAULT_CONFIG_NAMES:
            candidate = path / name
            if candidate.exists():
                return candidate

    raise FileNotFoundError(f"Could not resolve an inference config from: {path}")


def _load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Inference config must be a JSON object: {config_path}")
    return data


def _stringify_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _build_manifest(input_path: str) -> Path:
    input_dir = Path(input_path).expanduser().resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path must be a directory of images: {input_dir}")

    items = []
    for image_path in sorted(input_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS or not image_path.is_file():
            continue
        # The copied inference expects both `hr` and `res/lr`. For challenge inference we reuse the input image path.
        items.append({"hr": str(image_path), "res": str(image_path)})

    if not items:
        raise ValueError(f"No input images found under: {input_dir}")

    fd, temp_path = tempfile.mkstemp(prefix="ciplab_infer_", suffix=".json")
    os.close(fd)
    Path(temp_path).write_text(json.dumps(items, indent=2), encoding="utf-8")
    return Path(temp_path)


def _build_cli_args(config: dict, input_json: Path, output_path: str, device=None) -> list[str]:
    args = [
        "--input_json",
        str(input_json),
        "--output_dir",
        str(Path(output_path).expanduser().resolve()),
    ]

    for key, value in config.items():
        if value is None:
            continue
        option = f"--{key}"
        if isinstance(value, bool):
            if value:
                args.append(option)
            continue
        if isinstance(value, list):
            if not value:
                continue
            joined = ",".join(_stringify_value(item) for item in value)
            args.extend([option, joined])
            continue
        if value == "":
            continue
        args.extend([option, _stringify_value(value)])

    if device is not None:
        args.extend(["--device", str(device)])

    return args


def main(model_dir=None, input_path=None, output_path=None, device=None):
    if input_path is None or output_path is None:
        raise ValueError("`input_path` and `output_path` are required.")

    from .inference import lora_inference

    config_path = _resolve_config_path(model_dir)
    config = _load_config(config_path)
    manifest_path = _build_manifest(input_path)

    try:
        cli_args = _build_cli_args(config, manifest_path, output_path, device=device)
        lora_inference.main(cli_args)
    finally:
        manifest_path.unlink(missing_ok=True)
