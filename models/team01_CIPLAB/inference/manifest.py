from __future__ import annotations

import json
from pathlib import Path


def _resolve_image_path(base_dir: Path, path_value: str, field_name: str, sample_index: int) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()

    if not path.exists():
        raise ValueError(f"Sample {sample_index} has missing `{field_name}` image: {path}")

    return path


def load_manifest_entries(data_json_path: str, default_prompt: str | None = None):
    manifest_path = Path(data_json_path).expanduser().resolve()
    if not manifest_path.exists():
        raise ValueError(f"Inference manifest does not exist: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as handle:
        entries = json.load(handle)

    if isinstance(entries, dict):
        if isinstance(entries.get("items"), list):
            entries = entries["items"]
        elif isinstance(entries.get("samples"), list):
            entries = entries["samples"]

    if not isinstance(entries, list):
        raise ValueError("Inference manifest must be a JSON array of samples or an object with an `items` array.")

    samples = []
    has_custom_prompts = False
    base_dir = manifest_path.parent

    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Sample {index} must be a JSON object.")

        hr_path_value = entry.get("hr")
        cond_path_value = entry.get("res") or entry.get("lr")
        prompt_value = entry.get("prompt", default_prompt)

        if not hr_path_value:
            raise ValueError(f"Sample {index} is missing `hr`.")
        if not cond_path_value:
            raise ValueError(f"Sample {index} must include either `res` or `lr`.")
        if prompt_value is None or not str(prompt_value).strip():
            raise ValueError(f"Sample {index} is missing `prompt`, and no fallback prompt was provided.")

        if "prompt" in entry and str(entry["prompt"]).strip():
            has_custom_prompts = True

        samples.append(
            {
                "hr_path": str(_resolve_image_path(base_dir, hr_path_value, "hr", index)),
                "cond_path": str(_resolve_image_path(base_dir, cond_path_value, "res/lr", index)),
                "prompt": str(prompt_value).strip(),
            }
        )

    if not samples:
        raise ValueError("Inference manifest is empty.")

    return samples, has_custom_prompts
