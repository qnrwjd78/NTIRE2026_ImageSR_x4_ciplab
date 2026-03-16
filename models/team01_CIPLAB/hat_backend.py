from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


TEAM_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEAM_DIR.parent.parent
WORKSPACE_ROOT = REPO_ROOT.parent.parent
OPTIONS_DIR = TEAM_DIR / "options"
RESULTS_DIR = TEAM_DIR / "results"

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
DEFAULT_MODEL_KEY = "hat"
DEFAULT_WEIGHT_KEY = "hat_l_srx4_imagenet_pretrain"
DEFAULT_TEST_NAME = "hat_l_srx4_imagenet_pretrain"
DEFAULT_WEIGHT_FILENAME = "HAT-L_SRx4_ImageNet-pretrain.pth"
DEFAULT_WEIGHT_PATH = REPO_ROOT / "model_zoo" / "team01_CIPLAB" / DEFAULT_WEIGHT_FILENAME
DEFAULT_STAGE_ROOT = Path(
    os.environ.get("CIPLAB_HAT_STAGE_ROOT", "/tmp/team01_ciplab_hat_stage")
).expanduser()


@dataclass(frozen=True)
class SampleRef:
    lr_path: Path
    hr_path: str | None
    staged_name: str
    output_name: str


@dataclass(frozen=True)
class StageBundle:
    stage_dir: Path
    lr_dir: Path
    dataset_name: str
    samples: list[SampleRef]


@dataclass(frozen=True)
class HatRuntime:
    opt_path: Path
    weight_path: Path


def _safe_name(name: str) -> str:
    sanitized = "".join(char if char.isalnum() or char in "._-" else "_" for char in name.strip())
    return sanitized[:80] or "team01_hat"


def _load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"JSON not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON: {path}\n{error}") from error


def _resolve_override_path(raw_path: str, label: str) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        search_candidates = [candidate]
    else:
        search_candidates = [
            REPO_ROOT / candidate,
            WORKSPACE_ROOT / candidate,
            candidate,
        ]

    for item in search_candidates:
        resolved = item.resolve()
        if resolved.exists():
            return resolved

    searched = ", ".join(str(path.resolve()) for path in search_candidates)
    raise FileNotFoundError(f"Could not resolve {label}: {raw_path}. Tried: {searched}")


def _resolve_existing_input_path(raw_path: str, label: str, base_dir: Path | None = None) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        search_candidates = [candidate]
    else:
        search_candidates = [WORKSPACE_ROOT / candidate]
        if base_dir is not None:
            search_candidates.append(base_dir / candidate)
        search_candidates.extend([REPO_ROOT / candidate, candidate])

    for item in search_candidates:
        resolved = item.resolve()
        if resolved.exists():
            return resolved

    searched = ", ".join(str(path.resolve()) for path in search_candidates)
    raise FileNotFoundError(f"Could not resolve {label}: {raw_path}. Tried: {searched}")


def _resolve_output_path(raw_path: str, base_dir: Path | None = None) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    workspace_candidate = (WORKSPACE_ROOT / candidate).resolve()
    if workspace_candidate.parent.exists() or base_dir is None:
        return workspace_candidate
    return (base_dir / candidate).resolve()


def _has_image_files(path: Path) -> bool:
    return any(item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS for item in path.iterdir())


def resolve_input_dir(input_path: str) -> Path:
    input_dir = Path(input_path).expanduser().resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path must be a directory of images: {input_dir}")

    if _has_image_files(input_dir):
        return input_dir

    for child_name in ("LQ", "lq", "LR", "lr", "input", "inputs"):
        child_dir = input_dir / child_name
        if child_dir.is_dir() and _has_image_files(child_dir):
            return child_dir.resolve()

    raise ValueError(
        f"No input images found under: {input_dir}. "
        "Expected images directly in the folder or under an `LQ` subdirectory."
    )


def _list_image_paths(input_dir: Path) -> list[Path]:
    image_paths = [
        item.resolve()
        for item in sorted(input_dir.iterdir())
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not image_paths:
        raise ValueError(f"No input images found under: {input_dir}")
    return image_paths


def _resolve_weight_path() -> Path:
    for env_name in ("CIPLAB_HAT_WEIGHT_PATH", "CIPLAB_HAT_WEIGHT"):
        raw_path = os.environ.get(env_name)
        if raw_path:
            resolved = _resolve_override_path(raw_path, "HAT weight")
            if resolved.is_file():
                return resolved
            raise FileNotFoundError(f"HAT weight must be a file: {resolved}")

    if DEFAULT_WEIGHT_PATH.is_file():
        return DEFAULT_WEIGHT_PATH.resolve()

    team_model_zoo_dir = DEFAULT_WEIGHT_PATH.parent
    if team_model_zoo_dir.is_dir():
        matches = sorted(team_model_zoo_dir.rglob(DEFAULT_WEIGHT_FILENAME))
        if matches:
            return matches[0].resolve()

    raise FileNotFoundError(
        f"Could not find `{DEFAULT_WEIGHT_FILENAME}` under `{team_model_zoo_dir}`. "
        "Set `CIPLAB_HAT_WEIGHT_PATH` if your layout is different."
    )


def resolve_runtime(weight_key: str = DEFAULT_WEIGHT_KEY) -> HatRuntime:
    if weight_key != DEFAULT_WEIGHT_KEY:
        raise ValueError(
            f"Unsupported weight: {weight_key!r}. "
            f"Only `{DEFAULT_WEIGHT_KEY}` is implemented in team01_CIPLAB."
        )

    opt_path = OPTIONS_DIR / "hat_l_srx4_imagenet_pretrain.yml"
    if not opt_path.is_file():
        raise FileNotFoundError(f"Missing HAT option file: {opt_path}")

    return HatRuntime(
        opt_path=opt_path.resolve(),
        weight_path=_resolve_weight_path(),
    )


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def _validate_unique_stems(samples: list[SampleRef]) -> None:
    seen: dict[str, Path] = {}
    for sample in samples:
        stem = Path(sample.staged_name).stem
        previous = seen.get(stem)
        if previous is not None and previous != sample.lr_path:
            raise ValueError(
                "Duplicate LR stem detected. HAT outputs are keyed by stem, so these would overwrite "
                f"each other: {previous} and {sample.lr_path}"
            )
        seen[stem] = sample.lr_path


def _make_stage_dir(name_prefix: str) -> Path:
    DEFAULT_STAGE_ROOT.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f"{_safe_name(name_prefix)}_", dir=str(DEFAULT_STAGE_ROOT))).resolve()


def _build_stage_bundle(samples: list[SampleRef], name_prefix: str) -> StageBundle:
    if not samples:
        raise ValueError("At least one sample is required for HAT inference.")

    _validate_unique_stems(samples)

    stage_dir = _make_stage_dir(name_prefix)
    lr_dir = (stage_dir / "LR").resolve()
    lr_dir.mkdir(parents=True, exist_ok=True)

    for sample in samples:
        _link_or_copy(sample.lr_path, lr_dir / sample.staged_name)

    return StageBundle(
        stage_dir=stage_dir,
        lr_dir=lr_dir,
        dataset_name=stage_dir.name,
        samples=samples,
    )


def build_stage_bundle_from_input_dir(input_path: str, name_prefix: str = DEFAULT_TEST_NAME) -> StageBundle:
    input_dir = resolve_input_dir(input_path)
    image_paths = _list_image_paths(input_dir)
    samples = [
        SampleRef(
            lr_path=image_path,
            hr_path=None,
            staged_name=image_path.name,
            output_name=f"{image_path.stem}.png",
        )
        for image_path in image_paths
    ]
    return _build_stage_bundle(samples, name_prefix)


def build_stage_bundle_from_manifest_json(manifest_path: Path, name_prefix: str) -> StageBundle:
    data = _load_json(manifest_path)
    if not isinstance(data, list) or not data:
        raise ValueError(f"Manifest must be a non-empty list: {manifest_path}")

    samples: list[SampleRef] = []
    for index, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise ValueError(f"Manifest entry must be an object at index {index}: {manifest_path}")
        if "lr" not in entry:
            raise KeyError(f"Manifest entry is missing `lr` at index {index}: {manifest_path}")

        lr_path = _resolve_existing_input_path(str(entry["lr"]), f"manifest lr[{index}]", manifest_path.parent)
        hr_path = entry.get("hr")
        resolved_hr = None
        if hr_path is not None and str(hr_path).strip():
            resolved_hr = str(
                _resolve_existing_input_path(str(hr_path), f"manifest hr[{index}]", manifest_path.parent)
            )

        samples.append(
            SampleRef(
                lr_path=lr_path,
                hr_path=resolved_hr,
                staged_name=lr_path.name,
                output_name=f"{lr_path.stem}.png",
            )
        )

    return _build_stage_bundle(samples, name_prefix)


def _build_hat_command(bundle: StageBundle, runtime: HatRuntime, run_name: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "models.team01_CIPLAB.hat.test",
        "-opt",
        str(runtime.opt_path),
        "--force_yml",
        f"name={run_name}",
        f"datasets:test_1:name={bundle.dataset_name}",
        f"datasets:test_1:dataroot_lq={bundle.lr_dir}",
        f"path:pretrain_network_g={runtime.weight_path}",
        "val:suffix=x4",
    ]


def _locate_visualization_dir(bundle: StageBundle, run_name: str) -> Path:
    preferred = RESULTS_DIR / run_name / "visualization" / bundle.dataset_name
    if preferred.is_dir():
        return preferred.resolve()

    visualization_root = RESULTS_DIR / run_name / "visualization"
    if visualization_root.is_dir():
        for candidate in sorted(path for path in visualization_root.iterdir() if path.is_dir()):
            missing = [
                sample.staged_name
                for sample in bundle.samples
                if not (candidate / f"{Path(sample.staged_name).stem}_x4.png").is_file()
            ]
            if not missing:
                return candidate.resolve()

    raise FileNotFoundError(
        f"Could not find HAT visualization outputs for run `{run_name}` under {visualization_root}"
    )


def _copy_outputs(bundle: StageBundle, run_name: str, output_dir: Path) -> None:
    raw_dir = _locate_visualization_dir(bundle, run_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    for sample in bundle.samples:
        raw_file = raw_dir / f"{Path(sample.staged_name).stem}_x4.png"
        if not raw_file.is_file():
            raise FileNotFoundError(f"Expected HAT output not found: {raw_file}")
        shutil.copy2(raw_file, output_dir / sample.output_name)


def write_result_json(bundle: StageBundle, output_dir: Path) -> Path:
    items = []
    for sample in bundle.samples:
        entry = {"res": str((output_dir / sample.output_name).resolve())}
        if sample.hr_path:
            entry["hr"] = sample.hr_path
        items.append(entry)

    items.sort(key=lambda item: Path(item["res"]).name)
    payload = {"items": items}
    result_path = (output_dir / "result.json").resolve()
    result_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result_path


def _print_summary(bundle: StageBundle, output_dir: Path, runtime: HatRuntime, run_name: str) -> None:
    print("[team01_CIPLAB] HAT-L inference", flush=True)
    print(f"  run_name             : {run_name}", flush=True)
    print(f"  input_dir            : {bundle.lr_dir}", flush=True)
    print(f"  output_dir           : {output_dir}", flush=True)
    print(f"  num_samples          : {len(bundle.samples)}", flush=True)
    print(f"  opt                  : {runtime.opt_path}", flush=True)
    print(f"  weight               : {runtime.weight_path}", flush=True)
    print(f"  dataset_name         : {bundle.dataset_name}", flush=True)
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    print(
        f"  CUDA_VISIBLE_DEVICES : {cuda_visible_devices if cuda_visible_devices else '<inherit>'}",
        flush=True,
    )


def _run_hat(bundle: StageBundle, output_dir: Path, runtime: HatRuntime, run_name: str) -> None:
    _print_summary(bundle, output_dir, runtime, run_name)
    cmd = _build_hat_command(bundle, runtime, run_name)
    print(f"  command              : {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
    _copy_outputs(bundle, run_name, output_dir)


def run_from_input_dir(input_path: str, output_path: str, run_name: str = DEFAULT_TEST_NAME) -> None:
    runtime = resolve_runtime()
    output_dir = Path(output_path).expanduser().resolve()
    bundle = build_stage_bundle_from_input_dir(input_path, name_prefix=run_name)
    try:
        _run_hat(bundle, output_dir, runtime, run_name)
    finally:
        shutil.rmtree(bundle.stage_dir, ignore_errors=True)


def run_experiment(exp_path: str, dry_run: bool = False) -> None:
    exp_file = Path(exp_path).expanduser().resolve()
    experiment = _load_json(exp_file)
    if not isinstance(experiment, dict):
        raise ValueError(f"Experiment JSON must be an object: {exp_file}")

    data_input = experiment.get("data_input")
    output_path = experiment.get("output_path")
    settings = experiment.get("setting")

    if not isinstance(data_input, str) or not data_input.strip():
        raise ValueError(f"`data_input` must be a non-empty string: {exp_file}")
    if not isinstance(output_path, str) or not output_path.strip():
        raise ValueError(f"`output_path` must be a non-empty string: {exp_file}")
    if not isinstance(settings, list) or not settings:
        raise ValueError(f"`setting` must be a non-empty list: {exp_file}")

    if experiment.get("eval_env") or experiment.get("viz_env"):
        print("[team01_CIPLAB] `eval_env` and `viz_env` are ignored in the minimal HAT-only runner.", flush=True)

    manifest_path = _resolve_existing_input_path(data_input, "experiment.data_input", exp_file.parent)
    output_root = _resolve_output_path(output_path, exp_file.parent)
    runtime = resolve_runtime()
    manifest = _load_json(manifest_path)
    if not isinstance(manifest, list) or not manifest:
        raise ValueError(f"Manifest must be a non-empty list: {manifest_path}")

    seen_test_names: set[str] = set()

    for index, setting in enumerate(settings):
        if not isinstance(setting, dict):
            raise ValueError(f"experiment.setting[{index}] must be an object.")

        test_name = setting.get("test_name")
        model_key = setting.get("model")
        weight_key = setting.get("weight")
        if not isinstance(test_name, str) or not test_name.strip():
            raise ValueError(f"experiment.setting[{index}].test_name must be a non-empty string.")
        if test_name in seen_test_names:
            raise ValueError(f"Duplicate test_name: {test_name}")
        seen_test_names.add(test_name)

        if model_key != DEFAULT_MODEL_KEY:
            raise ValueError(
                f"Unsupported model: {model_key!r}. Only `{DEFAULT_MODEL_KEY}` is implemented in team01_CIPLAB."
            )
        if weight_key != DEFAULT_WEIGHT_KEY:
            raise ValueError(
                f"Unsupported weight: {weight_key!r}. "
                f"Only `{DEFAULT_WEIGHT_KEY}` is implemented in team01_CIPLAB."
            )

        output_dir = (output_root / test_name).resolve()

        print(f"[team01_CIPLAB] experiment setting: {test_name}", flush=True)
        print(f"  manifest             : {manifest_path}", flush=True)
        print(f"  output_dir           : {output_dir}", flush=True)
        print(f"  num_samples          : {len(manifest)}", flush=True)

        if dry_run:
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        bundle = build_stage_bundle_from_manifest_json(manifest_path, name_prefix=test_name)
        try:
            _run_hat(bundle, output_dir, runtime, test_name)
            result_json = write_result_json(bundle, output_dir)
            print(f"  result_json          : {result_json}", flush=True)
        finally:
            shutil.rmtree(bundle.stage_dir, ignore_errors=True)
