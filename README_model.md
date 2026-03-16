## team01_CIPLAB

`models/team01_CIPLAB` now contains a minimal, HAT-only inference path that is independent from `code/sr/stage1`.

What is included:

- local HAT registration code under `models/team01_CIPLAB/hat`
- a local HAT-L option file under `models/team01_CIPLAB/options`
- `io.py` for `input_dir -> output_dir`
- `run_experiment.py` for the minimal experiment JSON format

Only this model is supported:

- `model = hat`
- `weight = hat_l_srx4_imagenet_pretrain`

## io.py

`io.py` is the simple inference entrypoint.

```bash
cd /media/ssd1/users/jaeho/code/NTIRE2026_ImageSR_x4_ciplab
python -m models.team01_CIPLAB.io /path/to/input_or_dataset_root /path/to/output
```

It accepts:

- an image directory directly
- or a dataset root containing `LQ/`

The output images are written as PNG files and can be used as the next input to `lora_inference`.

## run_experiment.py

The following JSON format is supported:

```json
{
  "data_input": "data/train_DIV2K.json",
  "output_path": "data/DIV2K_trainset_w_hat_l",
  "eval_env": "eval",
  "viz_env": "eval",
  "setting": [
    {
      "test_name": "hat_l_srx4_imagenet_pretrain",
      "model": "hat",
      "weight": "hat_l_srx4_imagenet_pretrain",
      "env": "sr"
    }
  ]
}
```

Run it with:

```bash
python -m models.team01_CIPLAB.run_experiment --exp /path/to/experiment.json
```

This minimal runner ignores `eval_env`, `viz_env`, and `env`.
It only creates per-test output images and `result.json`.

## Weight path

The code first looks for:

- `CIPLAB_HAT_WEIGHT_PATH`
- `CIPLAB_HAT_WEIGHT`

If those are not set, the default path is:

- `model_zoo/team01_CIPLAB/HAT-L_SRx4_ImageNet-pretrain.pth`

The expected filename is:

- `HAT-L_SRx4_ImageNet-pretrain.pth`
