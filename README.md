## team01_CIPLAB


## Stert

```bash
git clone https://github.com/qnrwjd78/NTIRE2026_ImageSR_x4_ciplab.git
cd NTIRE2026_ImageSR_x4_ciplab
```

## Environment

Build from the repository root:

```bash
docker build -t team01_ciplab -f models/team01_CIPLAB/docker/Dockerfile .
```

Run an interactive container:

```bash
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  team01_ciplab
```

The container uses the `sr_env` conda environment by default.

## Weight Folder Structure

Expected layout under `model_zoo`:

```text
model_zoo/
└── team01_CIPLAB/
    ├── HAT-L_SRx4_ImageNet-pretrain.pth
    ├── flux2-klein-base-9b/
    │   ├── model_index.json
    │   ├── transformer/
    │   └── vae/
    └── step2_weight/
        ├── stage1/
        │   └── checkpoint-*/
        ├── stage2/
        │   └── checkpoint-*/
        └── stage3/
            └── checkpoint-*/
```

Notes:

- `HAT-L_SRx4_ImageNet-pretrain.pth` is the Step 1 HAT weight.
- `flux2-klein-base-9b/` is the Step 2 base diffusers checkpoint.
- `step2_weight/stage1`, `step2_weight/stage2`, and optional `step2_weight/stage3` should contain LoRA checkpoints such as `pytorch_lora_weights.safetensors` and `adapter_config.json`.

## Weight Download

The URLs are listed in [team01_CIPLAB.txt](NTIRE2026_ImageSR_x4_ciplab/model_zoo/team01_CIPLAB/team01_CIPLAB.txt).
Download the first line as the FLUX.2 base model
Download the second line as the HAT weight

Sources:

- FLUX.2:
  [black-forest-labs/FLUX.2-klein-base-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B)
- HAT-L_SRx4_ImageNet-pretrain.pth:
  [jaideepsingh/upscale_models HAT mirror](https://huggingface.co/jaideepsingh/upscale_models/resolve/main/HAT/HAT-L_SRx4_ImageNet-pretrain.pth)

## test.py

`test.py` can also launch `team01_CIPLAB`.

Use `model_id = 1`:

```bash
cd /media/ssd1/users/jaeho/code/NTIRE2026_ImageSR_x4_ciplab
python test.py --test_dir /path/to/LR --save_dir /path/to/save --model_id 1
```

This path calls `models.team01_CIPLAB.main`, which forwards to the same `io.py` pipeline.
