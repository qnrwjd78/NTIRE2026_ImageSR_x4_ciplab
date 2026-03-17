# team01_CIPLAB


## 1. Start

```bash
git clone https://github.com/qnrwjd78/NTIRE2026_ImageSR_x4_ciplab.git
cd NTIRE2026_ImageSR_x4_ciplab
```

## 2. Environment

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

## 3. Weight Download

```bash
hf download Jasonlee1995/ntire2026 --local-dir ./model_zoo/team01_CIPLAB/
```

The URL is also listed in `model_zoo/team01_CIPLAB/team01_CIPLAB.txt`.

Sources:
- FLUX.2:
  [black-forest-labs/FLUX.2-klein-base-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B)
- HAT-L_SRx4_ImageNet-pretrain.pth:
  [jaideepsingh/upscale_models HAT mirror](https://huggingface.co/jaideepsingh/upscale_models/resolve/main/HAT/HAT-L_SRx4_ImageNet-pretrain.pth)


Expected layout under `model_zoo/team01_CIPLAB/`:

```text
model_zoo/
└── team01_CIPLAB/
    ├── HAT-L_SRx4_ImageNet-pretrain.pth
    ├── flux2-klein-base-9b/
    │   ├── scheduler
    │   ├── text_encoder
    │   ├── ...
    |   └── model_index.json
    └── step2/
        ├── stage1/
        │   └── checkpoint-*/
        └── stage2/
            └── checkpoint-*/
```




## 4. test.py

`test.py` can also launch `team01_CIPLAB`.

Use `model_id = 1`.

> [!IMPORTANT]
> The Docker container mounts the repository root to `/workspace`:
>
> ```bash
> docker run --gpus all -it --rm \
>   -v $(pwd):/workspace \
>   team01_ciplab
> ```
>
> Therefore, when running `test.py`, both `--test_dir` and `--save_dir`
> must refer to paths visible **inside the container**, usually under `/workspace`.
>
> Do **not** use host-only absolute paths unless they are separately mounted into the container.

Example:

```bash
python test.py \
  --test_dir /workspace/path/to/LR \
  --save_dir /workspace/path/to/save \
  --model_id 1
