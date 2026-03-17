FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget ca-certificates tmux \
    build-essential ninja-build pkg-config \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Miniconda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
 && bash /tmp/miniconda.sh -b -p /opt/conda \
 && rm /tmp/miniconda.sh

ENV PATH=/opt/conda/bin:$PATH
SHELL ["/bin/bash", "-lc"]

# Use conda-forge only.
RUN mkdir -p /etc/conda && \
    printf "channels:\n  - conda-forge\nchannel_priority: strict\ndefault_channels: []\n" > /etc/conda/.condarc

# =========================
# Env: hat_env 
# =========================
RUN conda create -n hat_env -y --override-channels -c conda-forge python=3.10

RUN conda run -n hat_env python -m pip install -U \
    "pip<25" \
    "setuptools<82" \
    wheel

RUN conda run -n hat_env pip install --index-url https://download.pytorch.org/whl/cu117 \
    torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

RUN conda run -n hat_env pip install \
    "numpy==1.26.4" \
    opencv-python scikit-image scipy \
    pyyaml tqdm requests einops lmdb h5py addict future yapf packaging

RUN conda run -n hat_env pip install \
    timm==0.9.16 transformers==4.38.1 huggingface-hub tensorboard

RUN conda run -n hat_env pip install --no-build-isolation \
    mamba-ssm==1.0.1 causal-conv1d==1.0.0

RUN conda run -n hat_env pip install --no-build-isolation \
    basicsr==1.3.4.9
    
# =========================
# Env: flux2
# =========================
RUN conda create -n flux2 -y --override-channels -c conda-forge python=3.10

RUN conda run -n flux2 python -m pip install -U \
    "pip<25" \
    "setuptools<82" \
    wheel

RUN conda run -n flux2 pip install --index-url https://download.pytorch.org/whl/cu117 \
    torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

RUN conda run -n flux2 pip install \
    "numpy<2" \
    scipy \
    timm==0.9.16 \
    einops \
    pillow tqdm packaging pyyaml requests \
    "accelerate>=0.31.0" \
    "transformers>=4.41.2" \
    "peft>=0.11.1" \
    datasets \
    sentencepiece \
    ftfy \
    tensorboard \
    Jinja2 \
    bitsandbytes \
    prodigyopt \
    huggingface-hub \
    wandb \
    safetensors

RUN conda run -n flux2 pip install \
    "diffusers @ git+https://github.com/huggingface/diffusers.git"

# =========================
# Convenience wrappers
# =========================
RUN printf '%s\n' \
'#!/usr/bin/env bash' \
'set -e' \
'source /opt/conda/etc/profile.d/conda.sh' \
'conda run -n hat_env --no-capture-output "$@"' \
> /usr/local/bin/hat_env && chmod +x /usr/local/bin/hat_env

RUN printf '%s\n' \
'#!/usr/bin/env bash' \
'set -e' \
'source /opt/conda/etc/profile.d/conda.sh' \
'conda run -n flux2 --no-capture-output "$@"' \
> /usr/local/bin/flux2 && chmod +x /usr/local/bin/flux2

CMD ["bash"]
