FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    bzip2 \
    ca-certificates \
    ffmpeg \
    git \
    libgl1 \
    libglib2.0-0 \
    wget \
  && rm -rf /var/lib/apt/lists/*

# Install Miniforge (conda-forge default, no Anaconda TOS restrictions).
ARG CONDA_DIR=/opt/conda
RUN wget -qO /tmp/miniforge.sh \
      https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
  && bash /tmp/miniforge.sh -b -p "${CONDA_DIR}" \
  && rm -f /tmp/miniforge.sh

ENV PATH="${CONDA_DIR}/bin:${PATH}"

WORKDIR /workspace

# Create env + install exact versions from README.
RUN conda config --set channel_priority strict \
  && conda create -y -n b2scvr python=3.10 \
  && conda install -y -n b2scvr \
      pytorch==2.5.1 \
      torchvision==0.20.1 \
      torchaudio==2.5.1 \
      pytorch-cuda=12.1 \
      -c pytorch -c nvidia \
  && conda clean -afy

COPY . /workspace

RUN conda run -n b2scvr python -m pip install -U pip setuptools wheel \
  && conda run -n b2scvr pip install \
      mmcv==2.2.0 \
      -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html

RUN conda run -n b2scvr pip install -e model/modules/sam2 \
  && conda run -n b2scvr pip install -r requirements.txt

# Patch basicsr: functional_tensor was removed in torchvision >= 0.16
RUN sed -i \
    's|from torchvision.transforms.functional_tensor import rgb_to_grayscale|from torchvision.transforms.functional import rgb_to_grayscale|g' \
    /opt/conda/envs/b2scvr/lib/python3.10/site-packages/basicsr/data/degradations.py

COPY docker/entrypoint.sh /usr/local/bin/b2scvr-entrypoint
RUN chmod +x /usr/local/bin/b2scvr-entrypoint

ENTRYPOINT ["/usr/local/bin/b2scvr-entrypoint"]
CMD ["bash"]
