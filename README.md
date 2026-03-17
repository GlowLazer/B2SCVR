# Enhanced B2SCVR — NTIRE 2026 BSCVR Challenge (Team Vroom)

This repository contains Team Vroom's submission for the
NTIRE 2026 Challenge on Bitstream-corrupted Video Restoration (BSCVR).

Built upon the [B2SCVR baseline](https://github.com/LIUTIGHE/B2SCVR) with the
following enhancements (in development order):
1. Reverse TTA — bidirectional temporal ensemble
2. Temporal Difference Loss — enforces frame-to-frame consistency during training
3. Boundary Refinement Head — corrects mask-boundary artefacts via soft alpha blending
4. Residual Refiner — lightweight U-Net (~768K params) for per-frame residual correction
5. LoRA fine-tuning — low-rank adaptation of SAM2 Hiera trunk and Transformer attention blocks
6. Loss function modification — fine-tuned from baseline
7. Weighted loss combination — `λ₁·L_hole + λ₂·L_valid + λ₃·L_l1_stab` tuning → final submitted model

---

## Docker (recommended)

The repo ships a fully self-contained GPU-enabled Docker image based on
**Ubuntu 24.04**, **Miniforge (conda)**, and **PyTorch 2.5.1 + CUDA 12.1**.
All known compatibility patches (basicsr, MKL) are applied at build time.

### Host requirements

- NVIDIA GPU with drivers installed (`nvidia-smi` works)
- Docker Engine installed
- NVIDIA Container Toolkit installed and configured

If you are missing Docker or the NVIDIA Container Toolkit, run the included helper:

```bash
sudo bash docker/setup_ubuntu24_gpu.sh
```

Verify GPU passthrough works before building:

```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Build

```bash
docker build -t b2scvr:cu121 .
```

### Run — inference only (no GT needed)

```bash
docker run --rm -it --gpus all \
  -v /path/to/bsc_imgs:/data/bsc:ro \
  -v /path/to/masks:/data/masks:ro \
  -v "$(pwd)/checkpoints:/workspace/checkpoints:ro" \
  -v "$(pwd)/outputs:/workspace/outputs" \
  b2scvr:cu121 \
  bash vali.sh --bsc_dir /data/bsc --mask_dir /data/masks
```

### Run — full pipeline (inference + post-processing)

```bash
docker run --rm -it --gpus all \
  -v /path/to/bsc_imgs:/data/bsc:ro \
  -v /path/to/masks:/data/masks:ro \
  -v "$(pwd)/checkpoints:/workspace/checkpoints:ro" \
  -v "$(pwd)/outputs:/workspace/outputs" \
  -v "$(pwd)/Vroom:/workspace/Vroom" \
  b2scvr:cu121 \
  bash run_all.sh --bsc_dir /data/bsc --mask_dir /data/masks
```

> **Note:** `./outputs` and `./Vroom` are bind-mounted — results appear on the host
> in real time. These directories are created automatically by Docker if they
> don't exist. Do **not** delete them while the container is running; clear their
> contents instead (`rm -rf ./Vroom/* ./outputs/*`).

### Copying outputs out of a running / stopped container

If you ran without bind mounts (or the mount broke), use `docker cp`:

```bash
# 1. Find the container ID (works for running or stopped containers)
docker ps -a

# 2. Copy Vroom and outputs to the host
docker cp <container_id>:/workspace/Vroom   ./Vroom
docker cp <container_id>:/workspace/outputs ./outputs
```

If the container is still running, get its ID in one step:

```bash
docker cp $(docker ps -q --filter ancestor=b2scvr:cu121):/workspace/Vroom ./Vroom
```

---

## Manual Installation (without Docker)

**Requirements:** Python 3.10, CUDA 12.1, Linux (or WSL on Windows).

```bash
git clone https://github.com/GlowLazer/B2SCVR
cd B2SCVR

# 1. Create conda environment
conda create -n b2scvr python=3.10
conda activate b2scvr

# 2. Install PyTorch (CUDA 12.1)
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia

# 3. Install mmcv
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html

# 4. Install SAM2 (bundled)
cd model/modules/sam2
pip install -e .
cd ../../..

# 5. Install remaining dependencies
pip install -r requirements.txt

# 6. Patch basicsr (functional_tensor removed in torchvision >= 0.16)
sed -i \
  's|from torchvision.transforms.functional_tensor import rgb_to_grayscale|from torchvision.transforms.functional import rgb_to_grayscale|g' \
  $(python -c "import basicsr; import os; print(os.path.join(os.path.dirname(basicsr.__file__), 'data', 'degradations.py'))")
```

---

## Checkpoints

Place checkpoints in the `checkpoints/` directory before running inference.

`run_all.sh` / `vali.sh` / `test.py` expect the following layout:

| File | Description |
|------|-------------|
| `checkpoints/ckpt_best/gen_best.pth` | Main inpainting model |
| `checkpoints/boundary_head_best.pth` | Boundary Refinement Head |
| `checkpoints/refiner_best.pth` | Residual Refiner |
| `checkpoints/lora_sam_best.pth` | SAM2 LoRA + SAMFuser weights |
| `checkpoints/lora_transformer_on_sam_best.pth` | Transformer LoRA weights |
| `checkpoints/B2SCVR_ckpts/checkpoint.pt` | SAM2.1 Hiera-T backbone |

---

## Quickstart (main pipeline)

The main pipeline is `run_all.sh` (inference → post-processing):

```bash
bash run_all.sh \
  --bsc_dir  /path/to/bsc_imgs \
  --mask_dir /path/to/masks
```

What it does:
1. Runs inference (`vali.sh` → `test.py`) and writes raw outputs to `./outputs/<video>/frame_seq/*.png`
2. Upsamples + hard-mask composites and writes submission-ready frames to `./Vroom/<video>/*.png`

### Expected data layout

- `--bsc_dir`: `/path/to/bsc_imgs/<video_name>/*.jpg`
- `--mask_dir`: `/path/to/masks/<video_name>/*.png`

---

## Inference only (no GT needed)

```bash
bash vali.sh \
  --bsc_dir  /path/to/bsc_imgs \
  --mask_dir /path/to/masks
```

Output frames are saved to `outputs/<video_name>/frame_seq/`.

Optional flags:
- `--out_dir ./outputs` — output directory (default: `./outputs`)
- `--width 432 --height 240` — inference resolution (default)

---

## Package Submission (CodaBench)

After inference, run:

```bash
python make_submission.py \
  --bsc_dir  /path/to/bsc_imgs \
  --mask_dir /path/to/masks \
  --input_dir outputs \
  --zip_name Vroom.zip
```

This upsamples outputs to native resolution, applies a hard-mask composite
(+~0.17 dB PSNR), and packages everything as `Vroom.zip` for CodaBench submission.

---

## Acknowledgements

Built upon
[B2SCVR](https://github.com/LIUTIGHE/B2SCVR),
[SAM2](https://github.com/facebookresearch/sam2), and
[ATD](https://github.com/LabShuHangGU/Adaptive-Token-Dictionary).
