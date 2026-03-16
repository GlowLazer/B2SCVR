# Enhanced B2SCVR — NTIRE 2026 BSCVR Challenge (Team Vroom)

This repository contains Team Vroom's submission for the
NTIRE 2026 Challenge on Bitstream-corrupted Video Restoration (BSCVR).

Built upon the [B2SCVR baseline](https://github.com/LIUTIGHE/B2SCVR) with the
following enhancements:
- SAM2 Hiera-T feature extractor with LoRA fine-tuning
- Boundary Refinement Head for mask-boundary artefact correction
- Residual Refiner (lightweight U-Net, ~768K params) for per-frame correction
- Reverse TTA (bidirectional temporal ensemble with learned Spatial Alpha Net)
- MSE-only fine-tuning with tuned hole/valid loss weights

---

## Installation

**Requirements:** Python 3.10, CUDA 12.1

```bash
git clone <this-repo>
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
```

**Known issues:**
- If `ModuleNotFoundError: torchvision.transforms.functional_tensor` occurs, edit
  the reported `degradations.py` line:
  change `from torchvision.transforms.functional_tensor import rgb_to_grayscale`
  to `from torchvision.transforms.functional import rgb_to_grayscale`
- If Intel MKL errors occur, reinstall torch via pip:
  `pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121`
- For mmcv-related errors, replace `mmcv.cnn` → `mmengine.model` and
  `mmcv.runner` → `mmengine.runner` in the reported file.

---

## Checkpoints

Place the following checkpoints in the `checkpoints/` directory:

| File | Description |
|------|-------------|
| `checkpoints/psnr_run5/gen_best.pth` | Main inpainting model (submitted) |
| `checkpoints/boundary_head_best.pth` | Boundary Refinement Head |
| `checkpoints/refiner_1500_best.pth` | Residual Refiner |
| `checkpoints/lora_sam2_run3_best.pth` | SAM2 LoRA + SAMFuser weights |
| `checkpoints/lora_transformer_on_sam2_best.pth` | Transformer LoRA weights |
| `checkpoints/B2SCVR_ckpts/checkpoint.pt` | SAM2.1 Hiera-T backbone |

---

## Inference

```bash
bash vali.sh \
  --video_dir /path/to/bsc_imgs \
  --mask_dir  /path/to/masks
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

## Training

To reproduce the submitted checkpoint (`psnr_run5`):

```bash
# 1. Train base model
python train.py --c config/ema_cons_run1.json

# 2. Fine-tune with PSNR-optimized loss weights
python train.py --c config/config_psnr.json

# 3. Train LoRA adapters
python train_lora_sam2.py --c config/lora_sam2.json
python train_lora_transformer.py

# 4. Train Residual Refiner
bash gen_train_preds_1500.sh   # generate training predictions
python train_refiner.py

# 5. Train Boundary Head
python train_boundary.py
```

---

## Results (Validation — worst10 set, 432x240)

| Configuration | PSNR (dB) | SSIM |
|---|---|---|
| Baseline (`main_best.pth`) | 18.24 | — |
| + MSE weight tuning (`psnr_run5`) | 20.23 | 0.747 |
| + Residual Refiner (full pipeline) | 22.73 | 0.791 |

---

## Acknowledgements

Built upon [B2SCVR](https://github.com/LIUTIGHE/B2SCVR),
[SAM2](https://github.com/facebookresearch/sam2), and
[ATD](https://github.com/LabShuHangGU/Adaptive-Token-Dictionary).
