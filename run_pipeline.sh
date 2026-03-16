#!/bin/bash
# Full pipeline: test → diffuse → make_submission → psnr
# Usage: bash run_pipeline.sh
set -e   # stop on first error
cd /media/ajeet/data/MINI_BTP/B2SCVR

CKPT=checkpoints/main_best.pth
BOUNDARY_CKPT=checkpoints/boundary_head_best.pth
REFINER_CKPT=checkpoints/refiner_best.pth
LORA_CKPT=checkpoints/lora_sam2_best.pth

VIDEO_DIR=/media/ajeet/data/MINI_BTP/data/validation/bsc_imgs
MASK_DIR=/media/ajeet/data/MINI_BTP/data/validation/masks
W=432
H=240

DIFFUSION_STRENGTH=0.2
DIFFUSION_STEPS=20
DIFFUSION_THRESH=0.15

# ── Step 1: Main pipeline ────────────────────────────────────────────────────
echo "================================================================"
echo " STEP 1: test.py  →  ./outputs"
echo "================================================================"
python test.py \
    --ckpt            "$CKPT" \
    --boundary_ckpt   "$BOUNDARY_CKPT" \
    --refiner_ckpt    "$REFINER_CKPT" \
    --lora_ckpt       "$LORA_CKPT" \
    --video_dir       "$VIDEO_DIR" \
    --mask_dir        "$MASK_DIR" \
    --width  $W \
    --height $H

# ── Step 2: Diffusion post-refinement ───────────────────────────────────────
echo ""
echo "================================================================"
echo " STEP 2: diffuse_outputs.py  →  ./outputs_diffused"
echo "================================================================"
python diffuse_outputs.py \
    --input_dir  ./outputs \
    --mask_dir   "$MASK_DIR" \
    --out_dir    ./outputs_diffused \
    --strength   $DIFFUSION_STRENGTH \
    --steps      $DIFFUSION_STEPS \
    --thresh     $DIFFUSION_THRESH \
    --width  $W \
    --height $H

# ── Step 3: Package submission ───────────────────────────────────────────────
echo ""
echo "================================================================"
echo " STEP 3: make_submission.py  →  ./Vroom.zip"
echo "================================================================"
python make_submission.py \
    --input_dir ./outputs_diffused \
    --zip_name  ./Vroom.zip

# ── Step 4: PSNR evaluation ──────────────────────────────────────────────────
echo ""
echo "================================================================"
echo " STEP 4: psnr.py  (reads ./Vroom)"
echo "================================================================"
python psnr.py

echo ""
echo "All done. Submission: ./Vroom.zip"
