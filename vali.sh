#!/usr/bin/env bash
# Run full inference pipeline on NTIRE 2026 BSCVR test set.
#
# Usage:
#   bash vali.sh --video_dir /path/to/bsc_imgs --mask_dir /path/to/masks
#
# Optional:
#   --out_dir  ./outputs   (default)
#   --width    432         (default)
#   --height   240         (default)

VIDEO_DIR=""
MASK_DIR=""
OUT_DIR="outputs"
WIDTH=432
HEIGHT=240

while [[ $# -gt 0 ]]; do
    case "$1" in
        --video_dir) VIDEO_DIR="$2"; shift 2 ;;
        --mask_dir)  MASK_DIR="$2";  shift 2 ;;
        --out_dir)   OUT_DIR="$2";   shift 2 ;;
        --width)     WIDTH="$2";     shift 2 ;;
        --height)    HEIGHT="$2";    shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$VIDEO_DIR" || -z "$MASK_DIR" ]]; then
    echo "ERROR: --video_dir and --mask_dir are required."
    echo "Usage: bash vali.sh --video_dir /path/to/bsc_imgs --mask_dir /path/to/masks"
    exit 1
fi

python test.py \
  --ckpt checkpoints/psnr_run5/gen_best.pth \
  --boundary_ckpt checkpoints/boundary_head_best.pth \
  --refiner_ckpt  checkpoints/refiner_1500_best.pth \
  --lora_ckpt     checkpoints/lora_sam2_run3_best.pth \
  --lora_transformer_ckpt checkpoints/lora_transformer_on_sam2_best.pth \
  --video_dir "$VIDEO_DIR" \
  --mask_dir  "$MASK_DIR" \
  --width "$WIDTH" --height "$HEIGHT" \
  --out_dir "$OUT_DIR"
