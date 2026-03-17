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

BSC_DIR=""
MASK_DIR=""
OUT_DIR="outputs"
WIDTH=432
HEIGHT=240

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bsc_dir) BSC_DIR="$2"; shift 2 ;;
        --mask_dir)  MASK_DIR="$2";  shift 2 ;;
        --out_dir)   OUT_DIR="$2";   shift 2 ;;
        --width)     WIDTH="$2";     shift 2 ;;
        --height)    HEIGHT="$2";    shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$BSC_DIR" || -z "$MASK_DIR" ]]; then
    echo "ERROR: --bsc_dir and --mask_dir are required."
    echo "Usage: bash vali.sh --bsc_dir /path/to/bsc_imgs --mask_dir /path/to/masks"
    exit 1
fi

python test.py \
  --ckpt checkpoints/ckpt_best/gen_best.pth \
  --boundary_ckpt checkpoints/boundary_head_best.pth \
  --refiner_ckpt  checkpoints/refiner_best.pth \
  --lora_ckpt     checkpoints/lora_sam_best.pth \
  --lora_transformer_ckpt checkpoints/lora_transformer_on_sam_best.pth \
  --video_dir "$BSC_DIR" \
  --mask_dir  "$MASK_DIR" \
  --width "$WIDTH" --height "$HEIGHT" \
  --out_dir "$OUT_DIR"
