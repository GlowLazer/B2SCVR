#!/bin/bash
# Usage: bash run_all.sh --bsc_dir /path/to/bsc_imgs --mask_dir /path/to/masks --gt_dir /path/to/gt_imgs
set -e

BSC_DIR=""
MASK_DIR=""
GT_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bsc_dir) BSC_DIR="$2"; shift 2 ;;
        --mask_dir)    MASK_DIR="$2";    shift 2 ;;
        --gt_dir)   GT_DIR="$2";   shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$BSC_DIR" || -z "$MASK_DIR" || -z "$GT_DIR" ]]; then
    echo "ERROR: --bsc_dir, --mask_dir, and --gt_dir are required."
    echo "Usage: bash run_all.sh --bsc_dir /path/to/bsc_imgs --mask_dir /path/to/masks --gt_dir /path/to/gt_imgs"
    exit 1
fi

echo "========================================="
echo " Step 1/3: Running inference (vali.sh)"
echo "========================================="
bash vali.sh --bsc_dir "$BSC_DIR" --mask_dir "$MASK_DIR"

echo ""
echo "========================================="
echo " Step 2/3: Post-processing outputs (post_processing.py)"
echo "========================================="
python post_processing.py --bsc_dir "$BSC_DIR" --mask_dir "$MASK_DIR"

echo ""
echo "========================================="
echo " Step 3/3: Evaluating PSNR / SSIM"
echo "========================================="
python psnr.py --mask_dir "$MASK_DIR" --gt_dir "$GT_DIR"

echo ""
echo "========================================="
echo " Done."
echo "========================================="

