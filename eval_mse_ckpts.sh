#!/bin/bash
# Evaluate every 1000-iteration checkpoint from ema_cons_run1 (MSE loss run)
# Usage: bash eval_mse_ckpts.sh
# Results are logged to checkpoints/ema_cons_run1/eval_psnr.txt

cd /media/ajeet/data/MINI_BTP/B2SCVR

CKPT_DIR="checkpoints/ema_cons_run1"
LOG="$CKPT_DIR/eval_psnr.txt"
VIDEO_DIR="/media/ajeet/data/MINI_BTP/data/validation/bsc_imgs"
MASK_DIR="/media/ajeet/data/MINI_BTP/data/validation/masks"

echo "MSE loss run — per-1000-iter PSNR evaluation" > "$LOG"
echo "Started: $(date)" >> "$LOG"
echo "==========================================" >> "$LOG"

for it in $(seq 1000 1000 10000); do
    it_str=$(printf "%06d" $it)

    # Prefer EMA checkpoint (more stable), fall back to regular
    if [ -f "$CKPT_DIR/gen_${it_str}_ema.pth" ]; then
        CKPT="$CKPT_DIR/gen_${it_str}_ema.pth"
    elif [ -f "$CKPT_DIR/gen_${it_str}.pth" ]; then
        CKPT="$CKPT_DIR/gen_${it_str}.pth"
    else
        echo "Iter $it: checkpoint not found, skipping"
        continue
    fi

    OUT_DIR="outputs_mse_iter${it_str}"
    echo ""
    echo "=== Iter $it  ($CKPT) ===" | tee -a "$LOG"

    python test.py \
        --ckpt "$CKPT" \
        --boundary_ckpt checkpoints/boundary_head_best.pth \
        --refiner_ckpt  checkpoints/refiner_best.pth \
        --lora_ckpt     checkpoints/lora_sam2_best.pth \
        --lora_transformer_ckpt checkpoints/lora_transformer_on_sam2_best.pth \
        --video_dir "$VIDEO_DIR" \
        --mask_dir  "$MASK_DIR" \
        --width 432 --height 240

    # Move this run's outputs to a dedicated per-iter folder
    [ -d outputs ] && mv outputs "$OUT_DIR"

    # Compute PSNR / SSIM
    python psnr.py --input_dir "$OUT_DIR" 2>&1 | tee -a "$LOG"
    echo "------------------------------------------" >> "$LOG"
done

echo "" | tee -a "$LOG"
echo "All done. Results in $LOG" | tee -a "$LOG"
