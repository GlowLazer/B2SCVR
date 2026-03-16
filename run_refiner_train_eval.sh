#!/bin/bash
set -euo pipefail

# Train/resume the refiner, then run the validation pipeline + submission packaging + PSNR
# using the newly produced checkpoint.
#
# Usage examples:
#   bash run_refiner_train_eval.sh
#   RESUME=checkpoints/refiner_1500_iter16000.pth bash run_refiner_train_eval.sh
#   CKPT_TO_EVAL=checkpoints/refiner_1500_iter20000.pth bash run_refiner_train_eval.sh
#
# Key env vars:
#   SAVE             Path for refiner training save base (default: checkpoints/refiner_1500.pth)
#   RESUME           Optional resume checkpoint path
#   ITERS            Total iterations (default: 30000)
#   PRED_ROOT        Train pred_root (default: /media/ajeet/data/MINI_BTP/B2SCVR/output_train)
#   SAVE_FREQ        Iter checkpoint frequency (default: 1000)
#   VAL_PIPELINE_EVAL=1 to run validation PSNR every 2k iters during training (slow)
#   CKPT_TO_EVAL     If set, evaluate that checkpoint; otherwise prefer *_bestval.pth, else latest iter ckpt

cd /media/ajeet/data/MINI_BTP/B2SCVR

# Avoid OpenMP/MKL runtime conflicts in this environment (cv2/skimage/torch mix).
export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"
export KMP_DISABLE_SHM="${KMP_DISABLE_SHM:-1}"
export KMP_USE_SHM="${KMP_USE_SHM:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SAVE="${SAVE:-checkpoints/refiner_1500.pth}"
RESUME="${RESUME:-}"
ITERS="${ITERS:-30000}"
SAVE_FREQ="${SAVE_FREQ:-1000}"
PRED_ROOT="${PRED_ROOT:-/media/ajeet/data/MINI_BTP/B2SCVR/output_train}"

TRAIN_ARGS=(
  --bsc_root  /home/ajeet/MINIP/data/train/train_bsc_imgs
  --gt_root   /home/ajeet/MINIP/data/train/train_gt_imgs
  --mask_root /home/ajeet/MINIP/data/train/train_masks
  --pred_root "$PRED_ROOT"
  --save "$SAVE"
  --iters "$ITERS"
  --save_freq "$SAVE_FREQ"
)

if [ -n "$RESUME" ]; then
  TRAIN_ARGS+=( --resume "$RESUME" )
fi

if [ "${VAL_PIPELINE_EVAL:-0}" = "1" ]; then
  TRAIN_ARGS+=( --val_pipeline_eval --val_eval_freq 2000 )
fi

echo "=== Step 1: Train/Resume Refiner ==="
python train_refiner.py "${TRAIN_ARGS[@]}"

echo ""
echo "=== Step 2: Choose Checkpoint To Evaluate ==="
CKPT_TO_EVAL="${CKPT_TO_EVAL:-}"
SAVE_STEM="${SAVE%.pth}"
BESTVAL="${SAVE_STEM}_bestval.pth"

if [ -z "$CKPT_TO_EVAL" ] && [ -f "$BESTVAL" ]; then
  CKPT_TO_EVAL="$BESTVAL"
fi
if [ -z "$CKPT_TO_EVAL" ]; then
  # Fall back to latest iter checkpoint for this save stem.
  CKPT_TO_EVAL="$(ls -1 "${SAVE_STEM}_iter"*.pth 2>/dev/null | sort | tail -n 1 || true)"
fi
if [ -z "$CKPT_TO_EVAL" ] || [ ! -f "$CKPT_TO_EVAL" ]; then
  echo "ERROR: Could not find a checkpoint to evaluate."
  echo "Tried bestval: $BESTVAL"
  echo "Tried latest iter: ${SAVE_STEM}_iter*.pth"
  exit 1
fi
echo "Evaluating: $CKPT_TO_EVAL"

echo ""
echo "=== Step 3: Swap Into checkpoints/refiner_best.pth ==="
mkdir -p checkpoints
if [ -f checkpoints/refiner_best.pth ]; then
  ts="$(date +%Y%m%d_%H%M%S)"
  cp checkpoints/refiner_best.pth "checkpoints/refiner_best.pth.bak_${ts}"
fi
cp "$CKPT_TO_EVAL" checkpoints/refiner_best.pth

echo ""
echo "=== Step 4: Run Full Pipeline (vali.sh -> make_submission.py -> psnr.py) ==="
bash run_all.sh

