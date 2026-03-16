#!/usr/bin/env bash
# Train (optional) -> inference -> make_submission -> PSNR
#
# Usage (evaluate an existing checkpoint):
#   GEN_CKPT=checkpoints/ema_cons_run1/gen_016000_ema.pth bash pipeline_train_submit_psnr.sh
#
# Usage (train then evaluate best):
#   CONFIG=config/ema_cons_run1.json RUN_TRAIN=1 bash pipeline_train_submit_psnr.sh
#
# Common overrides:
#   OUT_DIR=outputs_iter16000 ZIP_NAME=Vroom_iter16000.zip bash pipeline_train_submit_psnr.sh
#   REFINER_CKPT=checkpoints/refiner_1500_bestval.pth bash pipeline_train_submit_psnr.sh
set -euo pipefail

cd /media/ajeet/data/MINI_BTP/B2SCVR

# Avoid OpenMP SHM issues in this environment.
export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"
export KMP_DISABLE_SHM="${KMP_DISABLE_SHM:-1}"
export KMP_USE_SHM="${KMP_USE_SHM:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

CONFIG="${CONFIG:-config/ema_cons_run1.json}"
RUN_TRAIN="${RUN_TRAIN:-0}"

BOUNDARY_CKPT="${BOUNDARY_CKPT:-checkpoints/boundary_head_best.pth}"
REFINER_CKPT="${REFINER_CKPT:-checkpoints/refiner_best.pth}"
LORA_CKPT="${LORA_CKPT:-checkpoints/lora_sam2_run3_best.pth}"
LORA_TRANSFORMER_CKPT="${LORA_TRANSFORMER_CKPT:-checkpoints/lora_transformer_on_sam2_best.pth}"

VIDEO_DIR="${VIDEO_DIR:-/media/ajeet/data/MINI_BTP/data/validation/bsc_imgs}"
MASK_DIR="${MASK_DIR:-/media/ajeet/data/MINI_BTP/data/validation/masks}"
W="${W:-432}"
H="${H:-240}"

OUT_DIR="${OUT_DIR:-outputs}"
ZIP_NAME="${ZIP_NAME:-Vroom.zip}"

if [[ "${RUN_TRAIN}" == "1" ]]; then
  echo "========================================="
  echo " Step 1/4: Training (train.py)"
  echo "========================================="
  python train.py --c "${CONFIG}"
fi

SAVE_DIR="$(python - <<'PY'
import json, os, sys
cfg = os.environ.get("CONFIG", "config/ema_cons_run1.json")
try:
    with open(cfg, "r") as f:
        j = json.load(f)
    print(j.get("save_dir", "checkpoints/ema_cons_run1"))
except Exception:
    print("checkpoints/ema_cons_run1")
PY
)"

GEN_CKPT="${GEN_CKPT:-}"
if [[ -z "${GEN_CKPT}" ]]; then
  if [[ -f "${SAVE_DIR}/gen_best.pth" ]]; then
    GEN_CKPT="${SAVE_DIR}/gen_best.pth"
  else
    # Prefer EMA if present.
    GEN_CKPT="$(ls -1 "${SAVE_DIR}"/gen_*_ema.pth 2>/dev/null | sort | tail -n 1 || true)"
    if [[ -z "${GEN_CKPT}" ]]; then
      GEN_CKPT="$(ls -1 "${SAVE_DIR}"/gen_*.pth 2>/dev/null | sort | tail -n 1 || true)"
    fi
  fi
fi

if [[ -z "${GEN_CKPT}" || ! -f "${GEN_CKPT}" ]]; then
  echo "ERROR: GEN_CKPT not found. Set GEN_CKPT=... or ensure ${SAVE_DIR}/gen_best.pth exists."
  exit 2
fi

echo ""
echo "========================================="
echo " Step 2/4: Inference (test.py)"
echo "   GEN_CKPT=${GEN_CKPT}"
echo "   OUT_DIR=${OUT_DIR}"
echo "========================================="
python test.py \
  --ckpt "${GEN_CKPT}" \
  --boundary_ckpt "${BOUNDARY_CKPT}" \
  --refiner_ckpt "${REFINER_CKPT}" \
  --lora_ckpt "${LORA_CKPT}" \
  --lora_transformer_ckpt "${LORA_TRANSFORMER_CKPT}" \
  --video_dir "${VIDEO_DIR}" \
  --mask_dir "${MASK_DIR}" \
  --width "${W}" --height "${H}" \
  --out_dir "${OUT_DIR}"

echo ""
echo "========================================="
echo " Step 3/4: Building submission ZIP"
echo "   input_dir=${OUT_DIR}"
echo "   zip_name=${ZIP_NAME}"
echo "========================================="
python make_submission.py \
  --input_dir "${OUT_DIR}" \
  --zip_name "${ZIP_NAME}"

echo ""
echo "========================================="
echo " Step 4/4: PSNR / SSIM (psnr.py)"
echo "   input_dir=./Vroom"
echo "========================================="
python psnr.py

echo ""
echo "Done:"
echo "  outputs: ${OUT_DIR}"
echo "  zip    : ${ZIP_NAME}"

