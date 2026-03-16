#!/bin/bash
set -euo pipefail

cd /media/ajeet/data/MINI_BTP/B2SCVR

# Avoid Intel/OpenMP shared-memory issues on some systems.
export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"
export KMP_DISABLE_SHM="${KMP_DISABLE_SHM:-1}"
export KMP_USE_SHM="${KMP_USE_SHM:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

# Generates base-pipeline predictions for the first N train videos into output_train/.
# Uses symlink subsets to avoid copying the dataset.
#
# Override via env:
#   N=1500 OUT_DIR=output_train bash gen_train_preds_1500.sh
#
# Important: do NOT pass --refiner_ckpt here (no circular training).

N="${N:-1500}"
OUT_DIR="${OUT_DIR:-output_train}"

BSC_ROOT="${BSC_ROOT:-/home/ajeet/MINIP/data/train/train_bsc_imgs}"
MASK_ROOT="${MASK_ROOT:-/home/ajeet/MINIP/data/train/train_masks}"

SUB_BSC="${SUB_BSC:-/tmp/train_bsc_sub_${N}}"
SUB_MSK="${SUB_MSK:-/tmp/train_msk_sub_${N}}"

CKPT="${CKPT:-/media/ajeet/data/MINI_BTP/B2SCVR/checkpoints/ema_cons_run1/gen_best.pth}"
BOUNDARY_CKPT="${BOUNDARY_CKPT:-/media/ajeet/data/MINI_BTP/B2SCVR/checkpoints/boundary_head_best.pth}"
LORA_CKPT="${LORA_CKPT:-/media/ajeet/data/MINI_BTP/B2SCVR/checkpoints/lora_sam2_run3_best.pth}"
LORA_TRANSFORMER_CKPT="${LORA_TRANSFORMER_CKPT:-/media/ajeet/data/MINI_BTP/B2SCVR/checkpoints/lora_transformer_on_sam2_best.pth}"

W="${W:-432}"
H="${H:-240}"

rm -rf "$SUB_BSC" "$SUB_MSK"
mkdir -p "$SUB_BSC" "$SUB_MSK"
mkdir -p "$OUT_DIR"

# Select first N videos deterministically without triggering pipefail/broken-pipe.
while read -r v; do
  [ -z "$v" ] && continue
  ln -s "$BSC_ROOT/$v" "$SUB_BSC/$v"
  ln -s "$MASK_ROOT/$v" "$SUB_MSK/$v"
done < <(python - <<PY
import os
N=int("${N}")
root=r"""${BSC_ROOT}"""
vids=sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root,d))])
for v in vids[:N]:
    print(v)
PY
)

python test.py \
  --ckpt "$CKPT" \
  --boundary_ckpt "$BOUNDARY_CKPT" \
  --lora_ckpt "$LORA_CKPT" \
  --lora_transformer_ckpt "$LORA_TRANSFORMER_CKPT" \
  --video_dir "$SUB_BSC" \
  --mask_dir  "$SUB_MSK" \
  --width "$W" --height "$H" \
  --out_dir "$OUT_DIR"

# Fail loudly if nothing was produced.
if ! find "$OUT_DIR" -mindepth 2 -maxdepth 3 -type f -name "*.png" | head -n 1 | rg -q .; then
  echo "ERROR: No PNG predictions were generated under: $OUT_DIR"
  echo "Check the console output above for errors from test.py."
  exit 1
fi
