"""
Upsample outputs from 432x240 inference resolution to native input resolution,
then package as Vroom.zip for CodaLab submission.
"""
import argparse
import os
import glob
import shutil
import zipfile
import numpy as np
import cv2
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument('--input_dir', default='./outputs',
                help='Directory containing per-video output subdirs (default: ./outputs)')
ap.add_argument('--bsc_dir', required=True,
                help='Path to BSC input frames root (e.g. /data/ntire_test/bsc_imgs)')
ap.add_argument('--mask_dir', required=True,
                help='Path to masks root (e.g. /data/ntire_test/masks)')
ap.add_argument('--zip_name',  default='./Vroom.zip')
_args = ap.parse_args()

INPUT_DIR  = _args.bsc_dir
MASK_DIR   = _args.mask_dir
OUT_DIR    = _args.input_dir
STAGE_DIR  = "./Vroom"
ZIP_NAME   = _args.zip_name

# Clean staging dir
if os.path.exists(STAGE_DIR):
    shutil.rmtree(STAGE_DIR)
os.makedirs(STAGE_DIR)

video_dirs = sorted(glob.glob(os.path.join(OUT_DIR, "*")))
video_dirs = [d for d in video_dirs if os.path.isdir(d)]
total = len(video_dirs)

for current, video_dir in enumerate(video_dirs, 1):
    video_name = os.path.basename(video_dir)
    frame_seq_dir = os.path.join(video_dir, "frame_seq")
    input_video_dir = os.path.join(INPUT_DIR, video_name)

    if not os.path.isdir(frame_seq_dir):
        print(f"[{current}/{total}] SKIP {video_name}  (no frame_seq)")
        continue
    if not os.path.isdir(input_video_dir):
        print(f"[{current}/{total}] SKIP {video_name}  (no input dir)")
        continue

    # Get native resolution from first input frame
    input_frames = sorted(glob.glob(os.path.join(input_video_dir, "*.jpg")))
    if not input_frames:
        print(f"[{current}/{total}] SKIP {video_name}  (no input frames)")
        continue
    ref = cv2.imread(input_frames[0])
    native_h, native_w = ref.shape[:2]

    out_video_dir = os.path.join(STAGE_DIR, video_name)
    os.makedirs(out_video_dir)

    # Cap to BSC frame count — prevents stale frames from accumulating across runs
    pred_frames = sorted(
        glob.glob(os.path.join(frame_seq_dir, "*.png")) +
        glob.glob(os.path.join(frame_seq_dir, "*.jpg"))
    )[:len(input_frames)]

    for pred_path in pred_frames:
        stem = os.path.splitext(os.path.basename(pred_path))[0]
        pred = cv2.imread(pred_path)
        if pred is None:
            continue
        if pred.shape[:2] != (native_h, native_w):
            pred = cv2.resize(pred, (native_w, native_h), interpolation=cv2.INTER_CUBIC)

        # Hard mask composite: replace uncorrupted pixels with the BSC source frame.
        # Empirically gains ~+0.17 dB avg PSNR vs pure bicubic upsampling.
        input_path = os.path.join(input_video_dir, stem + ".jpg")
        mask_path  = os.path.join(MASK_DIR, video_name, stem + ".png")
        if os.path.exists(mask_path) and os.path.exists(input_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            bsc  = cv2.imread(input_path)
            if mask is not None and bsc is not None:
                if mask.shape[:2] != (native_h, native_w):
                    mask = cv2.resize(mask, (native_w, native_h), interpolation=cv2.INTER_NEAREST)
                if bsc.shape[:2] != (native_h, native_w):
                    bsc = cv2.resize(bsc, (native_w, native_h), interpolation=cv2.INTER_CUBIC)
                corrupted = np.stack([mask > 0] * 3, axis=2)
                pred = np.where(corrupted, pred, bsc).astype(np.uint8)

        # Save as PNG (evaluation server expects .png)
        cv2.imwrite(os.path.join(out_video_dir, stem + ".png"), pred)

    print(f"[{current}/{total}] {video_name}  -> {native_w}x{native_h}  ({len(pred_frames)} frames)")

# Write readme.txt
with open(os.path.join(STAGE_DIR, "readme.txt"), "w") as f:
    f.write("""runtime per frame [s] : 1.5
CPU[1] / GPU[0] : 0
Extra Data [1] / No Extra Data [0] : 0
Other description : NTIRE 2026 BSCVR Challenge submission by team Vroom.
Method: Enhanced B2SCVR — Transformer-based video inpainting with the following additions over baseline:
1. SAM2 (Segment Anything Model 2) feature extractor with LoRA fine-tuning for richer spatial priors.
2. Boundary Refinement Head: corrects mask-boundary artifacts via soft alpha blending with a learned delta.
3. Residual Refiner: lightweight U-Net that applies per-frame residual corrections post-inpainting.
4. Temporal Difference (TD) loss with ramp-up scheduling for improved temporal consistency.
5. MSE-only fine-tuning runs (no discriminator) to stabilize reconstruction quality.
6. Loss weight distribution tuning: varied hole/valid weights (e.g. 1.5/0.5, 1.0/0.0) across runs to focus on corrupted region reconstruction.
7. Reverse TTA: bidirectional (forward + reverse) temporal ensemble averaged inside mask regions.
8. Hard mask composite at submission: uncorrupted pixels replaced with BSC source for +~0.17 dB PSNR.
Inference at 432x240, outputs upsampled to native resolution with bicubic interpolation.
No extra training data beyond the official challenge training set.
""")

# Create ZIP
print(f"\nCreating {ZIP_NAME} ...")
if os.path.exists(ZIP_NAME):
    os.remove(ZIP_NAME)
with zipfile.ZipFile(ZIP_NAME, "w", zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(STAGE_DIR):
        for file in sorted(files):
            abs_path = os.path.join(root, file)
            arc_path = os.path.relpath(abs_path, STAGE_DIR)
            zf.write(abs_path, arc_path)

size_mb = os.path.getsize(ZIP_NAME) / 1e6
print(f"Done: {ZIP_NAME}  ({size_mb:.1f} MB)")
