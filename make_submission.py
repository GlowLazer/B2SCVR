"""
Upsample outputs from 432x240 inference resolution to native input resolution,
then package as Vroom.zip for CodaLab submission.
"""
import os
import glob
import shutil
import zipfile
import numpy as np
import cv2
from PIL import Image

INPUT_DIR  = "/media/ajeet/data/MINI_BTP/data/validation/bsc_imgs"
OUT_DIR    = "./outputs"
STAGE_DIR  = "./Vroom"
ZIP_NAME   = "./Vroom.zip"

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

    # Upsample and save each output frame
    out_video_dir = os.path.join(STAGE_DIR, video_name)
    os.makedirs(out_video_dir)

    pred_frames = sorted(glob.glob(os.path.join(frame_seq_dir, "*.jpg")))
    for pred_path in pred_frames:
        fname = os.path.basename(pred_path)
        pred = cv2.imread(pred_path)
        if pred is None:
            continue
        if pred.shape[:2] != (native_h, native_w):
            pred = cv2.resize(pred, (native_w, native_h), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(out_video_dir, fname), pred,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])

    print(f"[{current}/{total}] {video_name}  {pred.shape[1]}x{pred.shape[0]} -> {native_w}x{native_h}  ({len(pred_frames)} frames)")

# Write readme.txt
with open(os.path.join(STAGE_DIR, "readme.txt"), "w") as f:
    f.write("""runtime per frame [s] : 1.5
CPU[1] / GPU[0] : 0
Extra Data [1] / No Extra Data [0] : 0
Other description : Baseline model for NTIRE 2026 BSCVR Challenge submitted by team Vroom.
Method: B2SCVR (Bitstream-corrupted Video Restoration baseline).
Transformer-based video inpainting with SAM2 backbone and temporal attention.
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
