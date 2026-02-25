import os
import glob
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim_func

GT_DIR   = "/media/ajeet/data/MINI_BTP/data/validation/gt_imgs"
MASK_DIR = "/media/ajeet/data/MINI_BTP/data/validation/masks"
OUT_DIR  = "./Vroom"

video_dirs = sorted([d for d in glob.glob(os.path.join(OUT_DIR, "*")) if os.path.isdir(d)])
total = len(video_dirs)

all_psnrs, all_ssims = [], []

for current, frame_seq_dir in enumerate(video_dirs, 1):
    video_name = os.path.basename(frame_seq_dir)
    gt_dir   = os.path.join(GT_DIR,   video_name)
    mask_dir = os.path.join(MASK_DIR, video_name)

    if not os.path.isdir(gt_dir):
        print(f"[{current}/{total}] SKIP {video_name}  (no GT at {gt_dir})")
        continue

    pred_files = sorted(glob.glob(os.path.join(frame_seq_dir, "*.jpg")))
    if not pred_files:
        print(f"[{current}/{total}] SKIP {video_name}  (no frames)")
        continue

    psnrs, ssims = [], []
    skip_gt, skip_read, skip_mask = 0, 0, 0

    for pred_path in pred_files:
        # Only evaluate bitstream-corrupted frames (non-empty mask)
        stem = os.path.splitext(os.path.basename(pred_path))[0]
        mask_path = os.path.join(mask_dir, stem + ".png")
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None or not np.any(mask > 0):
                skip_mask += 1
                continue

        gt_path = os.path.join(gt_dir, os.path.basename(pred_path))
        if not os.path.exists(gt_path):
            skip_gt += 1
            continue

        pred_bgr = cv2.imread(pred_path)
        gt_bgr   = cv2.imread(gt_path)
        if pred_bgr is None or gt_bgr is None:
            skip_read += 1
            continue

        pred = cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2RGB).astype(np.float64)
        gt   = cv2.cvtColor(gt_bgr,   cv2.COLOR_BGR2RGB).astype(np.float64)

        mse  = np.mean((pred - gt) ** 2)
        psnr = 20.0 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float("inf")
        ssim = ssim_func(pred, gt, data_range=255.0, channel_axis=2)
        psnrs.append(psnr)
        ssims.append(ssim)

    if psnrs:
        v_psnr, v_ssim = np.mean(psnrs), np.mean(ssims)
        all_psnrs.append(v_psnr)
        all_ssims.append(v_ssim)
        print(f"[{current}/{total}] {video_name:<20s}  PSNR: {v_psnr:.4f} dB   SSIM: {v_ssim:.4f}   ({len(psnrs)} frames)")
    else:
        reasons = []
        if skip_mask: reasons.append(f"{skip_mask} uncorrupted frames skipped")
        if skip_gt:   reasons.append(f"{skip_gt} missing GT")
        if skip_read: reasons.append(f"{skip_read} imread failed")
        print(f"[{current}/{total}] {video_name:<20s}  no valid frames  [{', '.join(reasons) or 'unknown'}]")

print()
print("=" * 56)
if all_psnrs:
    print(f"Videos evaluated  : {len(all_psnrs)} / {total}")
    print(f"Overall avg PSNR  : {np.mean(all_psnrs):.4f} dB")
    print(f"Overall avg SSIM  : {np.mean(all_ssims):.4f}")
else:
    print("No valid results found.")
