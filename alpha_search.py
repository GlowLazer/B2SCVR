# """
# Per-video alpha search for forward/backward TTA blending.

# For each video in ./outputs/:
#   - Reads fwd/ and bwd/ frame sequences (saved by test.py)
#   - Tries alpha in [0.0, 0.05, ..., 1.0]  (alpha * fwd + (1-alpha) * bwd)
#   - Evaluates PSNR against GT on corrupted frames only (non-empty mask)
#   - Saves the best-alpha blended frames to frame_seq/ (overwriting default)
#   - Prints per-video best alpha + PSNR, then overall average

# After running this, run make_submission.py as usual to build Vroom.zip.

# Usage:
#     python alpha_search.py
# """

# import os
# import glob
# import shutil
# import numpy as np
# import cv2

# GT_DIR   = "/media/ajeet/data/MINI_BTP/data/validation/gt_imgs"
# MASK_DIR = "/media/ajeet/data/MINI_BTP/data/validation/masks"
# OUT_DIR  = "./outputs"

# ALPHAS = np.round(np.arange(0.0, 1.0 + 1e-9, 0.05), 2)


# def read_frames(directory):
#     """Return sorted list of (stem, bgr_array) from a directory of PNG/JPG frames."""
#     paths = sorted(glob.glob(os.path.join(directory, "*.png")) +
#                    glob.glob(os.path.join(directory, "*.jpg")))
#     frames = []
#     for p in paths:
#         stem = os.path.splitext(os.path.basename(p))[0]
#         img = cv2.imread(p)
#         frames.append((stem, img))
#     return frames


# def psnr_value(pred_bgr, gt_bgr):
#     pred = pred_bgr.astype(np.float64)
#     gt   = gt_bgr.astype(np.float64)
#     mse  = np.mean((pred - gt) ** 2)
#     return 20.0 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float("inf")


# def blend(fwd_bgr, bwd_bgr, alpha):
#     """alpha * fwd + (1-alpha) * bwd, clipped to uint8."""
#     return np.clip(alpha * fwd_bgr.astype(np.float32) +
#                    (1.0 - alpha) * bwd_bgr.astype(np.float32), 0, 255).astype(np.uint8)


# def evaluate_alpha(fwd_frames, bwd_frames, gt_dir, mask_dir, alpha):
#     """Return mean PSNR over corrupted frames for a given alpha."""
#     psnrs = []
#     for (stem, fwd_bgr), (_, bwd_bgr) in zip(fwd_frames, bwd_frames):
#         # Only evaluate on frames with non-empty corruption mask
#         mask_path = os.path.join(mask_dir, stem + ".png")
#         if os.path.exists(mask_path):
#             mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#             if mask is None or not np.any(mask > 0):
#                 continue

#         # Find GT
#         gt_path = None
#         for ext in ('.jpg', '.png'):
#             cand = os.path.join(gt_dir, stem + ext)
#             if os.path.exists(cand):
#                 gt_path = cand
#                 break
#         if gt_path is None:
#             continue

#         gt_bgr = cv2.imread(gt_path)
#         if gt_bgr is None or fwd_bgr is None or bwd_bgr is None:
#             continue

#         pred = blend(fwd_bgr, bwd_bgr, alpha)

#         # Upsample pred to GT resolution if needed
#         if pred.shape[:2] != gt_bgr.shape[:2]:
#             pred = cv2.resize(pred, (gt_bgr.shape[1], gt_bgr.shape[0]),
#                               interpolation=cv2.INTER_CUBIC)

#         psnrs.append(psnr_value(pred, gt_bgr))

#     return np.mean(psnrs) if psnrs else None


# def save_best_frames(fwd_frames, bwd_frames, best_alpha, frame_seq_dir):
#     """Write best-alpha blended frames to frame_seq_dir (overwrite existing)."""
#     if os.path.exists(frame_seq_dir):
#         shutil.rmtree(frame_seq_dir)
#     os.makedirs(frame_seq_dir)
#     for i, ((stem, fwd_bgr), (_, bwd_bgr)) in enumerate(zip(fwd_frames, bwd_frames)):
#         if fwd_bgr is None and bwd_bgr is None:
#             continue
#         if fwd_bgr is None:
#             out = bwd_bgr
#         elif bwd_bgr is None:
#             out = fwd_bgr
#         else:
#             out = blend(fwd_bgr, bwd_bgr, best_alpha)
#         cv2.imwrite(os.path.join(frame_seq_dir, f"{stem}.jpg"), out)


# def main():
#     video_dirs = sorted([d for d in glob.glob(os.path.join(OUT_DIR, "*"))
#                          if os.path.isdir(d)])
#     total = len(video_dirs)

#     all_best_psnrs  = []
#     all_video_names = []

#     print(f"Alpha search over {len(ALPHAS)} values: {ALPHAS[0]:.2f} … {ALPHAS[-1]:.2f}\n")

#     for current, video_dir in enumerate(video_dirs, 1):
#         video_name = os.path.basename(video_dir)
#         fwd_dir  = os.path.join(video_dir, "fwd")
#         bwd_dir  = os.path.join(video_dir, "bwd")
#         frame_seq_dir = os.path.join(video_dir, "frame_seq")

#         if not os.path.isdir(fwd_dir) or not os.path.isdir(bwd_dir):
#             print(f"[{current}/{total}] SKIP {video_name}  (no fwd/bwd — run test.py first)")
#             continue

#         fwd_frames = read_frames(fwd_dir)
#         bwd_frames = read_frames(bwd_dir)

#         if len(fwd_frames) == 0 or len(fwd_frames) != len(bwd_frames):
#             print(f"[{current}/{total}] SKIP {video_name}  "
#                   f"(fwd={len(fwd_frames)} bwd={len(bwd_frames)} frame count mismatch)")
#             continue

#         gt_dir   = os.path.join(GT_DIR,   video_name)
#         mask_dir = os.path.join(MASK_DIR, video_name)

#         if not os.path.isdir(gt_dir):
#             print(f"[{current}/{total}] SKIP {video_name}  (no GT directory)")
#             continue

#         # Search over alphas
#         best_alpha, best_psnr = 0.5, -1.0
#         for alpha in ALPHAS:
#             p = evaluate_alpha(fwd_frames, bwd_frames, gt_dir, mask_dir, alpha)
#             if p is not None and p > best_psnr:
#                 best_psnr  = p
#                 best_alpha = alpha

#         # Save best-blended frames → frame_seq/
#         save_best_frames(fwd_frames, bwd_frames, best_alpha, frame_seq_dir)

#         all_best_psnrs.append(best_psnr)
#         all_video_names.append(video_name)
#         print(f"[{current}/{total}] {video_name:<20s}  best_alpha={best_alpha:.2f}  "
#               f"PSNR={best_psnr:.4f} dB  ({len(fwd_frames)} frames)  -> frame_seq/ updated")

#     print()
#     print("=" * 60)
#     if all_best_psnrs:
#         print(f"Videos evaluated  : {len(all_best_psnrs)} / {total}")
#         print(f"Overall avg PSNR  : {np.mean(all_best_psnrs):.4f} dB  (best possible per video)")
#     else:
#         print("No valid results found.")
#     print()
#     print("Run make_submission.py next to build Vroom.zip with these best frames.")


# if __name__ == "__main__":
#     main()
