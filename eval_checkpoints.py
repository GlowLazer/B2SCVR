"""
eval_checkpoints.py — rank all gen_*.pth checkpoints by PSNR on worst10 subset.

Usage:
    python eval_checkpoints.py                  # full TTA (slow, accurate)
    python eval_checkpoints.py --quick          # no reverse-TTA (2x faster)
    python eval_checkpoints.py --ckpt_dir checkpoints/finetune_5k
    python eval_checkpoints.py --video_dir worst10/bsc_imgs --mask_dir worst10/masks --gt_dir worst10/gt_imgs
"""

import argparse
import glob
import importlib
import os

import cv2
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim_func
from tqdm import tqdm

from core.utils import to_tensors

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_CKPT_DIR  = "checkpoints/finetune_5k"
DEFAULT_VIDEO_DIR = "worst10/bsc_imgs"
DEFAULT_MASK_DIR  = "worst10/masks"
DEFAULT_GT_DIR    = "worst10/gt_imgs"
BOUNDARY_CKPT     = "checkpoints/boundary_head_best.pth"
WIDTH, HEIGHT     = 432, 240
MODEL_NAME        = "b2scvr_hq"
NEIGHBOR_STRIDE   = 5
REF_STEP          = 10
FRAMESTRIDE       = 30
# ──────────────────────────────────────────────────────────────────────────────


def read_mask(mpath, size):
    masks = []
    for mp in sorted(os.listdir(mpath)):
        m = Image.open(os.path.join(mpath, mp))
        m = m.resize(size, Image.NEAREST)
        m = np.array(m.convert("L"))
        m = np.array(m > 0).astype(np.uint8)
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), iterations=4)
        masks.append(Image.fromarray(m * 255))
    return masks


def read_frames(vpath):
    frames = []
    for name in sorted(os.listdir(vpath)):
        img = cv2.imread(os.path.join(vpath, name))
        if img is None:
            continue
        frames.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    return frames


def get_ref_index(f, neighbor_ids, length, ref_step):
    return [i for i in range(0, length, ref_step) if i not in neighbor_ids]


def run_pass(model, device, rframes, rmasks, h, w, reverse=False, boundary_head=None):
    """
    Run one inference pass.
    Returns arrays in ORIGINAL (forward) order regardless of `reverse`.
    Caller can therefore always use index i directly.
    """
    if reverse:
        rframes = list(reversed(rframes))
        rmasks  = list(reversed(rmasks))

    video_length = len(rframes)
    sum_frames   = [None] * video_length
    count_frames = [0]    * video_length

    x_frames = [rframes[i:i + FRAMESTRIDE] for i in range(0, video_length, FRAMESTRIDE)]
    x_masks  = [rmasks[i:i + FRAMESTRIDE]  for i in range(0, video_length, FRAMESTRIDE)]

    for itern in range(len(x_frames)):
        stride_length = len(x_frames[itern])
        imgs  = to_tensors()(x_frames[itern]).unsqueeze(0) * 2 - 1
        raw_frames = [np.array(f).astype(np.uint8) for f in x_frames[itern]]
        binary_masks = [
            np.expand_dims((np.array(m) != 0).astype(np.uint8), 2)
            for m in x_masks[itern]
        ]
        masks = to_tensors()(x_masks[itern]).unsqueeze(0)
        imgs, masks = imgs.to(device), masks.to(device)

        for f in range(0, stride_length, NEIGHBOR_STRIDE):
            neighbor_ids = [i for i in range(
                max(0, f - NEIGHBOR_STRIDE), min(stride_length, f + NEIGHBOR_STRIDE + 1))]
            ref_ids  = get_ref_index(f, neighbor_ids, stride_length, REF_STEP)
            sel_imgs  = imgs[:1, neighbor_ids + ref_ids]
            sel_masks = masks[:1, neighbor_ids + ref_ids]

            with torch.no_grad():
                pred_imgs, _ = model(sel_imgs, sel_imgs, sel_masks, len(neighbor_ids))
                pred_imgs = pred_imgs[:, :, :h, :w]

                if boundary_head is not None:
                    n_nb = len(neighbor_ids)
                    restored = pred_imgs[:n_nb]
                    original = sel_imgs[0, :n_nb, :, :h, :w]
                    bmask    = sel_masks[0, :n_nb, :, :h, :w]
                    refined, _ = boundary_head(restored, original, bmask)
                    pred_imgs[:n_nb] = refined

                pred_imgs = (pred_imgs + 1) / 2
                pred_np   = pred_imgs.cpu().permute(0, 2, 3, 1).numpy() * 255

            for i, idx in enumerate(neighbor_ids):
                gidx   = itern * FRAMESTRIDE + idx
                mask2d = binary_masks[idx][:, :, 0:1].astype(np.float32)
                img    = (pred_np[i].astype(np.float32) * mask2d +
                          raw_frames[idx].astype(np.float32) * (1.0 - mask2d)).astype(np.uint8)
                if sum_frames[gidx] is None:
                    sum_frames[gidx] = img.astype(np.float32)
                else:
                    sum_frames[gidx] += img.astype(np.float32)
                count_frames[gidx] += 1

    # Un-reverse so callers always get arrays in original forward order
    if reverse:
        sum_frames   = list(reversed(sum_frames))
        count_frames = list(reversed(count_frames))

    return sum_frames, count_frames


def infer_video(model, device, video_path, mask_path, size, quick, boundary_head=None):
    rframes = read_frames(video_path)
    if not rframes:
        return None
    rframes = [f.resize(size) for f in rframes]
    h, w = size[1], size[0]
    rmasks = read_mask(mask_path, size)

    sum_fwd, cnt_fwd = run_pass(model, device, rframes, rmasks, h, w,
                                reverse=False, boundary_head=boundary_head)

    comp_frames = []
    if quick:
        for i in range(len(rframes)):
            f = sum_fwd[i] / cnt_fwd[i] if sum_fwd[i] is not None else None
            comp_frames.append(f.astype(np.uint8) if f is not None else None)
    else:
        # run_pass(reverse=True) already returns arrays in original forward order
        sum_bwd, cnt_bwd = run_pass(model, device, rframes, rmasks, h, w,
                                    reverse=True, boundary_head=boundary_head)
        for i in range(len(rframes)):
            fwd = sum_fwd[i] / cnt_fwd[i] if sum_fwd[i] is not None else None
            bwd = sum_bwd[i] / cnt_bwd[i] if sum_bwd[i] is not None else None  # FIX: use i not j
            if fwd is not None and bwd is not None:
                comp_frames.append((0.5 * fwd + 0.5 * bwd).astype(np.uint8))
            elif fwd is not None:
                comp_frames.append(fwd.astype(np.uint8))
            elif bwd is not None:
                comp_frames.append(bwd.astype(np.uint8))
            else:
                comp_frames.append(None)

    return comp_frames


def compute_psnr_video(comp_frames, gt_dir, mask_dir):
    """
    Returns (mean_psnr, mean_ssim, mean_psnr_maskonly, n_frames_evaluated).
    - mean_psnr        : full-frame PSNR on corrupted frames (matches leaderboard logic)
    - mean_psnr_maskonly: PSNR computed only over mask-pixel locations (diagnostic)
    """
    psnrs, ssims, psnrs_mask = [], [], []

    for i, frame in enumerate(comp_frames):
        if frame is None:
            continue

        # Skip uncorrupted frames (any mask pixel > 0 → frame is corrupted)
        mask_path = os.path.join(mask_dir, f"{i:05d}.png")
        mask_bin  = None
        if os.path.exists(mask_path):
            mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_raw is None or not np.any(mask_raw > 0):
                continue
            # Dilate to match read_mask() pre-processing
            mask_bin = cv2.dilate(
                (mask_raw > 0).astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
                iterations=4
            ).astype(bool)

        # Find GT file
        gt_path = None
        for ext in (".jpg", ".png"):
            cand = os.path.join(gt_dir, f"{i:05d}{ext}")
            if os.path.exists(cand):
                gt_path = cand
                break
        if gt_path is None:
            continue

        gt_bgr = cv2.imread(gt_path)
        if gt_bgr is None:
            continue

        pred_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if pred_bgr.shape[:2] != gt_bgr.shape[:2]:
            pred_bgr = cv2.resize(pred_bgr, (gt_bgr.shape[1], gt_bgr.shape[0]),
                                  interpolation=cv2.INTER_CUBIC)

        pred = cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2RGB).astype(np.float64)
        gt   = cv2.cvtColor(gt_bgr,   cv2.COLOR_BGR2RGB).astype(np.float64)

        # Full-frame PSNR (leaderboard metric)
        mse  = np.mean((pred - gt) ** 2)
        psnr = 20.0 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float("inf")
        ssim = ssim_func(pred, gt, data_range=255.0, channel_axis=2)
        psnrs.append(psnr)
        ssims.append(ssim)

        # Masked-region PSNR (diagnostic: how well are we filling the holes?)
        if mask_bin is not None:
            # Resize mask_bin to pred resolution if needed
            if mask_bin.shape[:2] != pred.shape[:2]:
                mask_bin = cv2.resize(
                    mask_bin.astype(np.uint8),
                    (pred.shape[1], pred.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)
            mask3 = np.stack([mask_bin] * 3, axis=-1)
            p_px  = pred[mask3].reshape(-1)
            g_px  = gt[mask3].reshape(-1)
            if p_px.size > 0:
                mse_m = np.mean((p_px - g_px) ** 2)
                psnrs_mask.append(20.0 * np.log10(255.0 / np.sqrt(mse_m)) if mse_m > 0 else float("inf"))

    n = len(psnrs)
    if n == 0:
        return None, None, None, 0

    mp   = float(np.mean(psnrs))
    ms   = float(np.mean(ssims))
    mpm  = float(np.mean(psnrs_mask)) if psnrs_mask else None
    return mp, ms, mpm, n


def load_model(ckpt_path, device):
    net   = importlib.import_module("model." + MODEL_NAME)
    model = net.InpaintGenerator(init_weights=False).to(device)
    data  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(data)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir",    default=DEFAULT_CKPT_DIR)
    ap.add_argument("--video_dir",   default=DEFAULT_VIDEO_DIR)
    ap.add_argument("--mask_dir",    default=DEFAULT_MASK_DIR)
    ap.add_argument("--gt_dir",      default=DEFAULT_GT_DIR)
    ap.add_argument("--quick",       action="store_true",
                    help="Skip reverse-TTA (2x faster, same relative ranking)")
    ap.add_argument("--no_boundary", action="store_true",
                    help="Skip boundary head")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size   = (WIDTH, HEIGHT)

    ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, "gen_*.pth")))
    if not ckpts:
        print(f"No gen_*.pth found in {args.ckpt_dir}"); return

    videos = sorted([d for d in os.listdir(args.video_dir)
                     if os.path.isdir(os.path.join(args.video_dir, d))])
    if not videos:
        print(f"No video subdirs in {args.video_dir}"); return

    print(f"Checkpoints : {len(ckpts)}")
    print(f"Videos      : {len(videos)}")
    print(f"Mode        : {'quick (fwd only)' if args.quick else 'full TTA (fwd+bwd)'}")
    print()

    bhead = None
    if not args.no_boundary and os.path.isfile(BOUNDARY_CKPT):
        from model.boundary_head import BoundaryRefinementHead
        bhead = BoundaryRefinementHead(channels=64).to(device)
        bhead.load_state_dict(torch.load(BOUNDARY_CKPT, map_location=device))
        bhead.eval()
        print(f"Loaded boundary head: {BOUNDARY_CKPT}\n")

    results = []   # (ckpt_name, mean_psnr, mean_ssim, mean_psnr_mask, total_frames)

    for ckpt_path in ckpts:
        ckpt_name = os.path.basename(ckpt_path)
        print(f"{'='*70}")
        print(f"Checkpoint: {ckpt_name}")
        model = load_model(ckpt_path, device)

        vid_psnrs, vid_ssims, vid_psnrs_m, total_frames = [], [], [], 0

        for video in videos:
            vpath = os.path.join(args.video_dir, video)
            mpath = os.path.join(args.mask_dir,  video)
            gpath = os.path.join(args.gt_dir,    video)
            if not os.path.isdir(mpath) or not os.path.isdir(gpath):
                continue

            comp = infer_video(model, device, vpath, mpath, size, args.quick, bhead)
            if comp is None:
                continue

            psnr, ssim, psnr_m, n_frames = compute_psnr_video(comp, gpath, mpath)
            total_frames += n_frames
            if psnr is not None:
                vid_psnrs.append(psnr)
                vid_ssims.append(ssim)
                if psnr_m is not None:
                    vid_psnrs_m.append(psnr_m)
                mask_str = f"  maskPSNR {psnr_m:.2f}" if psnr_m else ""
                print(f"  {video:<22s}  PSNR {psnr:.2f} dB   SSIM {ssim:.4f}{mask_str}   [{n_frames} frames]")
            else:
                print(f"  {video:<22s}  SKIP (0 evaluated frames)")

        if vid_psnrs:
            mp  = float(np.mean(vid_psnrs))
            ms  = float(np.mean(vid_ssims))
            mpm = float(np.mean(vid_psnrs_m)) if vid_psnrs_m else None
            mask_str = f"   maskPSNR {mpm:.4f}" if mpm else ""
            print(f"  >>> mean PSNR {mp:.4f} dB   SSIM {ms:.4f}{mask_str}   [{total_frames} frames total]")
            results.append((ckpt_name, mp, ms, mpm, total_frames))
        print()

        del model
        torch.cuda.empty_cache()

    # ── Final ranking ──────────────────────────────────────────────────────────
    if not results:
        print("No results collected."); return

    results.sort(key=lambda x: x[1], reverse=True)
    print("\n" + "="*78)
    print("RANKING  (best → worst PSNR on worst10 subset)")
    print("="*78)
    print(f"{'Rank':<5} {'Checkpoint':<22} {'PSNR':>10} {'SSIM':>8} {'maskPSNR':>10} {'Frames':>7}")
    print("-"*78)
    for rank, (name, psnr, ssim, psnr_m, nf) in enumerate(results, 1):
        m_str  = f"{psnr_m:>10.4f}" if psnr_m else f"{'N/A':>10}"
        marker = "  ← BEST" if rank == 1 else ""
        print(f"{rank:<5} {name:<22} {psnr:>10.4f} {ssim:>8.4f} {m_str} {nf:>7}{marker}")
    print()
    print(f"Best checkpoint : {results[0][0]}  ({results[0][1]:.4f} dB)")
    print()

    # Warn if frame count varies across checkpoints (could signal skipped frames)
    frame_counts = [r[4] for r in results]
    if len(set(frame_counts)) > 1:
        print("⚠  Frame counts differ across checkpoints — check for None frames:")
        for name, _, _, _, nf in results:
            print(f"   {name}  {nf} frames")


if __name__ == "__main__":
    main()
