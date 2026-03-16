# -*- coding: utf-8 -*-
import cv2
from PIL import Image
import numpy as np
import importlib
import os
import argparse
from tqdm import tqdm
import torch

from core.utils import to_tensors


parser = argparse.ArgumentParser(description="B2SCVR")
parser.add_argument("-v", "--video", type=str, required=False)
parser.add_argument("-c", "--ckpt", type=str, required=True)
parser.add_argument("-m", "--dac_mask", type=str, required=False)
parser.add_argument("--video_dir", type=str, required=False,
                    help="Batch mode: directory containing per-video subdirs")
parser.add_argument("--mask_dir", type=str, required=False,
                    help="Batch mode: directory containing per-video mask subdirs")
parser.add_argument("--model", type=str, default='b2scvr_hq')
# parser.add_argument("--type", type=str, required=True)
parser.add_argument("--step", type=int, default=10)
parser.add_argument("--num_ref", type=int, default=-1)
parser.add_argument("--neighbor_stride", type=int, default=5)
parser.add_argument("--framestride", type=int, default=30)
parser.add_argument("--width", type=int)
parser.add_argument("--height", type=int)
parser.add_argument("--boundary_ckpt", type=str, default=None,
                    help="Optional: path to boundary_head.pth for boundary refinement")
parser.add_argument("--alpha_ckpt", type=str, default=None,
                    help="Optional: path to spatial_alpha.pth for per-pixel TTA blend")
parser.add_argument("--refiner_ckpt", type=str, default=None,
                    help="Optional: path to refiner.pth for residual frame refinement")
parser.add_argument("--refiner_use_lap", action="store_true",
                    help="Force 10ch refiner input (pred laplacian features). "
                         "If not set, inferred from checkpoint when possible.")
parser.add_argument("--out_dir", type=str, default="./outputs",
                    help="Output folder root. Each video is saved under <out_dir>/<video>/frame_seq/*.png")
parser.add_argument("--lora_ckpt", type=str, default=None,
                    help="Optional: path to lora_sam2.pth for SAM2 LoRA + SAMFuser weights")
parser.add_argument("--lora_transformer_ckpt", type=str, default=None,
                    help="Optional: path to lora_transformer.pth for WindowAttention LoRA")
parser.add_argument("--tta8", action="store_true",
                    help="8x self-ensemble TTA: 4 flips × fwd+rev (averages inside mask)")
parser.add_argument("--refine_passes", type=int, default=1,
                    help="Iterative self-refinement passes (default=1 = no extra pass; 2 = one re-run on composite)")
args = parser.parse_args()

ref_length = args.step  # ref_step
num_ref = args.num_ref
neighbor_stride = args.neighbor_stride


# sample reference frames from the whole video
def get_ref_index(f, neighbor_ids, length):
    ref_index = []
    if num_ref == -1:
        for i in range(0, length, ref_length):
            if i not in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, f - ref_length * (num_ref // 2))
        end_idx = min(length, f + ref_length * (num_ref // 2))
        for i in range(start_idx, end_idx + 1, ref_length):
            if i not in neighbor_ids:
                if len(ref_index) > num_ref:
                    break
                ref_index.append(i)
    return ref_index


# read frame-wise masks
def read_mask(mpath, size):
    masks = []
    mnames = os.listdir(mpath)
    mnames.sort()
    for mp in mnames:
        m = Image.open(os.path.join(mpath, mp))
        m = m.resize(size, Image.NEAREST)
        m = np.array(m.convert('L'))
        m = np.array(m > 0).astype(np.uint8)
        m = cv2.dilate(m,
                       cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
                       iterations=4)
        masks.append(Image.fromarray(m * 255))
    return masks


#  read frames from video
def read_frame_from_videos(vname):
    frames = []
    if vname.endswith('.mp4'):
        vidcap = cv2.VideoCapture(vname)
        success, image = vidcap.read()
        while success:
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            frames.append(image)
            success, image = vidcap.read()
    else:
        lst = sorted(os.listdir(vname))
        for name in lst:
            image = cv2.imread(os.path.join(vname, name))
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            frames.append(image)
    return frames


# resize frames
def resize_frames(frames, size=None):
    if size is not None:
        frames = [f.resize(size) for f in frames]
    else:
        size = frames[0].size
    return frames, size


def _tensor_flip(t, flip):
    """Flip a (..., H, W) tensor. flip ∈ {'none','H','V','HV'}."""
    if flip == 'H':
        return torch.flip(t, dims=[-1])
    if flip == 'V':
        return torch.flip(t, dims=[-2])
    if flip == 'HV':
        return torch.flip(t, dims=[-2, -1])
    return t


def _numpy_flip(arr, flip):
    """Unflip a (H, W, C) numpy array (flip is its own inverse)."""
    if flip == 'H':
        return np.ascontiguousarray(np.flip(arr, axis=1))
    if flip == 'V':
        return np.ascontiguousarray(np.flip(arr, axis=0))
    if flip == 'HV':
        return np.ascontiguousarray(np.flip(arr, axis=(0, 1)))
    return arr


def soft_boundary_blend(pred, original, binary_mask, blur_ksize=7, dilate_iter=1):
    """Replace hard mask composite with Gaussian-softened boundary blend."""
    mask = binary_mask[:, :, 0].astype(np.float32)          # (H, W)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.dilate(mask, kernel, iterations=dilate_iter)
    soft = cv2.GaussianBlur(mask, (blur_ksize, blur_ksize), 0)
    soft = soft[:, :, np.newaxis]                            # (H, W, 1)
    return (pred.astype(np.float32) * soft +
            original.astype(np.float32) * (1.0 - soft)).astype(np.uint8)


def run_inference_pass(model, device, rframes, rmasks, h, w, framestride,
                       boundary_head=None, target_indices=None, flip='none',
                       return_raw=False):
    """Run one inference pass; returns per-frame float32 sum and hit count arrays.

    target_indices: optional set of global frame indices to accumulate results for.
    If None, all frames are accumulated (default behaviour).

    return_raw=True: skip compositing; accumulate raw predicted float32 (H,W,3)
    values so the caller can composite once after multi-pass averaging.
    return_raw=False (default): composite inside mask before accumulating (existing path).
    """
    video_length = len(rframes)
    sum_frames   = [None] * video_length
    count_frames = [0]    * video_length

    x_frames = [rframes[i:i + framestride] for i in range(0, video_length, framestride)]
    x_masks  = [rmasks[i:i + framestride]  for i in range(0, video_length, framestride)]

    for itern in range(len(x_frames)):
        stride_length = len(x_frames[itern])
        imgs = to_tensors()(x_frames[itern]).unsqueeze(0) * 2 - 1
        frames = [np.array(f).astype(np.uint8) for f in x_frames[itern]]

        binary_masks = [
            np.expand_dims((np.array(m) != 0).astype(np.uint8), 2) for m in x_masks[itern]
        ]
        masks = to_tensors()(x_masks[itern]).unsqueeze(0)
        imgs, masks = imgs.to(device), masks.to(device)

        for f in tqdm(range(0, stride_length, neighbor_stride), leave=False):
            neighbor_ids = [
                i for i in range(max(0, f - neighbor_stride),
                                 min(stride_length, f + neighbor_stride + 1))
            ]
            # skip this window if none of its neighbors are target frames
            if target_indices is not None:
                global_ids = {itern * framestride + i for i in neighbor_ids}
                if not global_ids.intersection(target_indices):
                    continue

            ref_ids = get_ref_index(f, neighbor_ids, stride_length)
            selected_imgs  = imgs[:1, neighbor_ids + ref_ids, :, :, :]
            selected_masks = masks[:1, neighbor_ids + ref_ids, :, :, :]
            # Apply spatial flip augmentation before model call
            si = _tensor_flip(selected_imgs,  flip)
            sm = _tensor_flip(selected_masks, flip)
            with torch.no_grad():
                pred_imgs, _ = model(si, si, sm, len(neighbor_ids))
                pred_imgs = pred_imgs[:, :, :h, :w]   # (N+R, 3, h, w) in [-1,1]

                if boundary_head is not None:
                    n_nb = len(neighbor_ids)
                    restored = pred_imgs[:n_nb]                    # (N, 3, h, w) flipped
                    original = si[0, :n_nb, :, :h, :w]            # (N, 3, h, w) flipped
                    bmask    = sm[0, :n_nb, :, :h, :w]            # (N, 1, h, w) flipped
                    with torch.no_grad():
                        refined, _ = boundary_head(restored, original, bmask)
                    pred_imgs[:n_nb] = refined

                pred_imgs = (pred_imgs + 1) / 2
                pred_imgs = pred_imgs.cpu().permute(0, 2, 3, 1).numpy() * 255

                for i in range(len(neighbor_ids)):
                    idx  = neighbor_ids[i]
                    gidx = itern * framestride + idx
                    if target_indices is not None and gidx not in target_indices:
                        continue
                    # Undo flip before accumulating / compositing
                    p = _numpy_flip(pred_imgs[i], flip)
                    if return_raw:
                        # Caller will composite; just accumulate raw predictions
                        val = p.astype(np.float32)
                    elif boundary_head is not None:
                        mask2d = binary_masks[idx][:, :, 0:1].astype(np.float32)
                        val = (p.astype(np.float32) * mask2d +
                               frames[idx].astype(np.float32) * (1.0 - mask2d)).astype(np.float32)
                    else:
                        val = soft_boundary_blend(p, frames[idx], binary_masks[idx]).astype(np.float32)
                    if sum_frames[gidx] is None:
                        sum_frames[gidx] = val
                    else:
                        sum_frames[gidx] += val
                    count_frames[gidx] += 1

    return sum_frames, count_frames


def _tta_merge(sum_fwd, cnt_fwd, sum_bwd, cnt_bwd, video_length,
               alpha_net=None, device=None):
    """Merge forward and reverse-TTA passes into final float32 frames (0–255 range).

    If alpha_net is provided, a per-pixel spatial alpha map is computed for
    each (fwd, bwd) pair:
        alpha  = alpha_net(F, B)          shape (1, 1, H, W), range [0,1]
        merged = alpha * F + (1-alpha) * B
    Otherwise falls back to a global 0.5/0.5 blend.
    """
    frames = []
    for i in range(video_length):
        j   = video_length - 1 - i
        fwd = (sum_fwd[i] / cnt_fwd[i]) if (sum_fwd[i] is not None and cnt_fwd[i] > 0) else None
        bwd = (sum_bwd[j] / cnt_bwd[j]) if (sum_bwd[j] is not None and cnt_bwd[j] > 0) else None
        if fwd is not None and bwd is not None:
            if alpha_net is not None and device is not None:
                f_t = torch.from_numpy(fwd / 255.0).permute(2, 0, 1).float().unsqueeze(0).to(device)
                b_t = torch.from_numpy(bwd / 255.0).permute(2, 0, 1).float().unsqueeze(0).to(device)
                with torch.no_grad():
                    alpha = alpha_net(f_t, b_t)   # (1, 1, H, W)
                alpha_np = alpha[0, 0].cpu().numpy()[:, :, np.newaxis]   # (H, W, 1)
                merged = alpha_np * fwd + (1.0 - alpha_np) * bwd
                frames.append(merged.clip(0, 255).astype(np.float32))
            else:
                frames.append((0.5 * fwd + 0.5 * bwd).astype(np.float32))
        elif fwd is not None:
            frames.append(fwd.astype(np.float32))
        elif bwd is not None:
            frames.append(bwd.astype(np.float32))
        else:
            frames.append(None)
    return frames


def _infer_to_comp(model, device, rframes, rmasks, h, w, framestride, bhead, tta8, alpha_net):
    """Run one full inference round (TTA8 or fwd+bwd) and return composited float32 frames (0–255)."""
    video_length = len(rframes)
    if tta8:
        flips      = ['none', 'H', 'V', 'HV']
        all_sum    = [None] * video_length
        all_passes = [0]    * video_length
        rev_frames = list(reversed(rframes))
        rev_masks  = list(reversed(rmasks))
        for flip in flips:
            for fr, mk, reverse in (
                    [(rframes, rmasks, False), (rev_frames, rev_masks, True)]):
                label = 'rev' if reverse else 'fwd'
                print(f'  TTA8 {label} flip={flip} ...')
                sf, cf = run_inference_pass(model, device, fr, mk, h, w, framestride,
                                            boundary_head=bhead, flip=flip, return_raw=True)
                for i in range(video_length):
                    src = video_length - 1 - i if reverse else i
                    if sf[src] is not None and cf[src] > 0:
                        pass_pred = sf[src] / cf[src]
                        if all_sum[i] is None:
                            all_sum[i] = pass_pred
                        else:
                            all_sum[i] += pass_pred
                        all_passes[i] += 1
        comp_frames = []
        for i in range(video_length):
            if all_sum[i] is not None and all_passes[i] > 0:
                avg_pred = all_sum[i] / all_passes[i]
                mask_bin = (np.array(rmasks[i].convert('L')) > 0).astype(np.float32)[:, :, None]
                bsc_np   = np.array(rframes[i]).astype(np.float32)
                merged   = (avg_pred * mask_bin + bsc_np * (1.0 - mask_bin)).clip(0, 255)
                comp_frames.append(merged.astype(np.float32))
            else:
                comp_frames.append(None)
    else:
        print('  Forward pass...')
        sum_fwd, cnt_fwd = run_inference_pass(model, device, rframes, rmasks, h, w, framestride,
                                              boundary_head=bhead, return_raw=True)
        print('  Reverse-time TTA pass...')
        sum_bwd, cnt_bwd = run_inference_pass(
            model, device, list(reversed(rframes)), list(reversed(rmasks)), h, w, framestride,
            boundary_head=bhead, return_raw=True)
        # Merge raw fwd+bwd predictions first, then composite once (cleaner for refinement)
        raw_merged = _tta_merge(sum_fwd, cnt_fwd, sum_bwd, cnt_bwd, video_length,
                                alpha_net=alpha_net, device=device)
        comp_frames = []
        for i, pred in enumerate(raw_merged):
            if pred is None:
                comp_frames.append(None)
            else:
                mask_bin = (np.array(rmasks[i].convert('L')) > 0).astype(np.float32)[:, :, None]
                bsc_np   = np.array(rframes[i]).astype(np.float32)
                comp_frames.append(
                    (pred * mask_bin + bsc_np * (1.0 - mask_bin)).clip(0, 255).astype(np.float32))
    return comp_frames


def process_one_video(model, device, video_path, mask_path, size, framestride):
    rframes = read_frame_from_videos(video_path)
    rframes, size = resize_frames(rframes, size)
    h, w = size[1], size[0]
    rmasks = read_mask(mask_path, size)

    bhead      = process_one_video.boundary_head
    tta8       = process_one_video.tta8
    alpha_net  = process_one_video.alpha_net
    n_passes   = process_one_video.refine_passes

    # Pre-compute per-frame mask ratios for refinement gating
    mask_ratios = [(np.array(m.convert('L')) > 0).mean() for m in rmasks]

    # Pass 1 uses original BSC frames; subsequent passes feed the float32 composite back.
    iter_frames = rframes
    comp_frames = None
    comp_pass1  = None   # saved for conservative pass1/pass2 blend
    for p in range(n_passes):
        if p > 0:
            print(f'  Self-refinement pass {p + 1}/{n_passes}...')
            # Only replace frames where the mask is large enough to benefit
            iter_frames = [
                Image.fromarray(np.clip(f, 0, 255).astype(np.uint8))
                if (f is not None and mask_ratios[i] > 0.05) else rframes[i]
                for i, f in enumerate(comp_frames)
            ]
        comp_frames = _infer_to_comp(model, device, iter_frames, rmasks,
                                     h, w, framestride, bhead, tta8, alpha_net)
        if p == 0:
            comp_pass1 = [None if f is None else f.copy() for f in comp_frames]

    # Conservative blend: 65% pass-N + 35% pass-1 inside mask (reduces hallucination)
    if n_passes >= 2 and comp_pass1 is not None:
        for i in range(len(comp_frames)):
            if comp_frames[i] is None or comp_pass1[i] is None:
                continue
            m = (np.array(rmasks[i].convert('L')) > 0).astype(np.float32)[:, :, None]
            comp_frames[i] = (0.65 * comp_frames[i] + 0.35 * comp_pass1[i]) * m \
                             + comp_frames[i] * (1.0 - m)
        # Sanity: report mean change vs pass-1 for frame 0
        i0 = next((i for i, f in enumerate(comp_frames) if f is not None), None)
        if i0 is not None and comp_pass1[i0] is not None:
            diff = np.mean(np.abs(comp_frames[i0].astype(np.float32)
                                  - comp_pass1[i0].astype(np.float32)))
            print(f'  Refinement mean abs diff (frame {i0}): {diff:.3f}')

    # Apply residual refiner (post-TTA, per-frame, only inside mask)
    refiner = process_one_video.refiner
    if refiner is not None:
        dev = process_one_video.device
        for i, f_frame in enumerate(comp_frames):
            if f_frame is None:
                continue
            bsc_t  = torch.from_numpy(
                np.array(rframes[i]).astype(np.float32) / 255.0
            ).permute(2, 0, 1).unsqueeze(0).to(dev)
            mask_bin = (np.array(rmasks[i].convert('L')) > 0).astype(np.float32)
            mask_t = torch.from_numpy(mask_bin).unsqueeze(0).unsqueeze(0).to(dev)
            pred_t = torch.from_numpy(
                f_frame.astype(np.float32) / 255.0
            ).permute(2, 0, 1).unsqueeze(0).to(dev)
            with torch.no_grad():
                refined = refiner(bsc_t, mask_t, pred_t)
            # Keep float32 to avoid cascaded quantization errors
            comp_frames[i] = (refined[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255)

    # Final data consistency: re-impose original BSC outside mask
    # Handles any sub-pixel bleed accumulated through float32 passes / blending / refiner
    for i, f in enumerate(comp_frames):
        if f is None:
            continue
        bsc_np = np.array(rframes[i]).astype(np.float32)
        m = (np.array(rmasks[i].convert('L')) > 0).astype(np.float32)[:, :, None]
        comp_frames[i] = (f * m + bsc_np * (1.0 - m)).clip(0, 255)

    out_root = getattr(process_one_video, "out_dir", "./outputs")
    save_base_name = os.path.basename(video_path.rstrip('/'))
    save_path   = os.path.join(out_root, save_base_name)
    imwrite_path = os.path.join(save_path, 'frame_seq')
    os.makedirs(imwrite_path, exist_ok=True)
    saved = 0
    for i, f in enumerate(comp_frames):
        if f is None:
            continue
        Image.fromarray(np.clip(f, 0, 255).astype(np.uint8)).save(
            os.path.join(imwrite_path, f'{i:05d}.png'))
        saved += 1
    gif_imgs = [Image.fromarray(np.clip(f, 0, 255).astype(np.uint8))
                for f in comp_frames if f is not None]
    gif_imgs[0].save(os.path.join(save_path, 'GIF_result.gif'),
                     save_all=True, append_images=gif_imgs[1:], duration=40, loop=0)
    print(f'  Saved {saved} frames to: {save_path}')


def main_worker():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model ONCE
    net = importlib.import_module('model.' + args.model)
    model = net.InpaintGenerator(init_weights=False).to(device)
    data = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(data)
    print(f'Loading model from: {args.ckpt}')
    model.eval()

    # Optionally load SAM2 LoRA + SAMFuser weights
    if args.lora_ckpt:
        from model.modules.lora_utils import inject_lora_sam2 as inject_lora
        lora_data = torch.load(args.lora_ckpt, map_location=device)
        rank  = lora_data.get('rank', 8)
        alpha = lora_data.get('alpha', 16)
        inject_lora(model.sam2_encoder.encoder, r=rank, alpha=alpha)
        missing, unexpected = model.load_state_dict(lora_data['lora'], strict=False)
        model.sam_fuser.load_state_dict(lora_data['sam_fuser'])
        model.eval()
        print(f'Loaded LoRA weights from: {args.lora_ckpt}  (rank={rank}, alpha={alpha})')
        print(f'  LoRA keys loaded: {len(lora_data["lora"])}  missing: {len(missing)}  unexpected: {len(unexpected)}')

    # Optionally load boundary refinement head
    bhead = None
    if args.boundary_ckpt:
        from model.boundary_head import BoundaryRefinementHead
        bhead = BoundaryRefinementHead(channels=64).to(device)
        bhead.load_state_dict(torch.load(args.boundary_ckpt, map_location=device))
        bhead.eval()
        print(f'Loaded boundary head from: {args.boundary_ckpt}')
    process_one_video.boundary_head = bhead

    # Optionally load spatial alpha net for per-pixel TTA merge
    alpha_net = None
    if args.alpha_ckpt:
        from model.spatial_alpha import SpatialAlphaNet
        alpha_net = SpatialAlphaNet(mid_ch=32).to(device)
        alpha_data = torch.load(args.alpha_ckpt, map_location=device)
        alpha_net.load_state_dict(alpha_data['model'] if 'model' in alpha_data else alpha_data)
        alpha_net.eval()
        print(f'Loaded spatial alpha net from: {args.alpha_ckpt}')
    process_one_video.alpha_net = alpha_net
    process_one_video.tta8      = args.tta8

    # Optionally load transformer LoRA (WindowAttention qkv+proj)
    if args.lora_transformer_ckpt:
        from model.modules.lora_utils import inject_lora_transformer
        tr_data = torch.load(args.lora_transformer_ckpt, map_location=device)
        inject_lora_transformer(model,
                                r=tr_data.get('rank', 8),
                                alpha=tr_data.get('alpha', 16),
                                include_proj=tr_data.get('include_proj', False))
        model.to(device)  # move newly created lora_A/lora_B (CPU) to GPU
        model.load_state_dict(tr_data['lora'], strict=False)
        model.eval()
        print(f'Loaded transformer LoRA from: {args.lora_transformer_ckpt}  '
              f'(rank={tr_data.get("rank",8)}, include_proj={tr_data.get("include_proj",False)})')

    # Optionally load residual refiner
    refiner = None
    if args.refiner_ckpt:
        from model.refiner import ResidualRefiner
        refiner_data = torch.load(args.refiner_ckpt, map_location=device)
        in_ch = 10 if args.refiner_use_lap else 7
        if isinstance(refiner_data, dict):
            if 'in_ch' in refiner_data:
                in_ch = int(refiner_data['in_ch'])
            elif 'args' in refiner_data and isinstance(refiner_data['args'], dict):
                if bool(refiner_data['args'].get('use_lap', False)):
                    in_ch = 10
        refiner = ResidualRefiner(in_ch=in_ch, base_ch=32).to(device)
        refiner.load_state_dict(refiner_data['model'] if 'model' in refiner_data else refiner_data)
        refiner.eval()
        print(f'Loaded residual refiner from: {args.refiner_ckpt}')
    process_one_video.refiner        = refiner
    process_one_video.device         = device
    process_one_video.refine_passes  = args.refine_passes
    process_one_video.out_dir        = args.out_dir

    size = (args.width, args.height) if (args.width is not None and args.height is not None) else None

    if args.video_dir:
        # Batch mode: loop over all video subdirs, model stays loaded
        video_names = sorted([
            d for d in os.listdir(args.video_dir)
            if os.path.isdir(os.path.join(args.video_dir, d))
        ])
        total = len(video_names)
        print(f'Batch mode: {total} videos in {args.video_dir}\n')
        for current, video_name in enumerate(video_names, 1):
            video_path = os.path.join(args.video_dir, video_name)
            mask_path  = os.path.join(args.mask_dir, video_name)
            print(f'[{current}/{total}] {video_name}')
            process_one_video(model, device, video_path, mask_path, size, args.framestride)
            print(f'[{current}/{total}] Done: {video_name}\n')
        print(f'All {total} videos processed.')
    else:
        # Single video mode
        print(f'Loading video from: {args.video}')
        process_one_video(model, device, args.video, args.dac_mask, size, args.framestride)


if __name__ == '__main__':
    main_worker()
