# -*- coding: utf-8 -*-
"""
Per-video Test-Time Optimization (TTO) using transformer LoRA params.
Self-supervised temporal warp + boundary ring losses — no GT needed.

Usage:
  python tto_test.py \\
      --ckpt checkpoints/ema_cons_run1/gen_best.pth \\
      --lora_transformer_ckpt checkpoints/lora_transformer_on_sam2_best.pth \\
      --boundary_ckpt checkpoints/boundary_head_best.pth \\
      --refiner_ckpt  checkpoints/refiner_best.pth \\
      --video_dir /media/ajeet/data/MINI_BTP/data/validation/bsc_imgs \\
      --mask_dir  /media/ajeet/data/MINI_BTP/data/validation/masks \\
      --width 432 --height 240 \\
      --tta_steps 80
"""

import copy, os, argparse, random, importlib
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torch.amp import autocast, GradScaler

from core.utils import to_tensors
from model.modules.flow_comp import flow_warp

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt",                  required=True)
parser.add_argument("--model",                 default="b2scvr_hq")
parser.add_argument("--video_dir",             required=True)
parser.add_argument("--mask_dir",              required=True)
parser.add_argument("--width",                 type=int, required=True)
parser.add_argument("--height",                type=int, required=True)
parser.add_argument("--lora_ckpt",             default=None,
                    help="SAM2 LoRA checkpoint (frozen during TTO, improves features)")
parser.add_argument("--lora_transformer_ckpt", default=None)
parser.add_argument("--boundary_ckpt",         default=None)
parser.add_argument("--refiner_ckpt",          default=None)
parser.add_argument("--refiner_use_lap",      action="store_true",
                    help="Force 10ch refiner input (pred laplacian features). "
                         "If not set, inferred from checkpoint when possible.")
parser.add_argument("--step",                  type=int, default=10,
                    help="Ref frame sampling stride (same as test.py --step)")
parser.add_argument("--neighbor_stride",       type=int, default=5)
parser.add_argument("--framestride",           type=int, default=30)
parser.add_argument("--tta_steps",             type=int, default=60,
                    help="Gradient steps per video (0 = skip TTO, run inference only)")
parser.add_argument("--tta_lr",                type=float, default=2e-5)
parser.add_argument("--w_temp",                type=float, default=0.6,
                    help="Weight for temporal warp consistency loss")
parser.add_argument("--w_ring",                type=float, default=0.03,
                    help="Weight for boundary ring loss")
parser.add_argument("--w_reg",                 type=float, default=5e-4,
                    help="Weight for L2 regulariser on LoRA params")
parser.add_argument("--min_mask_ratio",        type=float, default=0.08,
                    help="Skip TTO for videos whose avg mask ratio is below this")
args = parser.parse_args()

ref_length      = args.step
neighbor_stride = args.neighbor_stride


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def dilate_t(mask, k=7):
    return F.max_pool2d(mask, kernel_size=k, stride=1, padding=k // 2)


def erode_t(mask, k=5):
    return 1.0 - dilate_t(1.0 - mask, k=k)


def charbonnier(x, eps=1e-3):
    return torch.sqrt(x * x + eps * eps).mean()


def get_ref_index(f, neighbor_ids, length):
    ref_index = []
    for i in range(0, length, ref_length):
        if i not in neighbor_ids:
            ref_index.append(i)
    return ref_index


def read_frames(path, size):
    frames = []
    for name in sorted(os.listdir(path)):
        img = cv2.imread(os.path.join(path, name))
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if size is not None:
            img = img.resize(size)
        frames.append(img)
    return frames


def read_masks(path, size):
    masks = []
    for name in sorted(os.listdir(path)):
        m = Image.open(os.path.join(path, name)).convert('L')
        if size is not None:
            m = m.resize(size, Image.NEAREST)
        m_arr = np.array(m)
        m_bin = (m_arr > 0).astype(np.uint8)
        m_bin = cv2.dilate(m_bin,
                           cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
                           iterations=4)
        masks.append(Image.fromarray(m_bin * 255))
    return masks


# ---------------------------------------------------------------------------
# TTO core
# ---------------------------------------------------------------------------
def tto_one_video(model, device, frames_pil, masks_pil, h, w):
    """Optimize LoRA params for one video using self-supervised losses."""
    T = len(frames_pil)

    # Tensors in [0,1] for SpyNet + data-consistency
    frames_01 = torch.stack([
        torch.from_numpy(np.array(f).astype(np.float32) / 255.0).permute(2, 0, 1)
        for f in frames_pil
    ]).to(device)  # (T, 3, H, W)

    # Binary masks  (T, 1, H, W)
    masks_bin = torch.stack([
        torch.from_numpy((np.array(m.convert('L')) > 0).astype(np.float32))
        for m in masks_pil
    ]).unsqueeze(1).to(device)

    # Precompute flows once with frozen SpyNet: flow[t] = flow from t-1 to t
    spynet = getattr(model, 'update_spynet', None) or getattr(model, 'spynet', None)
    assert spynet is not None, 'Could not find SpyNet module (update_spynet/spynet)'
    spynet.eval()
    for p in spynet.parameters():
        p.requires_grad_(False)
    # Helper: fill masked region with cv2 inpaint so SpyNet sees no black holes
    def _inpaint_frame(frame_t, mask_t):
        """frame_t: (1,3,H,W) [0,1] tensor; mask_t: (1,1,H,W) binary tensor."""
        f_np = (frame_t[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        m_np = (mask_t[0, 0].cpu().numpy() * 255).astype(np.uint8)
        filled = cv2.inpaint(f_np, m_np, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        return torch.from_numpy(filled).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0

    flows     = [None]
    sign_logged = False
    with torch.no_grad():
        for t in range(1, T):
            f1c  = _inpaint_frame(frames_01[t - 1:t], masks_bin[t - 1:t])
            f2c  = _inpaint_frame(frames_01[t:t + 1], masks_bin[t:t + 1])
            flow = spynet(f1c, f2c)           # (1, 2, H, W)

            # Auto-pick flow sign: whichever direction warps f1c→f2c better on clean pixels
            m2c   = 1.0 - masks_bin[t:t + 1]
            w_pos = flow_warp(f1c,  flow.permute(0, 2, 3, 1))
            w_neg = flow_warp(f1c, (-flow).permute(0, 2, 3, 1))
            e_pos = (m2c * (w_pos - f2c).abs()).mean().item()
            e_neg = (m2c * (w_neg - f2c).abs()).mean().item()
            if not sign_logged:
                print(f'  [flow sign] t=1: e_pos={e_pos:.5f}  e_neg={e_neg:.5f}  '
                      f'→ using {"−flow" if e_neg < e_pos else "+flow"}')
                sign_logged = True
            if e_neg < e_pos:
                flow = -flow

            flows.append(flow.detach())

    # Freeze entire backbone; only transformer LoRA params get gradients → prevents OOM
    # SAM2 LoRA (sam2_encoder) stays frozen — it improves features but is not updated
    model.requires_grad_(False)
    for n, p in model.named_parameters():
        if 'lora_' in n and 'sam2_encoder' not in n:
            p.requires_grad_(True)
    torch.cuda.empty_cache()

    # Collect only transformer LoRA params for the optimizer
    lora_params = [p for n, p in model.named_parameters()
                   if 'lora_' in n and 'sam2_encoder' not in n and p.requires_grad]
    if not lora_params:
        print('  [TTO] No LoRA params found — skipping')
        return

    optimizer = torch.optim.Adam(lora_params, lr=args.tta_lr, weight_decay=0.0)
    scaler    = GradScaler("cuda")
    model.train()

    # Model-format frames in [-1, 1]
    frames_model = frames_01 * 2.0 - 1.0  # (T, 3, H, W)

    print(f'  [TTO] trainable params: {sum(p.numel() for p in lora_params):,}')

    best_loss = 1e9
    patience  = 0

    for _ in tqdm(range(args.tta_steps), desc='  TTO', leave=False):
        # Sample a fixed 3-frame window to keep attention size small during TTO
        t_c  = random.randint(1, T - 2)
        nb   = [t_c - 1, t_c, t_c + 1]
        refs = get_ref_index(t_c, nb, T)[:2]  # cap ref frames to save GPU memory
        all_ids = nb + refs
        N_nb    = len(nb)

        # Skip windows with very little corruption — they add noise, not signal
        if masks_bin[nb].mean().item() < 0.08:
            continue

        imgs_t  = frames_model[all_ids].unsqueeze(0)   # (1, N+R, 3, H, W)
        masks_t = masks_bin[all_ids].unsqueeze(0)      # (1, N+R, 1, H, W)

        optimizer.zero_grad(set_to_none=True)
        with autocast('cuda', dtype=torch.float16):
            pred, _ = model(imgs_t, imgs_t, masks_t, N_nb)
            # pred: (N+R, 3, H_pad, W_pad) in [-1,1]
            pred_nb = (pred[:N_nb, :, :h, :w] + 1.0) / 2.0  # (N, 3, H, W) [0,1]

            loss = torch.tensor(0.0, device=device)

            for i, gidx in enumerate(nb):
                It = frames_01[gidx:gidx + 1]   # (1, 3, H, W)
                Mt = masks_bin[gidx:gidx + 1]   # (1, 1, H, W)
                Rt = pred_nb[i:i + 1]

                # Hard data consistency: don't change clean pixels
                Rt = Rt * Mt + It * (1.0 - Mt)

                # Boundary ring: only anchor to clean (unmasked) pixels near edge
                ring       = (dilate_t(Mt, 7) - erode_t(Mt, 5)).clamp(0.0, 1.0)
                ring_clean = ring * (1.0 - Mt)
                denom      = ring_clean.sum() + 1e-6
                if ring_clean.sum() > 0:
                    loss = loss + args.w_ring * (ring_clean * (Rt - It).abs()).sum() / denom

                # Temporal warp: only on heavily-corrupted frames, boundary band only
                mask_ratio = Mt.mean().item()
                prev_gidx  = gidx - 1
                if (mask_ratio >= 0.08 and gidx > 0
                        and flows[gidx] is not None and prev_gidx in nb):
                    prev_i  = nb.index(prev_gidx)
                    Mp_prev = masks_bin[prev_gidx:prev_gidx + 1]
                    if Mp_prev.mean().item() >= 0.08:
                        Rp      = pred_nb[prev_i:prev_i + 1]
                        Ip_prev = frames_01[prev_gidx:prev_gidx + 1]
                        Rp      = Rp * Mp_prev + Ip_prev * (1.0 - Mp_prev)
                        # Boundary band, clean pixels only — no masked-pixel flow supervision
                        band       = (dilate_t(Mt, 9) - dilate_t(Mt, 3)).clamp(0.0, 1.0)
                        band_clean = band * (1.0 - Mt)
                        Rp_warp    = flow_warp(Rp, flows[gidx].permute(0, 2, 3, 1))
                        # Make warped prev clean-consistent at current t (no garbage into band)
                        Rp_warp    = Rp_warp * (1.0 - Mt) + It * Mt
                        loss       = loss + args.w_temp * charbonnier(band_clean * (Rt - Rp_warp))

                # Outer-mask stability: enforce clean temporal consistency far from mask
                # (low weight, no mask-ratio gate — stabilizes clean regions)
                if gidx > 0 and flows[gidx] is not None and prev_gidx in nb:
                    prev_i2   = nb.index(prev_gidx)
                    Rp2       = pred_nb[prev_i2:prev_i2 + 1]
                    Mp2       = masks_bin[prev_gidx:prev_gidx + 1]
                    Ip2       = frames_01[prev_gidx:prev_gidx + 1]
                    Rp2       = Rp2 * Mp2 + Ip2 * (1.0 - Mp2)
                    outer     = 1.0 - dilate_t(Mt, 11)
                    Rp2_warp  = flow_warp(Rp2, flows[gidx].permute(0, 2, 3, 1))
                    Rp2_warp  = Rp2_warp * (1.0 - Mt) + It * Mt
                    loss      = loss + 0.1 * charbonnier(outer * (Rt - Rp2_warp))

            # L2 regulariser — prevents LoRA from drifting too far
            reg  = sum((p * p).mean() for p in lora_params)
            loss = loss + args.w_reg * reg

        if loss.item() == 0.0:
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        scaler.step(optimizer)
        scaler.update()

        # Early stopping: halt if loss stops improving
        cur = loss.item()
        if cur < best_loss - 1e-6:
            best_loss = cur
            patience  = 0
        else:
            patience += 1
            if patience >= 8:
                break

    model.eval()
    # Keep backbone frozen; only transformer LoRA stays enabled for next video's TTO
    model.requires_grad_(False)
    for n, p in model.named_parameters():
        if 'lora_' in n and 'sam2_encoder' not in n:
            p.requires_grad_(True)


# ---------------------------------------------------------------------------
# Inference (mirrors test.py run_inference_pass, fwd+bwd merge)
# ---------------------------------------------------------------------------
def run_pass(model, device, rframes, rmasks, h, w, bhead):
    T            = len(rframes)
    sum_frames   = [None] * T
    count_frames = [0]    * T

    x_frames = [rframes[i:i + args.framestride] for i in range(0, T, args.framestride)]
    x_masks  = [rmasks[i:i + args.framestride]  for i in range(0, T, args.framestride)]

    for itern in range(len(x_frames)):
        stride_len = len(x_frames[itern])
        imgs      = to_tensors()(x_frames[itern]).unsqueeze(0) * 2 - 1
        frames_np = [np.array(f).astype(np.uint8) for f in x_frames[itern]]
        bin_masks = [np.expand_dims((np.array(m) != 0).astype(np.uint8), 2)
                     for m in x_masks[itern]]
        masks     = to_tensors()(x_masks[itern]).unsqueeze(0)
        imgs, masks = imgs.to(device), masks.to(device)

        for f in tqdm(range(0, stride_len, neighbor_stride), leave=False):
            nb_ids  = [i for i in range(max(0, f - neighbor_stride),
                                        min(stride_len, f + neighbor_stride + 1))]
            ref_ids = get_ref_index(f, nb_ids, stride_len)
            si      = imgs[:1, nb_ids + ref_ids]
            sm      = masks[:1, nb_ids + ref_ids]
            with torch.no_grad():
                pred, _ = model(si, si, sm, len(nb_ids))
                pred     = pred[:, :, :h, :w]
                if bhead is not None:
                    n_nb     = len(nb_ids)
                    restored = pred[:n_nb]
                    original = si[0, :n_nb, :, :h, :w]
                    bmask    = sm[0, :n_nb, :, :h, :w]
                    with torch.no_grad():
                        pred[:n_nb], _ = bhead(restored, original, bmask)
                pred = (pred + 1) / 2
                pred_np = pred.cpu().permute(0, 2, 3, 1).numpy() * 255

            for i, idx in enumerate(nb_ids):
                gidx  = itern * args.framestride + idx
                p     = pred_np[i].astype(np.float32)
                m2d   = bin_masks[idx][:, :, 0:1].astype(np.float32)
                val   = p * m2d + frames_np[idx].astype(np.float32) * (1.0 - m2d)
                if sum_frames[gidx] is None:
                    sum_frames[gidx] = val
                else:
                    sum_frames[gidx] += val
                count_frames[gidx] += 1

    return sum_frames, count_frames


# ---------------------------------------------------------------------------
# Per-video pipeline
# ---------------------------------------------------------------------------
def process_video(model, device, video_path, mask_path, lora_base_state, bhead, refiner):
    size    = (args.width, args.height)
    rframes = read_frames(video_path, size)
    rmasks  = read_masks(mask_path, size)
    h, w    = args.height, args.width
    T       = len(rframes)

    # TTO step
    avg_mask = np.mean([(np.array(m.convert('L')) > 0).mean() for m in rmasks])
    if args.tta_steps > 0 and avg_mask >= args.min_mask_ratio:
        print(f'  TTO ({args.tta_steps} steps, avg_mask={avg_mask:.4f})')
        tto_one_video(model, device, rframes, rmasks, h, w)
    else:
        print(f'  Skipping TTO (tta_steps={args.tta_steps}, avg_mask={avg_mask:.4f})')

    # Final inference with adapted LoRA
    print('  Forward pass...')
    sf, cf = run_pass(model, device, rframes, rmasks, h, w, bhead)
    print('  Reverse pass...')
    sb, cb = run_pass(model, device, list(reversed(rframes)),
                      list(reversed(rmasks)), h, w, bhead)

    # Merge fwd + bwd + data-consistency
    comp_frames = []
    for i in range(T):
        j   = T - 1 - i
        fwd = (sf[i] / cf[i]) if (sf[i] is not None and cf[i] > 0) else None
        bwd = (sb[j] / cb[j]) if (sb[j] is not None and cb[j] > 0) else None
        if fwd is not None and bwd is not None:
            pred = 0.5 * fwd + 0.5 * bwd
        elif fwd is not None:
            pred = fwd
        elif bwd is not None:
            pred = bwd
        else:
            comp_frames.append(None)
            continue
        m2d  = (np.array(rmasks[i].convert('L')) > 0).astype(np.float32)[:, :, None]
        bsc  = np.array(rframes[i]).astype(np.float32)
        comp_frames.append((pred * m2d + bsc * (1.0 - m2d)).clip(0, 255).astype(np.float32))

    # Residual refiner
    if refiner is not None:
        dev = device
        for i, f in enumerate(comp_frames):
            if f is None:
                continue
            bsc_t  = torch.from_numpy(np.array(rframes[i]).astype(np.float32) / 255.0
                     ).permute(2, 0, 1).unsqueeze(0).to(dev)
            mask_t = torch.from_numpy(
                         (np.array(rmasks[i].convert('L')) > 0).astype(np.float32)
                     ).unsqueeze(0).unsqueeze(0).to(dev)
            pred_t = torch.from_numpy(f / 255.0).permute(2, 0, 1).unsqueeze(0).to(dev)
            with torch.no_grad():
                refined = refiner(bsc_t, mask_t, pred_t)
            comp_frames[i] = (refined[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255)

    # Final data-consistency
    for i, f in enumerate(comp_frames):
        if f is None:
            continue
        bsc = np.array(rframes[i]).astype(np.float32)
        m2d = (np.array(rmasks[i].convert('L')) > 0).astype(np.float32)[:, :, None]
        comp_frames[i] = (f * m2d + bsc * (1.0 - m2d)).clip(0, 255)

    # Reset LoRA to base weights for next video (direct copy — bulletproof)
    if lora_base_state:
        with torch.no_grad():
            for n, p in model.named_parameters():
                if 'lora_' in n and n in lora_base_state:
                    p.copy_(lora_base_state[n])
    model.eval()

    # Save
    save_name = os.path.basename(video_path.rstrip('/'))
    out_dir   = os.path.join('./outputs', save_name, 'frame_seq')
    os.makedirs(out_dir, exist_ok=True)
    for i, f in enumerate(comp_frames):
        if f is None:
            continue
        Image.fromarray(np.clip(f, 0, 255).astype(np.uint8)).save(
            os.path.join(out_dir, f'{i:05d}.png'))
    print(f'  Saved {sum(1 for f in comp_frames if f is not None)} frames → {out_dir}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net   = importlib.import_module('model.' + args.model)
    model = net.InpaintGenerator(init_weights=False).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    print(f'Loaded model: {args.ckpt}')

    # SAM2 LoRA (frozen — kept as-is, improves feature extraction, not updated during TTO)
    if args.lora_ckpt:
        from model.modules.lora_utils import inject_lora_sam2
        lora_data = torch.load(args.lora_ckpt, map_location=device)
        inject_lora_sam2(model.sam2_encoder.encoder,
                         r=lora_data.get('rank', 8),
                         alpha=lora_data.get('alpha', 16))
        model.load_state_dict(lora_data['lora'], strict=False)
        model.sam_fuser.load_state_dict(lora_data['sam_fuser'])
        model.eval()
        print(f'Loaded SAM2 LoRA (frozen): {args.lora_ckpt}')

    # Transformer LoRA
    lora_base_state = {}
    if args.lora_transformer_ckpt:
        from model.modules.lora_utils import inject_lora_transformer
        tr_data = torch.load(args.lora_transformer_ckpt, map_location=device)
        inject_lora_transformer(model,
                                r=tr_data.get('rank', 8),
                                alpha=tr_data.get('alpha', 16),
                                include_proj=tr_data.get('include_proj', False))
        model.to(device)
        model.load_state_dict(tr_data['lora'], strict=False)
        model.eval()
        # Snapshot only transformer LoRA weights for per-video reset (SAM2 LoRA is frozen)
        lora_base_state = {n: p.detach().clone()
                           for n, p in model.named_parameters()
                           if 'lora_' in n and 'sam2_encoder' not in n}
        n_lora_trainable = sum(p.numel() for n, p in model.named_parameters()
                              if 'lora_' in n and 'sam2_encoder' not in n and p.requires_grad)
        print(f'Loaded transformer LoRA: {args.lora_transformer_ckpt}  '
              f'({len(lora_base_state)} tensors, {n_lora_trainable:,} trainable params)')
    else:
        print('Warning: no --lora_transformer_ckpt; TTO will have nothing to optimize')

    # Boundary head
    bhead = None
    if args.boundary_ckpt:
        from model.boundary_head import BoundaryRefinementHead
        bhead = BoundaryRefinementHead(channels=64).to(device)
        bhead.load_state_dict(torch.load(args.boundary_ckpt, map_location=device))
        bhead.eval()
        print(f'Loaded boundary head: {args.boundary_ckpt}')

    # Residual refiner
    refiner = None
    if args.refiner_ckpt:
        from model.refiner import ResidualRefiner
        rd = torch.load(args.refiner_ckpt, map_location=device)
        in_ch = 10 if args.refiner_use_lap else 7
        if isinstance(rd, dict):
            if 'in_ch' in rd:
                in_ch = int(rd['in_ch'])
            elif 'args' in rd and isinstance(rd['args'], dict):
                if bool(rd['args'].get('use_lap', False)):
                    in_ch = 10
        refiner = ResidualRefiner(in_ch=in_ch, base_ch=32).to(device)
        refiner.load_state_dict(rd['model'] if 'model' in rd else rd)
        refiner.eval()
        print(f'Loaded refiner: {args.refiner_ckpt}')

    # Process all videos
    video_names = sorted([d for d in os.listdir(args.video_dir)
                          if os.path.isdir(os.path.join(args.video_dir, d))])
    print(f'\nBatch mode: {len(video_names)} videos\n')

    for idx, vname in enumerate(video_names, 1):
        vpath = os.path.join(args.video_dir, vname)
        mpath = os.path.join(args.mask_dir,  vname)
        print(f'[{idx}/{len(video_names)}] {vname}')
        process_video(model, device, vpath, mpath, lora_base_state, bhead, refiner)
        print()


if __name__ == '__main__':
    main()
