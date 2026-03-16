"""
LoRA finetune the frozen SAM2 Hiera-T trunk + SAMFuser for video recovery.

Fixes vs original:
  1. infer_pass() matches test.py chunking (framestride-based, not whole-video)
  2. Worst-video baseline scan done BEFORE LoRA injection
  3. K windows per clip (default 3) — stable gradient direction
  4. Loss on composited frame (pred*mask + bsc*(1-mask)), not raw prediction
  5. Separate param groups: LoRA lr=1e-4 wd=0, SAMFuser lr=--lr wd=1e-5
  6. LoRA key safety prints after inject + after resume load

Usage
-----
  python train_lora_sam2.py \
      --config  config/lora_sam2.json \
      --ckpt    checkpoints/main_best.pth \
      --save    checkpoints/lora_sam2.pth \
      --iterations 10000 \
      --lora_lr 1e-4 --lr 2e-5 \
      --lora_rank 8 --lora_alpha 16 \
      --eval_videos_dir worst10/bsc_imgs \
      --val_bsc_dir  /path/to/val/bsc_imgs \
      --val_gt_dir   /path/to/val/gt_imgs \
      --val_mask_dir /path/to/val/masks
"""

import argparse
import importlib
import json
import math
import os
import random

import cv2
import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from PIL import Image
from tqdm import tqdm

from core.utils import to_tensors
from model.modules.lora_utils import inject_lora_sam2 as inject_lora


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
p = argparse.ArgumentParser()
p.add_argument('--config',           required=True, help='JSON config (for data paths)')
p.add_argument('--ckpt',             required=True, help='Frozen main model checkpoint')
p.add_argument('--save',             default='checkpoints/lora_sam2.pth')
p.add_argument('--iterations',       type=int,   default=10000)
p.add_argument('--lora_lr',          type=float, default=1e-4,  help='LR for LoRA adapters')
p.add_argument('--lr',               type=float, default=2e-5,  help='LR for SAMFuser')
p.add_argument('--lora_rank',        type=int,   default=8)
p.add_argument('--lora_alpha',       type=int,   default=16)
p.add_argument('--windows_per_clip', type=int,   default=1,
               help='Optimizer steps per clip (each window is an independent step)')
p.add_argument('--log_freq',         type=int,   default=100)
p.add_argument('--eval_freq',        type=int,   default=1000)
p.add_argument('--scheduler',        type=str,   default='cosine',
               choices=['cosine', 'multistep'],
               help='LR scheduler type')
p.add_argument('--lr_drop_milestones', type=str, default='12000,18000',
               help='Comma-separated iteration milestones for multistep scheduler')
p.add_argument('--lr_drop_gamma',    type=float, default=0.3,
               help='Gamma for multistep LR drops')
p.add_argument('--model',            type=str,   default='b2scvr_hq')
p.add_argument('--step',             type=int,   default=10)
p.add_argument('--num_ref',          type=int,   default=-1)
p.add_argument('--neighbor_stride',  type=int,   default=5)
p.add_argument('--framestride',      type=int,   default=30)
p.add_argument('--hard_window_min_mask', type=float, default=0.06,
               help='Minimum frame mask ratio to be considered for hard-window sampling')
p.add_argument('--hard_window_top_frac', type=float, default=0.35,
               help='Sample from top fraction (by mask ratio) among eligible frames')
p.add_argument('--hard_window_require', action='store_true',
               help='If set, skip the clip when no frame meets hard-window threshold')
p.add_argument('--resume',           type=str,   default=None)
p.add_argument('--val_bsc_dir',      default=None)
p.add_argument('--val_gt_dir',       default=None)
p.add_argument('--val_mask_dir',     default=None)
p.add_argument('--eval_videos_dir',  default=None,
               help='Folder whose sub-dirs are eval video names (e.g. worst10/bsc_imgs)')
p.add_argument('--eval_n',           type=int, default=10)
args = p.parse_args()

ref_length      = args.step
num_ref         = args.num_ref
neighbor_stride = args.neighbor_stride


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
with open(args.config) as f:
    config = json.load(f)

train_cfg = config['train_data_loader']
GT_ROOT   = train_cfg['gt_root']
BSC_ROOT  = train_cfg['bsc_root']
MASK_ROOT = train_cfg['mask_root']
W, H      = train_cfg['w'], train_cfg['h']


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def get_ref_index(f, neighbor_ids, length):
    ref_index = []
    if num_ref == -1:
        for i in range(0, length, ref_length):
            if i not in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, f - ref_length * (num_ref // 2))
        end_idx   = min(length, f + ref_length * (num_ref // 2))
        for i in range(start_idx, end_idx + 1, ref_length):
            if i not in neighbor_ids:
                if len(ref_index) > num_ref:
                    break
                ref_index.append(i)
    return ref_index


def read_frames(video_dir, size):
    imgs = []
    for name in sorted(os.listdir(video_dir)):
        img = cv2.imread(os.path.join(video_dir, name))
        if img is None:
            continue
        imgs.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).resize(size))
    return imgs


def read_masks(mask_dir, size):
    masks = []
    for name in sorted(os.listdir(mask_dir)):
        m = Image.open(os.path.join(mask_dir, name)).resize(size, Image.NEAREST)
        m = np.array(m.convert('L'))
        m = (m > 0).astype(np.uint8)
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), iterations=4)
        masks.append(Image.fromarray(m * 255))
    return masks


# ---------------------------------------------------------------------------
# Validation helpers — infer_pass matches test.py chunking exactly
# ---------------------------------------------------------------------------
def infer_pass(model, device, frames_all, masks_all):
    """One direction inference pass; returns (sum_frames, cnt_frames).
    Chunks the video by args.framestride, exactly like test.py."""
    vid_len  = len(frames_all)
    sum_f    = [None] * vid_len
    cnt_f    = [0]    * vid_len

    x_frames = [frames_all[i:i + args.framestride] for i in range(0, vid_len, args.framestride)]
    x_masks  = [masks_all[i:i + args.framestride]  for i in range(0, vid_len, args.framestride)]

    for itern, (chunk_fr, chunk_mk) in enumerate(zip(x_frames, x_masks)):
        chunk_len = len(chunk_fr)
        imgs  = to_tensors()(chunk_fr).unsqueeze(0) * 2 - 1
        msk_t = to_tensors()(chunk_mk).unsqueeze(0)
        imgs, msk_t = imgs.to(device), msk_t.to(device)

        for f in range(0, chunk_len, neighbor_stride):
            nb_ids  = [i for i in range(max(0, f - neighbor_stride),
                                        min(chunk_len, f + neighbor_stride + 1))]
            ref_ids = get_ref_index(f, nb_ids, chunk_len)
            sel_imgs  = imgs[:1,  nb_ids + ref_ids]
            sel_masks = msk_t[:1, nb_ids + ref_ids]

            with torch.no_grad():
                pred, _ = model(sel_imgs, sel_imgs, sel_masks, len(nb_ids))
                pred    = ((pred[:len(nb_ids)] + 1) / 2).clamp(0, 1)
                pred_np = pred.cpu().permute(0, 2, 3, 1).numpy() * 255

            for i, idx in enumerate(nb_ids):
                gidx   = itern * args.framestride + idx
                mask2d = (np.array(chunk_mk[idx].convert('L')) > 0).astype(np.float32)
                raw    = np.array(chunk_fr[idx]).astype(np.float32)
                blended = pred_np[i].astype(np.float32) * mask2d[:, :, None] + \
                          raw * (1.0 - mask2d[:, :, None])
                if sum_f[gidx] is None:
                    sum_f[gidx] = blended
                else:
                    sum_f[gidx] += blended
                cnt_f[gidx] += 1

    return sum_f, cnt_f


def eval_psnr_video(video, model, device, size):
    bsc_dir  = os.path.join(args.val_bsc_dir,  video)
    gt_dir   = os.path.join(args.val_gt_dir,   video)
    mask_dir = os.path.join(args.val_mask_dir, video)
    if not all(os.path.isdir(d) for d in [bsc_dir, gt_dir, mask_dir]):
        return None

    bsc_fr  = read_frames(bsc_dir,  size)
    gt_fr   = read_frames(gt_dir,   size)
    mask_im = read_masks(mask_dir,  size)
    vl      = min(len(bsc_fr), len(gt_fr), len(mask_im))
    if vl < neighbor_stride + 1:
        return None
    bsc_fr = bsc_fr[:vl]; gt_fr = gt_fr[:vl]; mask_im = mask_im[:vl]

    was_training = model.training
    model.eval()
    sum_f, cnt_f = infer_pass(model, device, bsc_fr, mask_im)
    sum_b, cnt_b = infer_pass(model, device,
                               list(reversed(bsc_fr)), list(reversed(mask_im)))
    if was_training:
        model.train()

    psnrs = []
    for i in range(vl):
        rev_i = vl - 1 - i
        if (sum_f[i] is None or cnt_f[i] == 0 or
                sum_b[rev_i] is None or cnt_b[rev_i] == 0):
            continue
        mask_bin = (np.array(mask_im[i].convert('L')) > 0).astype(np.float32)
        if mask_bin.mean() < 1e-6:
            continue
        merged = (0.5 * sum_f[i] / cnt_f[i] +
                  0.5 * sum_b[rev_i] / cnt_b[rev_i]).clip(0, 255)
        gt_np  = np.array(gt_fr[i]).astype(np.float64)
        mse    = np.mean((merged.astype(np.float64) - gt_np) ** 2)
        if mse > 0:
            psnrs.append(20.0 * np.log10(255.0 / np.sqrt(mse)))

    return float(np.mean(psnrs)) if psnrs else None


def run_eval(videos, model, device, size, label=''):
    results = {}
    for v in videos:
        pp = eval_psnr_video(v, model, device, size)
        if pp is not None:
            results[v] = pp
    if not results:
        tqdm.write(f'  [EVAL{label}] no valid results')
        return None
    mean_p = float(np.mean(list(results.values())))
    for v, pp in results.items():
        tqdm.write(f'  [EVAL{label}]  {v:<20s}  {pp:.4f} dB')
    tqdm.write(f'  [EVAL{label}]  {"MEAN":<20s}  {mean_p:.4f} dB')
    return mean_p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def charbonnier(x, eps=1e-3):
    """Charbonnier (smooth L1) loss: sqrt(x^2 + eps^2). More robust than plain L1."""
    return torch.sqrt(x * x + eps * eps)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    size   = (W, H)

    # Performance: TF32 for Ampere+ GPUs, cuDNN auto-tuner
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.backends.cudnn.benchmark        = True
    print(f'Device: {device}  (TF32={torch.backends.cuda.matmul.allow_tf32}, '
          f'cuDNN_benchmark={torch.backends.cudnn.benchmark})')

    # Load model — all frozen
    net_mod = importlib.import_module('model.' + args.model)
    model   = net_mod.InpaintGenerator(init_weights=False).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    print(f'Loaded frozen model: {args.ckpt}')

    # FIX 2: scan worst videos with clean frozen model BEFORE LoRA injection
    eval_videos = []
    if args.eval_freq > 0 and args.val_bsc_dir:
        if args.eval_videos_dir:
            eval_videos = sorted([v for v in os.listdir(args.eval_videos_dir)
                                  if os.path.isdir(os.path.join(args.eval_videos_dir, v))])
            print(f'Eval videos ({len(eval_videos)}): {eval_videos}\n')
        else:
            val_videos = sorted([v for v in os.listdir(args.val_bsc_dir)
                                 if os.path.isdir(os.path.join(args.val_bsc_dir, v))])
            print(f'Scanning val set for worst-{args.eval_n} (frozen baseline) ...')
            baseline = {}
            for v in val_videos:
                pp = eval_psnr_video(v, model, device, size)
                if pp is not None:
                    baseline[v] = pp
            sorted_bl   = sorted(baseline.items(), key=lambda x: x[1])
            eval_videos = [v for v, _ in sorted_bl[:args.eval_n]]
            print(f'Worst-{args.eval_n}: {eval_videos}')
            print(f'Frozen baseline mean: {np.mean([baseline[v] for v in eval_videos]):.4f} dB\n')

    # FIX 1+6: inject LoRA, print keys for safety
    inject_lora(model.sam2_encoder.encoder, r=args.lora_rank, alpha=args.lora_alpha)
    model.to(device)  # move newly created lora_A/lora_B (CPU) to GPU
    lora_keys   = [n for n, _ in model.named_parameters() if 'lora_' in n]
    lora_params = [p for n, p in model.named_parameters() if 'lora_' in n]
    assert len(lora_params) > 0, 'No LoRA params found — check inject_lora()'
    assert all(p.requires_grad for p in lora_params), 'Some LoRA params are frozen!'
    print(f'  LoRA adapters injected (rank={args.lora_rank}, alpha={args.lora_alpha})')
    print(f'  LoRA keys ({len(lora_keys)}):')
    for k in lora_keys[:6]:
        print(f'    {k}')
    if len(lora_keys) > 6:
        print(f'    ... ({len(lora_keys) - 6} more)')
    print(f'  LoRA params: {sum(p.numel() for p in lora_params):,}')

    # Unfreeze SAMFuser
    for param in model.sam_fuser.parameters():
        param.requires_grad_(True)
    fuser_params = list(model.sam_fuser.parameters())
    print(f'  SAMFuser params (unfrozen): {sum(p.numel() for p in fuser_params):,}')
    print(f'  Total trainable: {sum(p.numel() for p in lora_params + fuser_params):,}')

    # Safety: warn if any non-LoRA, non-fuser params accidentally leaked as trainable
    leaked = [(n, p.numel()) for n, p in model.named_parameters()
              if p.requires_grad and 'lora_' not in n and 'sam_fuser' not in n]
    if leaked:
        print(f'\nWARNING: {len(leaked)} unintended trainable params detected:')
        for n, c in leaked[:5]:
            print(f'  {n} ({c:,})')
        if len(leaked) > 5:
            print(f'  ... ({len(leaked) - 5} more)')
    print()

    model.train()

    # FIX 5: separate param groups — LoRA higher LR, no weight decay
    opt = torch.optim.Adam([
        {'params': lora_params,  'lr': args.lora_lr, 'weight_decay': 0.0},
        {'params': fuser_params, 'lr': args.lr,       'weight_decay': 1e-5},
    ], betas=(0.9, 0.999))

    milestones = []
    if args.lr_drop_milestones.strip():
        milestones = [int(x.strip()) for x in args.lr_drop_milestones.split(',')
                      if x.strip()]

    if args.scheduler == 'cosine':
        warmup = 200
        def lr_lambda(it):
            if it < warmup:
                return it / max(1, warmup)
            t = (it - warmup) / max(1, args.iterations - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * t))
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    else:
        # Manual multi-step in iteration units (not epoch units).
        scheduler = None
        milestones = sorted(milestones)
        dropped = set()
    scaler    = GradScaler()

    os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)

    # Resume
    start_it = 0
    if args.resume:
        ckpt_data = torch.load(args.resume, map_location=device)
        missing, unexpected = model.load_state_dict(ckpt_data['lora'], strict=False)
        loaded_lora = [k for k in ckpt_data['lora'] if 'lora_' in k]
        assert len(loaded_lora) > 0, 'No lora_ keys in resume checkpoint — wrong file?'
        print(f'Resume LoRA: {len(loaded_lora)} keys loaded, {len(missing)} missing, {len(unexpected)} unexpected')
        model.sam_fuser.load_state_dict(ckpt_data['sam_fuser'])
        if 'optimizer' in ckpt_data:
            opt.load_state_dict(ckpt_data['optimizer'])
        if scheduler is not None and 'scheduler' in ckpt_data:
            scheduler.load_state_dict(ckpt_data['scheduler'])
        start_it = ckpt_data.get('iteration', 0)
        # Always enforce CLI LR after resume; optimizer state can silently override it.
        opt.param_groups[0]['lr'] = args.lora_lr
        opt.param_groups[1]['lr'] = args.lr
        # Also enforce intended weight decay after optimizer state load.
        opt.param_groups[0]['weight_decay'] = 0.0
        opt.param_groups[1]['weight_decay'] = 1e-5
        if hasattr(scheduler, 'base_lrs'):
            scheduler.base_lrs = [args.lora_lr, args.lr]
        if args.scheduler == 'multistep':
            if 'dropped_milestones' in ckpt_data:
                dropped = set(int(x) for x in ckpt_data['dropped_milestones'])
            else:
                dropped = {ms for ms in milestones if start_it >= ms}
            # Rebuild LR to the correct stage at resume iteration.
            ndrops = sum(1 for ms in milestones if start_it >= ms)
            scale = args.lr_drop_gamma ** ndrops
            opt.param_groups[0]['lr'] = args.lora_lr * scale
            opt.param_groups[1]['lr'] = args.lr * scale
        print(f'Resumed from {args.resume}  (iteration {start_it})')
        print(f'  LR override after resume: lora={args.lora_lr:.2e}, fuser={args.lr:.2e}')

    videos = sorted([v for v in os.listdir(BSC_ROOT)
                     if os.path.isdir(os.path.join(BSC_ROOT, v))])
    print(f'{len(videos)} training videos')
    print(f'Training for {args.iterations:,} iters  '
          f'(lora_lr={args.lora_lr}, fuser_lr={args.lr}, '
          f'windows_per_clip={args.windows_per_clip})\n')

    it           = start_it
    running_loss = 0.0
    valid_steps  = 0
    skipped_mask = 0
    skipped_hard = 0
    best_psnr    = -1.0
    best_ckpt    = args.save.replace('.pth', '_best.pth')

    pbar = tqdm(total=args.iterations, initial=start_it, unit='it',
                dynamic_ncols=True, smoothing=0.01)

    while it < args.iterations:
        video    = random.choice(videos)
        bsc_dir  = os.path.join(BSC_ROOT,  video)
        gt_dir   = os.path.join(GT_ROOT,   video)
        mask_dir = os.path.join(MASK_ROOT, video)
        if not all(os.path.isdir(d) for d in [bsc_dir, gt_dir, mask_dir]):
            continue

        bsc_frames = read_frames(bsc_dir, size)
        gt_frames  = read_frames(gt_dir,  size)
        mask_imgs  = read_masks(mask_dir, size)
        vid_len    = min(len(bsc_frames), len(gt_frames), len(mask_imgs))
        if vid_len < neighbor_stride + 1:
            continue
        bsc_frames = bsc_frames[:vid_len]
        gt_frames  = gt_frames[:vid_len]
        mask_imgs  = mask_imgs[:vid_len]

        # Random clip
        s         = random.randint(0, max(0, vid_len - args.framestride))
        clip_bsc  = bsc_frames[s:s + args.framestride]
        clip_gt   = gt_frames [s:s + args.framestride]
        clip_mask = mask_imgs  [s:s + args.framestride]
        clip_len  = len(clip_bsc)

        imgs    = to_tensors()(clip_bsc).unsqueeze(0) * 2 - 1
        gt_t    = to_tensors()(clip_gt ).unsqueeze(0) * 2 - 1
        masks_t = to_tensors()(clip_mask).unsqueeze(0)
        imgs, gt_t, masks_t = imgs.to(device), gt_t.to(device), masks_t.to(device)

        # Hard-window candidates by frame-wise mask ratio in this clip.
        hard_ids = None
        frame_ratios = masks_t[0, :clip_len].mean(dim=(1, 2, 3))
        elig = torch.where(frame_ratios > args.hard_window_min_mask)[0]
        if elig.numel() > 0:
            top_k = max(1, int(math.ceil(float(elig.numel()) * args.hard_window_top_frac)))
            top_local = torch.topk(frame_ratios[elig], k=top_k, largest=True).indices
            hard_ids = elig[top_local]
        elif args.hard_window_require:
            skipped_hard += 1
            continue

        # Each window is an independent step (separate fwd/bwd) — minimises peak VRAM
        model.train()
        loss = None
        for _ in range(args.windows_per_clip):
            if it >= args.iterations:
                break
            if hard_ids is not None and hard_ids.numel() > 0:
                pick = torch.randint(low=0, high=hard_ids.numel(), size=(1,), device=hard_ids.device)
                f = int(hard_ids[pick].item())
            else:
                f = random.randint(0, clip_len - 1)
            nb_ids  = [i for i in range(max(0, f - neighbor_stride),
                                        min(clip_len, f + neighbor_stride + 1))]
            ref_ids = get_ref_index(f, nb_ids, clip_len)
            n_local = len(nb_ids)

            sel_imgs  = imgs[:1,    nb_ids + ref_ids]
            sel_masks = masks_t[:1, nb_ids + ref_ids]
            sel_gt    = gt_t[:1,    nb_ids + ref_ids]

            if sel_masks[:, :n_local].mean().item() < 1e-6:
                skipped_mask += 1
                continue

            opt.zero_grad(set_to_none=True)
            with autocast(dtype=torch.float16):
                pred, _ = model(sel_imgs, sel_imgs, sel_masks, n_local)
                pred     = pred[:n_local, :, :H, :W]
                gt_loc   = sel_gt[0,   :n_local, :, :H, :W]
                mask_loc = sel_masks[0, :n_local, :, :H, :W]
                bsc_loc  = sel_imgs[0,  :n_local, :, :H, :W]
                comp     = pred * mask_loc + bsc_loc * (1.0 - mask_loc)
                err      = charbonnier(comp - gt_loc)
                loss     = ((err * mask_loc).sum() / (mask_loc.sum() + 1e-6)
                            + 0.05 * (err * (1.0 - mask_loc)).mean())

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(lora_params + fuser_params, 1.0)
            scaler.step(opt)
            scaler.update()
            if args.scheduler == 'cosine':
                scheduler.step()
            else:
                # Manual multistep-by-iteration: drop exactly once at each milestone.
                next_it = it + 1
                for ms in milestones:
                    if next_it >= ms and ms not in dropped:
                        for pg in opt.param_groups:
                            pg['lr'] *= args.lr_drop_gamma
                        dropped.add(ms)

            running_loss += loss.item()
            valid_steps += 1
            it += 1
            pbar.update(1)

        if it % 10 == 0 and loss is not None:
            pbar.set_description(
                f"L1={loss.item():.4f} "
                f"lr_lora={opt.param_groups[0]['lr']:.1e}")

        if it % args.log_freq == 0:
            avg = running_loss / max(1, valid_steps)
            skip_total = skipped_mask + skipped_hard
            skip_rate = (100.0 * skip_total / max(1, valid_steps + skip_total))
            tqdm.write(
                f'[{it:6d}/{args.iterations}]  L1={avg:.5f}  '
                f'valid={valid_steps}/{args.log_freq}  '
                f'skip_mask={skipped_mask}  skip_hard={skipped_hard}  '
                f'skip_rate={skip_rate:.1f}%  '
                f'lr_lora={opt.param_groups[0]["lr"]:.1e}  '
                f'lr_fuser={opt.param_groups[1]["lr"]:.1e}')
            running_loss = 0.0
            valid_steps = 0
            skipped_mask = 0
            skipped_hard = 0

        if it % 1000 == 0:
            ckpt_path = args.save.replace('.pth', f'_iter{it:05d}.pth')
            torch.save({
                'iteration': it,
                'rank':      args.lora_rank,
                'alpha':     args.lora_alpha,
                'lora':      {k: v for k, v in model.state_dict().items() if 'lora_' in k},
                'sam_fuser': model.sam_fuser.state_dict(),
                'optimizer': opt.state_dict(),
                **({'scheduler': scheduler.state_dict()} if scheduler is not None else {}),
                **({'dropped_milestones': sorted(list(dropped))}
                   if args.scheduler == 'multistep' else {}),
            }, ckpt_path)
            tqdm.write(f'  -> saved {ckpt_path}')

            if args.eval_freq > 0 and it % args.eval_freq == 0 and eval_videos:
                pbar.set_description('eval')
                mean_p = run_eval(eval_videos, model, device, size, label=f' iter={it}')
                pbar.set_description('train')
                if mean_p is not None and mean_p > best_psnr:
                    best_psnr = mean_p
                    torch.save({
                        'iteration': it,
                        'rank':      args.lora_rank,
                        'alpha':     args.lora_alpha,
                        'lora':      {k: v for k, v in model.state_dict().items() if 'lora_' in k},
                        'sam_fuser': model.sam_fuser.state_dict(),
                    }, best_ckpt)
                    tqdm.write(f'  -> best  {best_psnr:.4f} dB  -> {best_ckpt}')

    pbar.close()
    torch.save({
        'iteration': it,
        'rank':      args.lora_rank,
        'alpha':     args.lora_alpha,
        'lora':      {k: v for k, v in model.state_dict().items() if 'lora_' in k},
        'sam_fuser': model.sam_fuser.state_dict(),
    }, args.save)
    print(f'\nDone. Saved: {args.save}')


if __name__ == '__main__':
    main()
