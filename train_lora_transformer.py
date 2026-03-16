"""
LoRA finetune the frozen TemporalFocalTransformer (WindowAttention qkv+proj).

Targets the 8 WindowAttention blocks (dim=512) that directly produce restoration output.
~265K trainable params, no SAMFuser involvement.

Usage
-----
  python train_lora_transformer.py \
      --config  config/lora_sam2.json \
      --ckpt    checkpoints/main_best.pth \
      --save    checkpoints/lora_transformer.pth \
      --iterations 10000 \
      --lora_lr 1e-4 \
      --lora_rank 8 --lora_alpha 16 \
      --eval_videos_dir worst10/bsc_imgs \
      --val_bsc_dir  /path/to/val/bsc_imgs \
      --val_gt_dir   /path/to/val/gt_imgs \
      --val_mask_dir /path/to/val/masks

Optionally stack on top of existing SAM2 LoRA:
      --sam2_lora_ckpt checkpoints/lora_sam2_best.pth
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
from model.modules.lora_utils import inject_lora_sam2 as inject_lora, inject_lora_transformer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
p = argparse.ArgumentParser()
p.add_argument('--config',           required=True)
p.add_argument('--ckpt',             required=True, help='Frozen main model checkpoint')
p.add_argument('--save',             default='checkpoints/lora_transformer.pth')
p.add_argument('--sam2_lora_ckpt',   default=None,
               help='Optional: pre-trained SAM2 LoRA checkpoint to load before training')
p.add_argument('--iterations',       type=int,   default=10000)
p.add_argument('--lora_lr',          type=float, default=1e-4)
p.add_argument('--lora_rank',        type=int,   default=8)
p.add_argument('--lora_alpha',       type=int,   default=16)
p.add_argument('--windows_per_clip', type=int,   default=1)
p.add_argument('--log_freq',         type=int,   default=100)
p.add_argument('--eval_freq',        type=int,   default=1000)
p.add_argument('--model',            type=str,   default='b2scvr_hq')
p.add_argument('--step',             type=int,   default=10)
p.add_argument('--num_ref',          type=int,   default=-1)
p.add_argument('--neighbor_stride',  type=int,   default=5)
p.add_argument('--framestride',      type=int,   default=30)
p.add_argument('--resume',           type=str,   default=None)
p.add_argument('--val_bsc_dir',      default=None)
p.add_argument('--val_gt_dir',       default=None)
p.add_argument('--val_mask_dir',     default=None)
p.add_argument('--eval_videos_dir',  default=None)
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
# Data helpers  (identical to train_lora_sam2.py)
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
# Validation  (identical to train_lora_sam2.py)
# ---------------------------------------------------------------------------
def infer_pass(model, device, frames_all, masks_all):
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
# Loss
# ---------------------------------------------------------------------------
def charbonnier(x, eps=1e-3):
    return torch.sqrt(x * x + eps * eps)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    size   = (W, H)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.backends.cudnn.benchmark        = True
    print(f'Device: {device}  (TF32={torch.backends.cuda.matmul.allow_tf32}, '
          f'cuDNN_benchmark={torch.backends.cudnn.benchmark})')

    # Load model — fully frozen
    net_mod = importlib.import_module('model.' + args.model)
    model   = net_mod.InpaintGenerator(init_weights=False).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    print(f'Loaded frozen model: {args.ckpt}')

    # Optionally load a pre-trained SAM2 LoRA checkpoint (keeps SAM2 adapted)
    if args.sam2_lora_ckpt:
        sam2_data = torch.load(args.sam2_lora_ckpt, map_location=device)
        rank_s  = sam2_data.get('rank', 8)
        alpha_s = sam2_data.get('alpha', 16)
        inject_lora(model.sam2_encoder.encoder, r=rank_s, alpha=alpha_s)
        model.load_state_dict(sam2_data['lora'], strict=False)
        model.sam_fuser.load_state_dict(sam2_data['sam_fuser'])
        # Keep SAM2 LoRA + SAMFuser frozen for this training stage
        for param in model.parameters():
            param.requires_grad_(False)
        print(f'Loaded SAM2 LoRA from: {args.sam2_lora_ckpt}  (kept frozen)')

    # Scan worst videos BEFORE transformer LoRA injection
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

    # --- Inject transformer LoRA ONLY ---
    # include_proj=False: qkv-only first (cheaper, usually sufficient)
    # Set include_proj=True to also adapt the output projection.
    include_proj = False
    inject_lora_transformer(model, r=args.lora_rank, alpha=args.lora_alpha,
                            include_proj=include_proj)
    model.to(device)  # move newly created lora_A/lora_B (CPU) to GPU

    # Filter by BOTH name AND requires_grad — avoids pulling in frozen SAM2 lora_ params
    # if --sam2_lora_ckpt was used (those are frozen after loading above).
    lora_named  = [(n, p) for n, p in model.named_parameters()
                   if 'lora_' in n and p.requires_grad]
    lora_params = [p for _, p in lora_named]
    assert len(lora_params) > 0, 'No trainable LoRA params found — inject_lora_transformer() may have hit 0 WindowAttention blocks'

    print(f'Trainable LoRA params: {sum(p.numel() for p in lora_params):,}')
    print('Example keys:')
    for n, _ in lora_named[:10]:
        print(f'  {n}')

    # Safety: confirm no unintended non-lora params are trainable
    leaked = [n for n, p in model.named_parameters() if p.requires_grad and 'lora_' not in n]
    if leaked:
        print(f'WARNING leaked trainables: {leaked[:10]}')
    print()

    model.train()

    opt = torch.optim.Adam(lora_params, lr=args.lora_lr, weight_decay=0.0,
                           betas=(0.9, 0.999))

    warmup = 200
    def lr_lambda(it):
        if it < warmup:
            return it / max(1, warmup)
        t = (it - warmup) / max(1, args.iterations - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * t))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler    = GradScaler()

    os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)

    # Resume
    start_it = 0
    if args.resume:
        ckpt_data = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt_data['lora'], strict=False)
        if 'optimizer' in ckpt_data:
            opt.load_state_dict(ckpt_data['optimizer'])
        if 'scheduler' in ckpt_data:
            scheduler.load_state_dict(ckpt_data['scheduler'])
        start_it = ckpt_data.get('iteration', 0)
        print(f'Resumed from {args.resume}  (iteration {start_it})')

    videos = sorted([v for v in os.listdir(BSC_ROOT)
                     if os.path.isdir(os.path.join(BSC_ROOT, v))])
    print(f'{len(videos)} training videos')
    print(f'Training for {args.iterations:,} iters  '
          f'(lora_lr={args.lora_lr}, rank={args.lora_rank})\n')

    it           = start_it
    running_loss = 0.0
    best_psnr    = -1.0
    best_ckpt    = args.save.replace('.pth', '_best.pth')

    # Pre-compute keys that are actually trainable in this run — excludes any frozen SAM2 lora_ params
    trainable_lora_keys = {n for n, p in model.named_parameters()
                           if 'lora_' in n and p.requires_grad}

    def save_ckpt(path, include_opt=True):
        d = {
            'iteration':    it,
            'rank':         args.lora_rank,
            'alpha':        args.lora_alpha,
            'include_proj': include_proj,
            'lora':         {k: v for k, v in model.state_dict().items()
                             if k in trainable_lora_keys},
        }
        if include_opt:
            d['optimizer'] = opt.state_dict()
            d['scheduler'] = scheduler.state_dict()
        torch.save(d, path)

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

        s         = random.randint(0, max(0, vid_len - args.framestride))
        clip_bsc  = bsc_frames[s:s + args.framestride]
        clip_gt   = gt_frames [s:s + args.framestride]
        clip_mask = mask_imgs  [s:s + args.framestride]
        clip_len  = len(clip_bsc)

        imgs    = to_tensors()(clip_bsc).unsqueeze(0) * 2 - 1
        gt_t    = to_tensors()(clip_gt ).unsqueeze(0) * 2 - 1
        masks_t = to_tensors()(clip_mask).unsqueeze(0)
        imgs, gt_t, masks_t = imgs.to(device), gt_t.to(device), masks_t.to(device)

        model.train()
        loss = None
        for _ in range(args.windows_per_clip):
            if it >= args.iterations:
                break
            f       = random.randint(0, clip_len - 1)
            nb_ids  = [i for i in range(max(0, f - neighbor_stride),
                                        min(clip_len, f + neighbor_stride + 1))]
            ref_ids = get_ref_index(f, nb_ids, clip_len)
            n_local = len(nb_ids)

            sel_imgs  = imgs[:1,    nb_ids + ref_ids]
            sel_masks = masks_t[:1, nb_ids + ref_ids]
            sel_gt    = gt_t[:1,    nb_ids + ref_ids]

            if sel_masks[:, :n_local].mean().item() < 1e-6:
                continue

            opt.zero_grad(set_to_none=True)
            with autocast(dtype=torch.float16):
                pred, _ = model(sel_imgs, sel_imgs, sel_masks, n_local)
                pred     = pred[:n_local, :, :H, :W]
                gt_loc   = sel_gt[0,   :n_local, :, :H, :W]
                mask_loc = sel_masks[0, :n_local, :, :H, :W]
                bsc_loc  = sel_imgs[0,  :n_local, :, :H, :W]
                comp     = pred * mask_loc + bsc_loc * (1.0 - mask_loc)
                err  = charbonnier(comp - gt_loc)
                loss = (err * mask_loc).sum() / (mask_loc.sum() + 1e-6)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            scaler.step(opt)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()
            it += 1
            pbar.update(1)

        if it % 10 == 0 and loss is not None:
            pbar.set_description(
                f"loss={loss.item():.4f} lr={opt.param_groups[0]['lr']:.1e}")

        if it % args.log_freq == 0 and it > 0:
            avg = running_loss / args.log_freq
            tqdm.write(f'[{it:6d}/{args.iterations}]  loss={avg:.5f}  '
                       f'lr={opt.param_groups[0]["lr"]:.1e}')
            running_loss = 0.0

        if it % 1000 == 0 and it > 0:
            ckpt_path = args.save.replace('.pth', f'_iter{it:05d}.pth')
            save_ckpt(ckpt_path)
            tqdm.write(f'  -> saved {ckpt_path}')

            if args.eval_freq > 0 and it % args.eval_freq == 0 and eval_videos:
                pbar.set_description('eval')
                mean_p = run_eval(eval_videos, model, device, size, label=f' iter={it}')
                pbar.set_description('train')
                if mean_p is not None and mean_p > best_psnr:
                    best_psnr = mean_p
                    save_ckpt(best_ckpt, include_opt=False)
                    tqdm.write(f'  -> best  {best_psnr:.4f} dB  -> {best_ckpt}')

    pbar.close()
    save_ckpt(args.save)
    print(f'\nDone. Saved: {args.save}')


if __name__ == '__main__':
    main()
