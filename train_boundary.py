"""
Training script for BoundaryRefinementHead.

Strategy:
  - InpaintGenerator is fully FROZEN (no gradients)
  - Only BoundaryRefinementHead parameters are trained
  - Loss: L1 on boundary band pixels vs GT (same weighting as hole_loss)
  - Reads directly from gt_root / bsc_root / mask_root dirs (dir format)

Usage:
    python train_boundary.py --config config/finetune_5k.json \
                             --ckpt   checkpoints/B2SCVR_ckpts/B2SCVR.pth \
                             --save   checkpoints/boundary_head.pth \
                             --iters  5000 \
                             --lr     1e-4
"""

import os
import glob
import random
import argparse
import json
import importlib
import logging
import numpy as np
from tqdm import tqdm

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from model.boundary_head import BoundaryRefinementHead


# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True,
                    help='Path to training config JSON')
parser.add_argument('--ckpt',   type=str, required=True,
                    help='Path to pretrained InpaintGenerator .pth')
parser.add_argument('--save',   type=str, default='checkpoints/boundary_head.pth',
                    help='Where to save the trained boundary head')
parser.add_argument('--iters',  type=int, default=5000)
parser.add_argument('--lr',     type=float, default=1e-4)
parser.add_argument('--batch',  type=int, default=1)
parser.add_argument('--log_freq',  type=int, default=100)
parser.add_argument('--save_freq', type=int, default=1000)
parser.add_argument('--val_freq',  type=int, default=2500,
                    help='Run validation every N iters (0 = disabled)')
parser.add_argument('--val_n',     type=int, default=10,
                    help='Number of validation videos to evaluate')
parser.add_argument('--val_gt_root',   type=str,
                    default='/media/ajeet/data/MINI_BTP/data/validation/gt_imgs')
parser.add_argument('--val_bsc_root',  type=str,
                    default='/media/ajeet/data/MINI_BTP/data/validation/bsc_imgs')
parser.add_argument('--val_mask_root', type=str,
                    default='/media/ajeet/data/MINI_BTP/data/validation/masks')
args = parser.parse_args()

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger()

# ── Config ────────────────────────────────────────────────────────────────────
with open(args.config) as f:
    config = json.load(f)

dl_cfg = config['train_data_loader']
GT_ROOT   = dl_cfg['gt_root']
BSC_ROOT  = dl_cfg['bsc_root']
MASK_ROOT = dl_cfg['mask_root']
NUM_LOCAL = dl_cfg['num_local_frames']
NUM_REF   = dl_cfg['num_ref_frames']
W, H      = dl_cfg['w'], dl_cfg['h']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log.info(f'Device: {device}')


# ── Dataset ──────────────────────────────────────────────────────────────────
class DirTrainDataset(Dataset):
    """Reads gt/bsc/mask from plain image directories."""

    def __init__(self, gt_root, bsc_root, mask_root, num_local, num_ref, h, w):
        self.gt_root   = gt_root
        self.bsc_root  = bsc_root
        self.mask_root = mask_root
        self.num_local = num_local
        self.num_ref   = num_ref
        self.h, self.w = h, w
        total_needed   = num_local + num_ref

        videos = sorted(os.listdir(gt_root))
        self.videos      = []
        self.frame_count = {}
        for v in videos:
            frames = sorted(glob.glob(os.path.join(gt_root, v, '*.jpg')) +
                            glob.glob(os.path.join(gt_root, v, '*.png')))
            if len(frames) >= total_needed + 1:
                self.videos.append(v)
                self.frame_count[v] = len(frames)

        log.info(f'DirTrainDataset: {len(self.videos)} usable clips')

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        needed = self.num_local + self.num_ref
        for _ in range(10):
            video = self.videos[idx % len(self.videos)]
            n     = self.frame_count[video]

            pivot     = random.randint(0, n - self.num_local)
            local_idx = list(range(pivot, pivot + self.num_local))
            remain    = [i for i in range(n) if i not in set(local_idx)]
            ref_idx   = sorted(random.sample(remain, self.num_ref))
            selected  = local_idx + ref_idx

            gt_imgs, bsc_imgs, mask_imgs = [], [], []
            for i in selected:
                stem = f'{i:05d}'
                gt   = cv2.imread(os.path.join(self.gt_root,   video, stem + '.jpg'))
                bsc  = cv2.imread(os.path.join(self.bsc_root,  video, stem + '.jpg'))
                mask = cv2.imread(os.path.join(self.mask_root, video, stem + '.png'),
                                  cv2.IMREAD_GRAYSCALE)
                if gt is None:
                    gt = cv2.imread(os.path.join(self.gt_root, video, stem + '.png'))
                if bsc is None:
                    bsc = cv2.imread(os.path.join(self.bsc_root, video, stem + '.png'))
                if gt is None or bsc is None or mask is None:
                    continue
                gt   = cv2.resize(cv2.cvtColor(gt,  cv2.COLOR_BGR2RGB), (self.w, self.h))
                bsc  = cv2.resize(cv2.cvtColor(bsc, cv2.COLOR_BGR2RGB), (self.w, self.h))
                mask = cv2.resize(mask, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
                gt_imgs.append(gt)
                bsc_imgs.append(bsc)
                mask_imgs.append(mask)

            # Require enough frames AND at least one local frame with corruption
            local_has_mask = any(mask_imgs[i].max() > 0 for i in range(min(self.num_local, len(mask_imgs))))
            if len(gt_imgs) >= needed and local_has_mask:
                break
            # retry with a random video
            idx = random.randint(0, len(self.videos) - 1)

        def to_rgb_tensor(imgs):
            arr = np.stack(imgs, 0).astype(np.float32) / 255.0  # (T,H,W,3)
            arr = arr.transpose(0, 3, 1, 2)                     # (T,3,H,W)
            return torch.from_numpy(arr) * 2.0 - 1.0            # [-1,1]

        def to_mask_tensor(imgs):
            arr = np.stack(imgs, 0).astype(np.float32)          # (T,H,W) 0..255
            arr = (arr > 0).astype(np.float32)
            return torch.from_numpy(arr).unsqueeze(1)           # (T,1,H,W) [0,1]

        return to_rgb_tensor(gt_imgs[:needed]), to_rgb_tensor(bsc_imgs[:needed]), to_mask_tensor(mask_imgs[:needed])


# ── Validation helper ─────────────────────────────────────────────────────────
def run_validation(netG, bhead, device,
                   val_gt_root, val_bsc_root, val_mask_root,
                   num_local, num_ref, h, w, n_videos=10):
    """Quick proxy PSNR on a fixed subset of validation videos.
    Evaluates on a single temporal window per video; measures hole PSNR only.
    """
    bhead.eval()
    needed  = num_local + num_ref
    psnrs   = []
    rng     = random.Random(42)   # fixed seed → same windows every call

    videos = sorted(os.listdir(val_bsc_root))[:n_videos]
    for video in videos:
        gt_dir   = os.path.join(val_gt_root,   video)
        bsc_dir  = os.path.join(val_bsc_root,  video)
        mask_dir = os.path.join(val_mask_root, video)
        if not all(os.path.isdir(d) for d in [gt_dir, bsc_dir, mask_dir]):
            continue

        bsc_paths = sorted(glob.glob(os.path.join(bsc_dir, '*.jpg')) +
                           glob.glob(os.path.join(bsc_dir, '*.png')))
        n = len(bsc_paths)
        if n < needed + 1:
            continue

        # Find a window that contains at least one corrupted frame
        pivot = None
        for _ in range(30):
            p = rng.randint(0, n - num_local)
            for fi in range(p, p + num_local):
                stem = os.path.splitext(os.path.basename(bsc_paths[fi]))[0]
                mp = os.path.join(mask_dir, stem + '.png')
                if os.path.exists(mp):
                    m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
                    if m is not None and m.max() > 0:
                        pivot = p
                        break
            if pivot is not None:
                break
        if pivot is None:
            continue

        local_idx = list(range(pivot, pivot + num_local))
        remain    = [i for i in range(n) if i not in set(local_idx)]
        if len(remain) < num_ref:
            continue
        ref_idx  = sorted(rng.sample(remain, num_ref))
        selected = local_idx + ref_idx

        # gt_nat: native GT resolution for PSNR eval
        # bsc_imgs / mask_inf: resized to inference resolution for model
        # mask_nat: native GT resolution for PSNR eval
        gt_nat_imgs, bsc_imgs, mask_inf_imgs, mask_nat_imgs = [], [], [], []
        ok = True
        for i in selected:
            stem = os.path.splitext(os.path.basename(bsc_paths[i]))[0]
            gt   = cv2.imread(os.path.join(gt_dir,  stem + '.jpg'))
            if gt is None:
                gt = cv2.imread(os.path.join(gt_dir, stem + '.png'))
            bsc  = cv2.imread(bsc_paths[i])
            mask = cv2.imread(os.path.join(mask_dir, stem + '.png'), cv2.IMREAD_GRAYSCALE)
            if gt is None or bsc is None or mask is None:
                ok = False; break
            gt_h_nat, gt_w_nat = gt.shape[:2]
            gt_nat_imgs.append(cv2.cvtColor(gt, cv2.COLOR_BGR2RGB))          # native res
            bsc_imgs.append(cv2.resize(cv2.cvtColor(bsc, cv2.COLOR_BGR2RGB), (w, h)))
            mask_inf_imgs.append(cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST))
            mask_nat_imgs.append(cv2.resize(mask, (gt_w_nat, gt_h_nat), interpolation=cv2.INTER_NEAREST))
        if not ok or len(gt_nat_imgs) < needed:
            continue

        def arr_t(imgs):
            a = np.stack(imgs, 0).astype(np.float32) / 255.0
            return torch.from_numpy(a.transpose(0, 3, 1, 2)).unsqueeze(0).to(device) * 2.0 - 1.0

        def msk_t(imgs):
            a = (np.stack(imgs, 0).astype(np.float32) > 0).astype(np.float32)
            return torch.from_numpy(a).unsqueeze(0).unsqueeze(2).to(device)

        bsc_t  = arr_t(bsc_imgs[:needed])        # (1, T, 3, h, w)  inference res
        mask_t = msk_t(mask_inf_imgs[:needed])   # (1, T, 1, h, w)  inference res

        with torch.no_grad():
            pred, _ = netG(bsc_t, bsc_t, mask_t, num_local)
            pred    = pred.view(1, needed, 3, h, w)[0, :num_local]  # (nl, 3, h, w)
            refined, _ = bhead(pred,
                               bsc_t[0, :num_local],
                               mask_t[0, :num_local])               # (nl, 3, h, w)
            refined_np = ((refined + 1) / 2 * 255.0).cpu().permute(0, 2, 3, 1).numpy()

        for i in range(num_local):
            m = mask_nat_imgs[i] > 0
            if not m.any():
                continue
            gt_nat  = gt_nat_imgs[i]                                 # (gt_H, gt_W, 3)
            gt_h_i, gt_w_i = gt_nat.shape[:2]
            pred_up = cv2.resize(refined_np[i], (gt_w_i, gt_h_i),
                                 interpolation=cv2.INTER_CUBIC)      # upsample to GT res
            mse = np.mean((pred_up[m].astype(np.float64) -
                           gt_nat[m].astype(np.float64)) ** 2)
            if mse > 0:
                psnrs.append(20.0 * np.log10(255.0 / np.sqrt(mse)))

    bhead.train()
    return float(np.mean(psnrs)) if psnrs else None


dataset = DirTrainDataset(GT_ROOT, BSC_ROOT, MASK_ROOT, NUM_LOCAL, NUM_REF, H, W)
loader  = DataLoader(dataset,
                     batch_size=args.batch,
                     shuffle=True,
                     num_workers=4,
                     drop_last=True)


# ── Main model (frozen) ──────────────────────────────────────────────────────
net  = importlib.import_module('model.' + config['model']['net'])
netG = net.InpaintGenerator(init_weights=False).to(device)
data = torch.load(args.ckpt, map_location=device)
netG.load_state_dict(data)
netG.eval()
for p in netG.parameters():
    p.requires_grad = False
log.info(f'Loaded frozen InpaintGenerator from {args.ckpt}')


# ── Boundary head ─────────────────────────────────────────────────────────────
bhead = BoundaryRefinementHead(channels=64).to(device)
opt   = torch.optim.Adam(bhead.parameters(), lr=args.lr, betas=(0.9, 0.999))
l1    = nn.L1Loss()
log.info(f'BoundaryRefinementHead params: '
         f'{sum(p.numel() for p in bhead.parameters())/1e6:.2f} M')

os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)

# Loss log file sits next to the checkpoint
loss_log_path = args.save.replace('.pth', '_loss.csv')
loss_log = open(loss_log_path, 'w')
loss_log.write('iteration,loss,avg100\n')
loss_log.flush()
log.info(f'Loss log -> {loss_log_path}')

# Validation PSNR log
val_log_path = args.save.replace('.pth', '_val_psnr.csv')
val_log = open(val_log_path, 'w')
val_log.write('iteration,val_psnr_db\n')
val_log.flush()
val_enabled = args.val_freq > 0 and os.path.isdir(args.val_bsc_root)
if val_enabled:
    log.info(f'Validation every {args.val_freq} iters on {args.val_n} videos -> {val_log_path}')
else:
    log.info('Validation disabled (val_freq=0 or val_bsc_root not found)')


# ── Training loop ─────────────────────────────────────────────────────────────
iteration   = 0
best_loss   = float('inf')
recent_losses = []
loader_iter = iter(loader)

pbar = tqdm(total=args.iters, desc='boundary head')
while iteration < args.iters:
    try:
        gt_frames, bsc_frames, masks = next(loader_iter)
    except StopIteration:
        loader_iter = iter(loader)
        gt_frames, bsc_frames, masks = next(loader_iter)

    gt_frames  = gt_frames.to(device)    # (B, T, 3, H, W)  [-1,1]
    bsc_frames = bsc_frames.to(device)   # (B, T, 3, H, W)  [-1,1]
    masks      = masks.to(device)        # (B, T, 1, H, W)  [0,1]

    b, t, c, h, w = gt_frames.size()

    # ── Run frozen main model in test-mode (bsc as both inputs, matches test.py) ──
    with torch.no_grad():
        pred_imgs, _ = netG(bsc_frames, bsc_frames, masks, NUM_LOCAL)
        pred_imgs = pred_imgs.view(b, t, c, h, w)[:, :NUM_LOCAL]   # (B, lt, 3, H, W)

    gt_local    = gt_frames[:, :NUM_LOCAL]   # (B, lt, 3, H, W)  — GT used only for loss
    bsc_local   = bsc_frames[:, :NUM_LOCAL]  # (B, lt, 3, H, W)
    masks_local = masks[:, :NUM_LOCAL]       # (B, lt, 1, H, W)

    # Pass full BSC frame (not masked-out): boundary band straddles mask edge,
    # so inner-band pixels must see real BSC values, not zeros.
    original_local = bsc_local

    # ── Run boundary head frame-by-frame ─────────────────────────────────────
    pred_flat = pred_imgs.reshape(b * NUM_LOCAL, c, h, w)
    orig_flat = original_local.reshape(b * NUM_LOCAL, c, h, w)
    mask_flat = masks_local.reshape(b * NUM_LOCAL, 1, h, w)

    refined, band = bhead(pred_flat, orig_flat, mask_flat)

    # ── Loss: L1 only inside boundary band ───────────────────────────────────
    gt_flat   = gt_local.reshape(b * NUM_LOCAL, c, h, w)
    band_mean = band.mean()

    # Skip if boundary band is essentially empty — no training signal
    if band_mean < 1e-4:
        continue

    loss = l1(refined * band, gt_flat * band) / band_mean.clamp(min=1e-6)

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(bhead.parameters(), 1.0)
    opt.step()

    loss_val = loss.item()
    recent_losses.append(loss_val)
    if len(recent_losses) > 100:
        recent_losses.pop(0)
    avg100 = sum(recent_losses) / len(recent_losses)

    iteration += 1
    pbar.update(1)
    pbar.set_postfix(loss=f'{loss_val:.4f}', avg100=f'{avg100:.4f}')

    # Write to loss CSV every iteration
    loss_log.write(f'{iteration},{loss_val:.6f},{avg100:.6f}\n')
    if iteration % 50 == 0:
        loss_log.flush()

    if iteration % args.log_freq == 0:
        log.info(f'[{iteration}/{args.iters}]  loss: {loss_val:.4f}  avg100: {avg100:.4f}')

    if iteration % args.save_freq == 0 or iteration == args.iters:
        iter_path = args.save.replace('.pth', f'_{iteration}.pth')
        torch.save(bhead.state_dict(), iter_path)
        log.info(f'  Saved -> {iter_path}')

    # Save best model separately
    if avg100 < best_loss:
        best_loss = avg100
        best_path = args.save.replace('.pth', '_best.pth')
        torch.save(bhead.state_dict(), best_path)

    # ── Periodic validation ───────────────────────────────────────────────────
    if val_enabled and iteration % args.val_freq == 0:
        val_psnr = run_validation(
            netG, bhead, device,
            args.val_gt_root, args.val_bsc_root, args.val_mask_root,
            NUM_LOCAL, NUM_REF, H, W, n_videos=args.val_n)
        if val_psnr is not None:
            log.info(f'  ★ [VAL iter={iteration}]  hole_PSNR = {val_psnr:.4f} dB')
            val_log.write(f'{iteration},{val_psnr:.6f}\n')
            val_log.flush()
        else:
            log.info(f'  [VAL iter={iteration}]  no valid frames found')

pbar.close()
loss_log.close()
val_log.close()
final_path = args.save.replace('.pth', f'_{args.iters}.pth')
torch.save(bhead.state_dict(), final_path)
log.info(f'Training complete. best avg100={best_loss:.4f}  Saved -> {final_path}')
