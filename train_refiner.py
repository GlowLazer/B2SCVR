"""
Train a lightweight per-frame residual refiner to post-process the main model output.

Expected directory layout (each root has subfolders per video):
  bsc_root/<video>/*.(png|jpg)
  gt_root/<video>/*.(png|jpg)
  mask_root/<video>/*.(png|jpg)   (binary-ish; nonzero treated as 1)
  pred_root/<video>/*.(png|jpg)   (main model predictions to refine)

Refiner runs in [0,1] and is constrained to never modify pixels outside mask.
"""

import argparse
import os
import random
import time
import subprocess

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.amp import GradScaler, autocast
from tqdm import tqdm


def _list_frames(dir_path):
    if not os.path.isdir(dir_path):
        return []
    names = sorted([n for n in os.listdir(dir_path) if not n.startswith('.')])
    out = []
    for n in names:
        ext = os.path.splitext(n)[1].lower()
        if ext in ('.png', '.jpg', '.jpeg', '.bmp', '.webp'):
            out.append(os.path.join(dir_path, n))
    return out


def _resolve_video_dir(root, video):
    """
    Support both:
      root/<video>/*.png
      root/<video>/frame_seq/*.png   (test.py output layout)
    Returns (dir_path, frame_list).
    """
    d0 = os.path.join(root, video)
    f0 = _list_frames(d0)
    if f0:
        return d0, f0
    d1 = os.path.join(root, video, 'frame_seq')
    f1 = _list_frames(d1)
    if f1:
        return d1, f1
    return d0, []


def _load_rgb01(path, size=None):
    im = Image.open(path).convert('RGB')
    if size is not None and im.size != size:
        im = im.resize(size, Image.BILINEAR)
    arr = np.asarray(im).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)  # (3,H,W)


def _load_mask01(path, size=None):
    im = Image.open(path).convert('L')
    if size is not None and im.size != size:
        im = im.resize(size, Image.NEAREST)
    arr = (np.asarray(im) > 0).astype(np.float32)
    return torch.from_numpy(arr).unsqueeze(0)  # (1,H,W)


def _charbonnier(x, eps=1e-3):
    return torch.sqrt(x * x + eps * eps)

def _masked_charb_mean(diff, mask, eps=1e-6):
    # diff: (B,C,H,W), mask: (B,1,H,W)
    c = float(diff.shape[1])
    num = _charbonnier(diff * mask).sum()
    den = mask.sum() * c + eps
    return num / den


def _compute_band(mask, dil_k=9, ero_k=5):
    # mask: (B,1,H,W) in {0,1}
    dil = F.max_pool2d(mask, kernel_size=dil_k, stride=1, padding=dil_k // 2)
    ero = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=ero_k, stride=1, padding=ero_k // 2)
    return (dil - ero).clamp(0.0, 1.0)


def _sobel(x):
    # x: (B,3,H,W) in [0,1] -> returns gradient magnitude (B,3,H,W)
    kx = torch.tensor([[-1.0, 0.0, 1.0],
                       [-2.0, 0.0, 2.0],
                       [-1.0, 0.0, 1.0]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    ky = torch.tensor([[-1.0, -2.0, -1.0],
                       [0.0, 0.0, 0.0],
                       [1.0, 2.0, 1.0]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    # depthwise conv for RGB
    kx = kx.repeat(3, 1, 1, 1)
    ky = ky.repeat(3, 1, 1, 1)
    gx = F.conv2d(x, kx, padding=1, groups=3)
    gy = F.conv2d(x, ky, padding=1, groups=3)
    return torch.sqrt(gx * gx + gy * gy + 1e-12)


def _crop_biased(bsc, pred, gt, mask, patch, min_mask_ratio=0.01):
    # tensors: (C,H,W) for rgb; (1,H,W) for mask
    _, h, w = bsc.shape
    if patch <= 0 or patch >= min(h, w):
        return bsc, pred, gt, mask

    m = mask[0]
    ys, xs = torch.where(m > 0.5)
    use_bias = ys.numel() > 0 and float(m.mean().item()) >= min_mask_ratio

    if use_bias:
        idx = torch.randint(0, ys.numel(), (1,)).item()
        cy = int(ys[idx].item())
        cx = int(xs[idx].item())
        y0 = max(0, min(h - patch, cy - patch // 2))
        x0 = max(0, min(w - patch, cx - patch // 2))
    else:
        y0 = random.randint(0, h - patch)
        x0 = random.randint(0, w - patch)

    y1 = y0 + patch
    x1 = x0 + patch
    return (bsc[:, y0:y1, x0:x1],
            pred[:, y0:y1, x0:x1],
            gt[:, y0:y1, x0:x1],
            mask[:, y0:y1, x0:x1])


@torch.no_grad()
def eval_psnr(refiner, device, bsc_root, gt_root, mask_root, pred_root, max_videos=10, max_frames=None):
    videos = sorted([d for d in os.listdir(bsc_root) if os.path.isdir(os.path.join(bsc_root, d))])
    videos = videos[:max_videos]

    psnrs = []
    for v in videos:
        _, bsc_list = _resolve_video_dir(bsc_root, v)
        _, gt_list = _resolve_video_dir(gt_root, v)
        _, mk_list = _resolve_video_dir(mask_root, v)
        _, pr_list = _resolve_video_dir(pred_root, v)
        n = min(len(bsc_list), len(gt_list), len(mk_list), len(pr_list))
        if n == 0:
            continue
        if max_frames is not None:
            n = min(n, int(max_frames))

        for i in range(n):
            bsc0 = Image.open(bsc_list[i]).convert('RGB')
            size = bsc0.size
            m = _load_mask01(mk_list[i], size=size)
            if float(m.mean().item()) < 1e-6:
                continue
            bsc = (torch.from_numpy(np.asarray(bsc0).astype(np.float32) / 255.0)
                     .permute(2, 0, 1)).unsqueeze(0).to(device)
            gt = _load_rgb01(gt_list[i], size=size).unsqueeze(0).to(device)
            pred = _load_rgb01(pr_list[i], size=size).unsqueeze(0).to(device)
            m = m.unsqueeze(0).to(device)

            out = refiner(bsc, m, pred)
            # Masked PSNR (tracks leaderboard gains): measure error only inside corruption mask.
            diff = (out - gt) * m
            mse = (diff ** 2).sum() / (m.sum() * 3.0 + 1e-6)
            mse = float(mse.item())
            if mse <= 0:
                continue
            psnr = 10.0 * np.log10(1.0 / mse)
            psnrs.append(psnr)

    return float(np.mean(psnrs)) if psnrs else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bsc_root', required=True)
    p.add_argument('--gt_root', required=True)
    p.add_argument('--mask_root', required=True)
    p.add_argument('--pred_root', required=True)
    p.add_argument('--eval_bsc_root', default=None,
                   help='Optional: eval BSC root (defaults to --bsc_root)')
    p.add_argument('--eval_gt_root', default=None,
                   help='Optional: eval GT root (defaults to --gt_root)')
    p.add_argument('--eval_mask_root', default=None,
                   help='Optional: eval mask root (defaults to --mask_root)')
    p.add_argument('--eval_pred_root', default=None,
                   help='Optional: eval pred root (defaults to --pred_root)')
    p.add_argument('--save', default='checkpoints/refiner.pth')
    p.add_argument('--iters', type=int, default=30000)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--wd', type=float, default=1e-4)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--patch', type=int, default=192)
    p.add_argument('--save_freq', type=int, default=1000,
                   help='Save an iter checkpoint every N iterations')
    p.add_argument('--resume', type=str, default=None,
                   help='Path to a refiner checkpoint to resume from')
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--log_freq', type=int, default=50)
    p.add_argument('--eval_freq', type=int, default=2000)
    p.add_argument('--eval_videos', type=int, default=10)
    p.add_argument('--eval_frames', type=int, default=None)
    p.add_argument('--min_mask_ratio', type=float, default=0.03)
    p.add_argument('--hard_prob', type=float, default=0.8,
                   help='Probability to sample a frame from top mask-ratio subset')
    p.add_argument('--hard_top_frac', type=float, default=0.30,
                   help='Top fraction of frames (by mask ratio) considered hard')
    p.add_argument('--hard_min_ratio', type=float, default=0.03,
                   help='Minimum per-frame mask ratio to be eligible as hard')
    p.add_argument('--use_lap', action='store_true',
                   help='Use 10ch input by adding pred laplacian features')
    p.add_argument('--dil_k', type=int, default=9)
    p.add_argument('--ero_k', type=int, default=5)
    p.add_argument('--w_mask', type=float, default=1.0)
    p.add_argument('--w_band', type=float, default=0.3)
    p.add_argument('--w_grad', type=float, default=0.05)

    # Optional: run full validation pipeline (test.py + psnr.py) every N iters.
    # This does NOT train on validation; it only evaluates.
    p.add_argument('--val_pipeline_eval', action='store_true',
                   help='If set, run test.py+psnr.py on validation every --val_eval_freq iters')
    p.add_argument('--val_eval_freq', type=int, default=2000)
    p.add_argument('--val_ckpt', type=str, default='checkpoints/ema_cons_run1/gen_best.pth')
    p.add_argument('--val_boundary_ckpt', type=str, default='checkpoints/boundary_head_best.pth')
    p.add_argument('--val_lora_ckpt', type=str, default='checkpoints/lora_sam2_run3_best.pth')
    p.add_argument('--val_lora_transformer_ckpt', type=str, default='checkpoints/lora_transformer_on_sam2_best.pth')
    p.add_argument('--val_video_dir', type=str, default='/media/ajeet/data/MINI_BTP/data/validation/bsc_imgs')
    p.add_argument('--val_mask_dir', type=str, default='/media/ajeet/data/MINI_BTP/data/validation/masks')
    p.add_argument('--val_width', type=int, default=432)
    p.add_argument('--val_height', type=int, default=240)
    p.add_argument('--val_out_root', type=str, default='outputs_val_refiner')
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)

    from model.refiner import ResidualRefiner
    in_ch = 10 if args.use_lap else 7
    refiner = ResidualRefiner(in_ch=in_ch, base_ch=32).to(device).train()

    opt = torch.optim.AdamW(refiner.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.999))
    scaler = GradScaler(enabled=False)
    start_it = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        if isinstance(ckpt, dict) and 'model' in ckpt:
            refiner.load_state_dict(ckpt['model'])
            if 'optimizer' in ckpt:
                opt.load_state_dict(ckpt['optimizer'])
            if 'scaler' in ckpt:
                scaler.load_state_dict(ckpt['scaler'])
            if 'in_ch' in ckpt:
                # If checkpoint says in_ch=10, it must match current run.
                ck_in = int(ckpt['in_ch'])
                if ck_in != in_ch:
                    raise SystemExit(f'Resume in_ch mismatch: ckpt in_ch={ck_in}, run in_ch={in_ch}')
            start_it = int(ckpt.get('iteration', 0))
        else:
            # Allow raw state_dict resume (no optimizer/scaler).
            refiner.load_state_dict(ckpt)
            start_it = 0
        # Enforce CLI hyperparams in case optimizer state overrides them.
        opt.param_groups[0]['lr'] = args.lr
        opt.param_groups[0]['weight_decay'] = args.wd
        print(f'Resumed from {args.resume} (iteration {start_it})')

    # Train on the subset that exists in pred_root (so partial pred_root works).
    pred_videos = sorted([d for d in os.listdir(args.pred_root)
                          if os.path.isdir(os.path.join(args.pred_root, d)) and not d.startswith('.')])
    # Keep only videos that exist across all roots
    usable = []
    meta = {}
    mask_ratio_cache = {}
    skipped = {'missing_gt': 0, 'missing_mask': 0, 'missing_pred': 0, 'no_frames': 0}
    for v in pred_videos:
        bsc_dir = os.path.join(args.bsc_root, v)
        gt_dir = os.path.join(args.gt_root, v)
        mk_dir = os.path.join(args.mask_root, v)
        pr_dir = os.path.join(args.pred_root, v)
        if not os.path.isdir(pr_dir):
            skipped['missing_pred'] += 1
            continue
        if not os.path.isdir(bsc_dir):
            # No counter for this currently; treat as no_frames.
            skipped['no_frames'] += 1
            continue
        if not os.path.isdir(gt_dir):
            skipped['missing_gt'] += 1
            continue
        if not os.path.isdir(mk_dir):
            skipped['missing_mask'] += 1
            continue

        _, bsc_list = _resolve_video_dir(args.bsc_root, v)
        _, gt_list = _resolve_video_dir(args.gt_root, v)
        _, mk_list = _resolve_video_dir(args.mask_root, v)
        _, pr_list = _resolve_video_dir(args.pred_root, v)
        n = min(len(bsc_list), len(gt_list), len(mk_list), len(pr_list))
        if n <= 0:
            skipped['no_frames'] += 1
            continue
        meta[v] = (bsc_list[:n], gt_list[:n], mk_list[:n], pr_list[:n])
        usable.append(v)

    def get_mask_ratios(video):
        if video in mask_ratio_cache:
            return mask_ratio_cache[video]
        _, _, mk_list, _ = meta[video]
        ratios = []
        for mp in mk_list:
            try:
                im = Image.open(mp).convert('L')
                arr = (np.asarray(im) > 0).astype(np.float32)
                ratios.append(float(arr.mean()))
            except Exception:
                ratios.append(0.0)
        mask_ratio_cache[video] = ratios
        return ratios

    def sample_frame_index(video):
        bsc_list, gt_list, mk_list, pr_list = meta[video]
        n = len(bsc_list)
        if n <= 1 or random.random() > args.hard_prob:
            return random.randint(0, n - 1)
        ratios = get_mask_ratios(video)
        elig = [i for i, r in enumerate(ratios) if r >= args.hard_min_ratio]
        if not elig:
            return random.randint(0, n - 1)
        elig = sorted(elig, key=lambda i: ratios[i])
        k = max(1, int(np.ceil(len(elig) * float(args.hard_top_frac))))
        cand = elig[-k:]
        return int(random.choice(cand))

    if not usable:
        msg = (
            "No usable videos found across roots.\n"
            f"bsc_root={args.bsc_root}\n"
            f"gt_root={args.gt_root}\n"
            f"mask_root={args.mask_root}\n"
            f"pred_root={args.pred_root}\n"
            f"skipped: {skipped}\n"
            "Note: pred_root may be test.py outputs; both pred_root/<video> and pred_root/<video>/frame_seq are supported."
        )
        raise SystemExit(msg)

    best_psnr = -1.0
    best_path = args.save.replace('.pth', '_best.pth')
    eval_bsc_root = args.eval_bsc_root or args.bsc_root
    eval_gt_root = args.eval_gt_root or args.gt_root
    eval_mask_root = args.eval_mask_root or args.mask_root
    eval_pred_root = args.eval_pred_root or args.pred_root

    best_val_psnr = -1.0
    best_val_path = args.save.replace('.pth', '_bestval.pth')
    val_csv = args.save.replace('.pth', '_val_psnr.csv')
    if args.val_pipeline_eval:
        os.makedirs(args.val_out_root, exist_ok=True)
        if not os.path.exists(val_csv):
            with open(val_csv, 'w') as f:
                f.write('iter,ckpt,psnr_db,ssim,out_dir\n')

    def run_val_pipeline(it, ckpt_path):
        out_dir = os.path.join(args.val_out_root, f'iter{it:05d}')
        # Fresh outputs per iter so psnr.py reads the right set.
        if os.path.isdir(out_dir):
            # Avoid rm -rf; remove only if exists to keep it safe.
            for root, dirs, files in os.walk(out_dir, topdown=False):
                for name in files:
                    try:
                        os.remove(os.path.join(root, name))
                    except Exception:
                        pass
                for name in dirs:
                    try:
                        os.rmdir(os.path.join(root, name))
                    except Exception:
                        pass
            try:
                os.rmdir(out_dir)
            except Exception:
                pass
        os.makedirs(out_dir, exist_ok=True)

        env = os.environ.copy()
        # Fix OpenMP/MKL runtime conflicts (cv2/skimage/torch mix).
        env.setdefault('MKL_THREADING_LAYER', 'GNU')
        env.setdefault('KMP_DISABLE_SHM', '1')
        env.setdefault('KMP_USE_SHM', '0')
        env.setdefault('OMP_NUM_THREADS', '1')
        env.setdefault('MKL_NUM_THREADS', '1')

        cmd_test = [
            'python', 'test.py',
            '--ckpt', args.val_ckpt,
            '--boundary_ckpt', args.val_boundary_ckpt,
            '--refiner_ckpt', ckpt_path,
            '--lora_ckpt', args.val_lora_ckpt,
            '--lora_transformer_ckpt', args.val_lora_transformer_ckpt,
            '--video_dir', args.val_video_dir,
            '--mask_dir', args.val_mask_dir,
            '--width', str(args.val_width),
            '--height', str(args.val_height),
            '--out_dir', out_dir,
        ]
        subprocess.run(cmd_test, check=True, env=env)

        cmd_psnr = ['python', 'psnr.py', '--input_dir', out_dir]
        proc = subprocess.run(cmd_psnr, check=True, env=env, text=True, capture_output=True)
        psnr = None
        ssim = None
        for line in proc.stdout.splitlines():
            if 'Overall avg PSNR' in line:
                # Overall avg PSNR  : 23.0751 dB
                parts = line.replace('dB', '').split()
                for tok in parts:
                    try:
                        psnr = float(tok)
                        break
                    except Exception:
                        continue
            if 'Overall avg SSIM' in line:
                parts = line.split()
                for tok in parts:
                    try:
                        ssim = float(tok)
                        break
                    except Exception:
                        continue
        return psnr, ssim, out_dir

    running = {'loss': 0.0, 'mask': 0.0, 'band': 0.0, 'grad': 0.0}
    last_t = time.time()

    # Show correct progress when resuming: start at start_it / args.iters.
    pbar = tqdm(range(start_it + 1, args.iters + 1),
                total=args.iters,
                initial=start_it,
                dynamic_ncols=True,
                smoothing=0.01)
    for it in pbar:
        opt.zero_grad(set_to_none=True)

        batch_loss = 0.0
        batch_lm = 0.0
        batch_lb = 0.0
        batch_lg = 0.0

        # Manual micro-batching: sample random frames
        for _ in range(args.batch):
            v = random.choice(usable)
            bsc_list, gt_list, mk_list, pr_list = meta[v]
            idx = sample_frame_index(v)

            bsc0 = Image.open(bsc_list[idx]).convert('RGB')
            size = bsc0.size
            bsc = (torch.from_numpy(np.asarray(bsc0).astype(np.float32) / 255.0)
                     .permute(2, 0, 1))
            gt = _load_rgb01(gt_list[idx], size=size)
            pred = _load_rgb01(pr_list[idx], size=size)
            mask = _load_mask01(mk_list[idx], size=size)

            bsc, pred, gt, mask = _crop_biased(
                bsc, pred, gt, mask, args.patch, min_mask_ratio=args.min_mask_ratio
            )

            # Random HFlip
            if random.random() < 0.5:
                bsc = torch.flip(bsc, dims=[2])
                gt = torch.flip(gt, dims=[2])
                pred = torch.flip(pred, dims=[2])
                mask = torch.flip(mask, dims=[2])

            bsc = bsc.unsqueeze(0).to(device)
            gt = gt.unsqueeze(0).to(device)
            pred = pred.unsqueeze(0).to(device)
            mask = mask.unsqueeze(0).to(device)

            mask = (mask > 0.5).float()
            if float(mask.mean().item()) < 1e-6:
                continue

            with autocast(device_type=device.type, dtype=torch.float16, enabled=False):
                out = refiner(bsc, mask, pred)
                band = _compute_band(mask, dil_k=args.dil_k, ero_k=args.ero_k)
                # Boundary ring (inside-mask seam region).
                ring = band * mask

                diff = out - gt
                l_mask = _masked_charb_mean(diff, mask)
                if float(ring.sum().item()) > 1e-6:
                    l_band = _masked_charb_mean(diff, ring)
                    l_grad = _masked_charb_mean(_sobel(out) - _sobel(gt), ring)
                else:
                    l_band = torch.zeros((), device=device, dtype=diff.dtype)
                    l_grad = torch.zeros((), device=device, dtype=diff.dtype)
                loss = args.w_mask * l_mask + args.w_band * l_band + args.w_grad * l_grad

            scaler.scale(loss / max(1, args.batch)).backward()
            batch_loss += float(loss.item())
            batch_lm += float(l_mask.item())
            batch_lb += float(l_band.item())
            batch_lg += float(l_grad.item())

        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(refiner.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        running['loss'] += batch_loss
        running['mask'] += batch_lm
        running['band'] += batch_lb
        running['grad'] += batch_lg

        if it % 10 == 0:
            pbar.set_description(f"loss={batch_loss/max(1,args.batch):.4f}")

        if it % args.log_freq == 0:
            dt = time.time() - last_t
            last_t = time.time()
            denom = float(args.log_freq)
            msg = (f'[{it:6d}/{args.iters}] '
                   f'loss={running["loss"]/denom:.4f} '
                   f'mask={running["mask"]/denom:.4f} '
                   f'band={running["band"]/denom:.4f} '
                   f'grad={running["grad"]/denom:.4f} '
                   f'lr={opt.param_groups[0]["lr"]:.1e} '
                   f'{dt:.1f}s')
            tqdm.write(msg)
            for k in running:
                running[k] = 0.0

        if args.save_freq > 0 and it % args.save_freq == 0:
            ckpt_path = args.save.replace('.pth', f'_iter{it:05d}.pth')
            torch.save({
                'iteration': it,
                'in_ch': in_ch,
                'model': refiner.state_dict(),
                'optimizer': opt.state_dict(),
                'scaler': scaler.state_dict(),
                'args': vars(args),
            }, ckpt_path)
            tqdm.write(f'  -> saved {ckpt_path}')

        # Optional: full pipeline validation every N iters (uses the iter checkpoint file).
        if args.val_pipeline_eval and args.val_eval_freq > 0 and it % args.val_eval_freq == 0:
            ckpt_path = args.save.replace('.pth', f'_iter{it:05d}.pth')
            if not os.path.isfile(ckpt_path):
                # Ensure the checkpoint exists for this iteration.
                torch.save({
                    'iteration': it,
                    'in_ch': in_ch,
                    'model': refiner.state_dict(),
                    'optimizer': opt.state_dict(),
                    'scaler': scaler.state_dict(),
                    'args': vars(args),
                }, ckpt_path)
                tqdm.write(f'  -> saved {ckpt_path}')
            try:
                psnr, ssim, out_dir = run_val_pipeline(it, ckpt_path)
            except subprocess.CalledProcessError as e:
                tqdm.write(f'  [VAL] pipeline failed at iter={it}: {e}')
                psnr, ssim, out_dir = None, None, ''
            if psnr is not None:
                tqdm.write(f'  [VAL] PSNR: {psnr:.4f} dB  SSIM: {ssim if ssim is not None else "NA"}')
                with open(val_csv, 'a') as f:
                    f.write(f'{it},{ckpt_path},{psnr:.4f},{"" if ssim is None else f"{ssim:.4f}"},{out_dir}\n')
                if psnr > best_val_psnr:
                    best_val_psnr = psnr
                    # Copy the exact checkpoint that produced best validation.
                    torch.save(torch.load(ckpt_path, map_location='cpu'), best_val_path)
                    tqdm.write(f'  -> best-val {best_val_psnr:.4f} dB  -> {best_val_path}')

        if args.eval_freq > 0 and it % args.eval_freq == 0:
            refiner.eval()
            psnr = eval_psnr(
                refiner, device,
                eval_bsc_root, eval_gt_root, eval_mask_root, eval_pred_root,
                max_videos=args.eval_videos, max_frames=args.eval_frames
            )
            refiner.train()
            if psnr is not None:
                tqdm.write(f'  [EVAL] mean PSNR: {psnr:.4f} dB')
                if psnr > best_psnr:
                    best_psnr = psnr
                    torch.save({
                        'iteration': it,
                        'in_ch': in_ch,
                        'psnr': psnr,
                        'model': refiner.state_dict(),
                        'optimizer': opt.state_dict(),
                        'scaler': scaler.state_dict(),
                        'args': vars(args),
                    }, best_path)
                    tqdm.write(f'  -> best {best_psnr:.4f} dB  -> {best_path}')

    torch.save({
        'iteration': args.iters,
        'in_ch': in_ch,
        'model': refiner.state_dict(),
        'optimizer': opt.state_dict(),
        'scaler': scaler.state_dict(),
        'args': vars(args),
    }, args.save)
    print(f'Done. Saved: {args.save}')


if __name__ == '__main__':
    main()
