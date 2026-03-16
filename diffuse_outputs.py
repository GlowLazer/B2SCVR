"""
Standalone diffusion post-refinement on already-generated outputs.

Reads frames from ./outputs/<video>/frame_seq/
Reads masks from --mask_dir/<video>/
Applies SD inpainting only where mask_ratio > --thresh
Saves refined frames to --out_dir/<video>/frame_seq/ (default: overwrites in-place)

PSNR-safe mode (recommended):
  --beta 0.10        residual blend weight (0=no diffusion, 1=full overwrite)
  --erode 3          erode mask before diffusion call to protect boundaries
  --strength 0.08    low noise = small, controlled edits
  --steps  30        more steps at low strength = cleaner result

Usage:
    python diffuse_outputs.py \
        --input_dir  ./outputs \
        --mask_dir   /media/ajeet/data/MINI_BTP/data/validation/masks \
        --out_dir    ./outputs_diffused \
        --strength   0.08 \
        --steps      30 \
        --beta       0.10 \
        --erode      3 \
        --thresh     0.15
"""

import argparse
import os
import cv2
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

p = argparse.ArgumentParser()
p.add_argument('--input_dir',  required=True,
               help='Directory containing per-video output subdirs (each with frame_seq/)')
p.add_argument('--mask_dir',   required=True,
               help='Directory containing per-video mask subdirs')
p.add_argument('--out_dir',    default=None,
               help='Output directory. If None, overwrites input_dir in-place.')
p.add_argument('--model',      default='runwayml/stable-diffusion-inpainting')
p.add_argument('--strength',   type=float, default=0.08,
               help='Noise strength (0=no change, 1=full generation). 0.05–0.12 for PSNR.')
p.add_argument('--steps',      type=int,   default=30,
               help='DDIM steps. More steps at low strength = cleaner result.')
p.add_argument('--beta',       type=float, default=0.10,
               help='Residual blend weight: out = comp + beta*(diff-comp)*mask. '
                    '0=keep comp, 1=full overwrite. 0.05–0.20 for PSNR safety.')
p.add_argument('--erode',      type=int,   default=3,
               help='Pixels to erode from mask before diffusion call (protects boundaries). '
                    '0=disabled.')
p.add_argument('--thresh',     type=float, default=0.15,
               help='Only refine frames where mask_ratio > thresh')
p.add_argument('--width',      type=int,   default=432)
p.add_argument('--height',     type=int,   default=240)
args = p.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
size   = (args.width, args.height)

# Load SD inpainting pipeline
from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler
print(f'Loading {args.model} ...')
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    args.model,
    torch_dtype=torch.float16,
    safety_checker=None,
    use_safetensors=True,
    variant="fp16",
).to(device)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.set_progress_bar_config(disable=True)
print(f'Model loaded.')
print(f'  strength={args.strength}  steps={args.steps}  beta={args.beta}  '
      f'erode={args.erode}  thresh={args.thresh}\n')


def read_mask_binary(mask_path):
    m = Image.open(mask_path).resize(size, Image.NEAREST).convert('L')
    return (np.array(m) > 0).astype(np.float32)


def erode_mask(mask_bin, pixels):
    """Erode binary mask by `pixels` pixels to pull back from boundaries."""
    if pixels <= 0:
        return mask_bin
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pixels * 2 + 1, pixels * 2 + 1))
    m_u8   = (mask_bin * 255).astype(np.uint8)
    return (cv2.erode(m_u8, kernel, iterations=1) > 0).astype(np.float32)


# Collect video names
video_names = sorted([
    v for v in os.listdir(args.input_dir)
    if os.path.isdir(os.path.join(args.input_dir, v, 'frame_seq'))
])
print(f'Videos to process: {len(video_names)}\n')

total_refined = 0
total_skipped = 0

for vi, video in enumerate(video_names, 1):
    frame_seq_dir = os.path.join(args.input_dir, video, 'frame_seq')
    mask_vid_dir  = os.path.join(args.mask_dir,   video)

    if not os.path.isdir(mask_vid_dir):
        print(f'[{vi}/{len(video_names)}] {video}: mask dir not found, skipping')
        continue

    frame_files = sorted([f for f in os.listdir(frame_seq_dir)
                          if f.lower().endswith(('.jpg', '.png'))])
    mask_files  = sorted([f for f in os.listdir(mask_vid_dir)
                          if f.lower().endswith(('.jpg', '.png'))])

    n = min(len(frame_files), len(mask_files))
    if n == 0:
        print(f'[{vi}/{len(video_names)}] {video}: no frames found, skipping')
        continue

    if args.out_dir:
        out_frame_dir = os.path.join(args.out_dir, video, 'frame_seq')
    else:
        out_frame_dir = frame_seq_dir
    os.makedirs(out_frame_dir, exist_ok=True)

    n_refined = 0
    n_skipped = 0

    for idx in tqdm(range(n), desc=f'[{vi}/{len(video_names)}] {video}', leave=False):
        frame_path = os.path.join(frame_seq_dir, frame_files[idx])
        mask_path  = os.path.join(mask_vid_dir,  mask_files[idx])

        mask_bin   = read_mask_binary(mask_path)
        mask_ratio = mask_bin.mean()

        comp_pil = Image.open(frame_path).resize(size).convert('RGB')
        out_path = os.path.join(out_frame_dir, frame_files[idx])

        if mask_ratio < args.thresh:
            if args.out_dir:
                comp_pil.save(out_path)
            n_skipped += 1
            continue

        # Erode mask for the diffusion call (protects boundary pixels)
        mask_for_diff = erode_mask(mask_bin, args.erode)
        if mask_for_diff.sum() < 1:
            # Erosion removed everything — skip diffusion, just copy
            if args.out_dir:
                comp_pil.save(out_path)
            n_skipped += 1
            continue

        mask_pil = Image.fromarray((mask_for_diff * 255).astype(np.uint8))

        with torch.no_grad():
            result = pipe(
                prompt='',
                image=comp_pil,
                mask_image=mask_pil,
                num_inference_steps=args.steps,
                strength=args.strength,
                guidance_scale=1.0,
                height=args.height,
                width=args.width,
            ).images[0]

        comp_np   = np.array(comp_pil).astype(np.float32)
        result_np = np.array(result).astype(np.float32)

        # Residual blend: out = comp + beta*(diff - comp)*mask
        # beta=1.0 → full overwrite inside mask (original behaviour)
        # beta<1.0 → partial correction, more PSNR-safe
        mask_3ch  = mask_bin[:, :, None]   # use original (un-eroded) mask for blend
        delta     = result_np - comp_np
        final_np  = (comp_np + args.beta * delta * mask_3ch).clip(0, 255).astype(np.uint8)

        Image.fromarray(final_np).save(out_path)
        n_refined += 1

    total_refined += n_refined
    total_skipped += n_skipped
    print(f'[{vi}/{len(video_names)}] {video}: '
          f'refined={n_refined}  skipped(below thresh)={n_skipped}')

print(f'\nDone.  Total refined: {total_refined}  skipped: {total_skipped}')
if args.out_dir:
    print(f'Results saved to: {args.out_dir}')
else:
    print('Frames refined in-place inside input_dir.')
