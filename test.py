# -*- coding: utf-8 -*-
import cv2
from PIL import Image
import numpy as np
import importlib
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import animation
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


def soft_boundary_blend(pred, original, binary_mask, blur_ksize=7, dilate_iter=1):
    """Replace hard mask composite with Gaussian-softened boundary blend."""
    mask = binary_mask[:, :, 0].astype(np.float32)          # (H, W)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.dilate(mask, kernel, iterations=dilate_iter)
    soft = cv2.GaussianBlur(mask, (blur_ksize, blur_ksize), 0)
    soft = soft[:, :, np.newaxis]                            # (H, W, 1)
    return (pred.astype(np.float32) * soft +
            original.astype(np.float32) * (1.0 - soft)).astype(np.uint8)


def run_inference_pass(model, device, rframes, rmasks, h, w, framestride):
    """Run one inference pass; returns per-frame float32 sum and hit count arrays."""
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
            ref_ids = get_ref_index(f, neighbor_ids, stride_length)
            selected_imgs  = imgs[:1, neighbor_ids + ref_ids, :, :, :]
            selected_masks = masks[:1, neighbor_ids + ref_ids, :, :, :]
            with torch.no_grad():
                masked_imgs        = selected_imgs * (1 - selected_masks)
                corrupted_contents = selected_imgs * selected_masks
                mod_size_h, mod_size_w = 60, 108
                h_pad = (mod_size_h - h % mod_size_h) % mod_size_h
                w_pad = (mod_size_w - w % mod_size_w) % mod_size_w

                masked_imgs = torch.cat(
                    [masked_imgs, torch.flip(masked_imgs, [3])],
                    3)[:, :, :, :h + h_pad, :]
                masked_imgs = torch.cat(
                    [masked_imgs, torch.flip(masked_imgs, [4])],
                    4)[:, :, :, :, :w + w_pad]
                corrupted_contents = torch.cat(
                    [corrupted_contents, torch.flip(corrupted_contents, [3])],
                    3)[:, :, :, :h + h_pad, :]
                corrupted_contents = torch.cat(
                    [corrupted_contents, torch.flip(corrupted_contents, [4])],
                    4)[:, :, :, :, :w + w_pad]

                pred_imgs, _ = model(selected_imgs, selected_imgs, selected_masks, len(neighbor_ids))
                pred_imgs = pred_imgs[:, :, :h, :w]
                pred_imgs = (pred_imgs + 1) / 2
                pred_imgs = pred_imgs.cpu().permute(0, 2, 3, 1).numpy() * 255

                for i in range(len(neighbor_ids)):
                    idx  = neighbor_ids[i]
                    gidx = itern * framestride + idx
                    img  = soft_boundary_blend(pred_imgs[i], frames[idx], binary_masks[idx])
                    if sum_frames[gidx] is None:
                        sum_frames[gidx] = img.astype(np.float32)
                    else:
                        sum_frames[gidx] += img.astype(np.float32)
                    count_frames[gidx] += 1

    return sum_frames, count_frames


def process_one_video(model, device, video_path, mask_path, size, framestride):
    rframes = read_frame_from_videos(video_path)
    rframes, size = resize_frames(rframes, size)
    h, w = size[1], size[0]
    video_length = len(rframes)
    rmasks = read_mask(mask_path, size)

    print('  Forward pass...')
    sum_fwd, cnt_fwd = run_inference_pass(model, device, rframes, rmasks, h, w, framestride)

    print('  Reverse-time TTA pass...')
    sum_bwd, cnt_bwd = run_inference_pass(
        model, device, list(reversed(rframes)), list(reversed(rmasks)), h, w, framestride)

    comp_frames = []
    for i in range(video_length):
        j   = video_length - 1 - i
        fwd = sum_fwd[i] / cnt_fwd[i] if sum_fwd[i] is not None else None
        bwd = sum_bwd[j] / cnt_bwd[j] if sum_bwd[j] is not None else None
        if fwd is not None and bwd is not None:
            alpha = 0.7
            comp_frames.append((alpha * fwd + (1 - alpha) * bwd).astype(np.uint8))
        elif fwd is not None:
            comp_frames.append(fwd.astype(np.uint8))
        elif bwd is not None:
            comp_frames.append(bwd.astype(np.uint8))
        else:
            comp_frames.append(None)

    save_base_name = os.path.basename(video_path.rstrip('/'))
    save_path   = os.path.join('./outputs', save_base_name)
    imwrite_path = os.path.join(save_path, 'frame_seq')
    os.makedirs(imwrite_path, exist_ok=True)
    new_comp_frames = [f for f in comp_frames if f is not None]
    imgs = [Image.fromarray(f) for f in new_comp_frames]
    for i, img in enumerate(imgs):
        img.save(os.path.join(imwrite_path, f'{i:05d}.jpg'))
    imgs[0].save(os.path.join(save_path, 'GIF_result.gif'),
                 save_all=True, append_images=imgs[1:], duration=40, loop=0)
    print(f'  Saved {len(imgs)} frames to: {save_path}')


def main_worker():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model ONCE
    net = importlib.import_module('model.' + args.model)
    model = net.InpaintGenerator(init_weights=False).to(device)
    data = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(data)
    print(f'Loading model from: {args.ckpt}')
    model.eval()

    size = (args.width, args.height)

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
