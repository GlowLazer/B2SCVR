# -*- coding: utf-8 -*-
"""
Batch validation: loads model ONCE, then loops over all validation videos.
Replaces vali.sh + test.py for faster full-dataset inference.
"""
import cv2
from PIL import Image
import numpy as np
import importlib
import os
from tqdm import tqdm
import torch

from core.utils import to_tensors

# ---- config ----
CKPT           = "checkpoints/B2SCVR_ckpts/B2SCVR.pth"
VIDEO_DIR      = "/media/ajeet/data/MINI_BTP/data/validation/bsc_imgs"
MASK_DIR       = "/media/ajeet/data/MINI_BTP/data/validation/masks"
OUT_DIR        = "./outputs"
MODEL_NAME     = "b2scvr_hq"
WIDTH, HEIGHT  = 432, 240
STEP           = 10
NUM_REF        = -1
NEIGHBOR_STRIDE = 5
FRAMESTRIDE    = 30

ref_length      = STEP
num_ref         = NUM_REF
neighbor_stride = NEIGHBOR_STRIDE


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


def read_mask(mpath, size):
    masks  = []
    mnames = sorted(os.listdir(mpath))
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


def resize_frames(frames, size=None):
    if size is not None:
        frames = [f.resize(size) for f in frames]
    else:
        size = frames[0].size
    return frames, size


def soft_boundary_blend(pred, original, binary_mask, blur_ksize=7, dilate_iter=1):
    mask   = binary_mask[:, :, 0].astype(np.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask   = cv2.dilate(mask, kernel, iterations=dilate_iter)
    soft   = cv2.GaussianBlur(mask, (blur_ksize, blur_ksize), 0)
    soft   = soft[:, :, np.newaxis]
    return (pred.astype(np.float32) * soft +
            original.astype(np.float32) * (1.0 - soft)).astype(np.uint8)


def run_inference_pass(model, device, rframes, rmasks, h, w):
    video_length = len(rframes)
    sum_frames   = [None] * video_length
    count_frames = [0]    * video_length

    x_frames = [rframes[i:i + FRAMESTRIDE] for i in range(0, video_length, FRAMESTRIDE)]
    x_masks  = [rmasks[i:i + FRAMESTRIDE]  for i in range(0, video_length, FRAMESTRIDE)]

    for itern in range(len(x_frames)):
        stride_length = len(x_frames[itern])
        imgs   = to_tensors()(x_frames[itern]).unsqueeze(0) * 2 - 1
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
            ref_ids        = get_ref_index(f, neighbor_ids, stride_length)
            selected_imgs  = imgs[:1, neighbor_ids + ref_ids, :, :, :]
            selected_masks = masks[:1, neighbor_ids + ref_ids, :, :, :]

            with torch.no_grad():
                mod_size_h, mod_size_w = 60, 108
                h_pad = (mod_size_h - h % mod_size_h) % mod_size_h
                w_pad = (mod_size_w - w % mod_size_w) % mod_size_w

                masked_imgs = selected_imgs * (1 - selected_masks)
                corrupted_contents = selected_imgs * selected_masks

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
                    gidx = itern * FRAMESTRIDE + idx
                    img  = soft_boundary_blend(pred_imgs[i], frames[idx], binary_masks[idx])
                    if sum_frames[gidx] is None:
                        sum_frames[gidx] = img.astype(np.float32)
                    else:
                        sum_frames[gidx] += img.astype(np.float32)
                    count_frames[gidx] += 1

    return sum_frames, count_frames


def process_video(model, device, video_name, video_path, mask_path):
    size   = (WIDTH, HEIGHT)
    rframes = read_frame_from_videos(video_path)
    rframes, size = resize_frames(rframes, size)
    h, w = size[1], size[0]
    video_length = len(rframes)
    rmasks = read_mask(mask_path, size)

    print('  Forward pass...')
    sum_fwd, cnt_fwd = run_inference_pass(model, device, rframes, rmasks, h, w)

    print('  Reverse-time TTA pass...')
    sum_bwd, cnt_bwd = run_inference_pass(
        model, device, list(reversed(rframes)), list(reversed(rmasks)), h, w)

    comp_frames = []
    for i in range(video_length):
        j   = video_length - 1 - i
        fwd = sum_fwd[i] / cnt_fwd[i] if sum_fwd[i] is not None else None
        bwd = sum_bwd[j] / cnt_bwd[j] if sum_bwd[j] is not None else None
        if fwd is not None and bwd is not None:
            comp_frames.append(((fwd + bwd) / 2).astype(np.uint8))
        elif fwd is not None:
            comp_frames.append(fwd.astype(np.uint8))
        elif bwd is not None:
            comp_frames.append(bwd.astype(np.uint8))
        else:
            comp_frames.append(None)

    new_comp_frames = [f for f in comp_frames if f is not None]
    imgs = [Image.fromarray(f) for f in new_comp_frames]

    save_path   = os.path.join(OUT_DIR, video_name)
    imwrite_path = os.path.join(save_path, 'frame_seq')
    os.makedirs(imwrite_path, exist_ok=True)
    for i, img in enumerate(imgs):
        img.save(os.path.join(imwrite_path, f'{i:05d}.jpg'))
    imgs[0].save(os.path.join(save_path, 'GIF_result.gif'),
                 save_all=True, append_images=imgs[1:], duration=40, loop=0)
    print(f'  Saved {len(imgs)} frames to: {save_path}')


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')

    net   = importlib.import_module('model.' + MODEL_NAME)
    model = net.InpaintGenerator(init_weights=False).to(device)
    data  = torch.load(CKPT, map_location=device)
    model.load_state_dict(data)
    print(f'Loaded model from: {CKPT}')
    model.eval()

    video_names = sorted([
        d for d in os.listdir(VIDEO_DIR)
        if os.path.isdir(os.path.join(VIDEO_DIR, d))
    ])
    total = len(video_names)
    print(f'Found {total} videos\n')

    os.makedirs(OUT_DIR, exist_ok=True)

    for current, video_name in enumerate(video_names, 1):
        video_path = os.path.join(VIDEO_DIR, video_name)
        mask_path  = os.path.join(MASK_DIR,  video_name)
        print(f'[{current}/{total}] {video_name}')
        process_video(model, device, video_name, video_path, mask_path)
        print(f'[{current}/{total}] Done: {video_name}\n')

    print(f'All {total} videos processed.')


if __name__ == '__main__':
    main()
