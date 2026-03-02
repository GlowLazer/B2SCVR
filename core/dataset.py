import os
import glob
import json
import random

import cv2
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms

from core.utils import (TrainZipReader, TestZipReader,
                        create_random_shape_with_random_motion, Stack,
                        ToTorchFormatTensor, GroupRandomHorizontalFlip) 

import concurrent.futures

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, args: dict, debug=False):
        self.args = args
        self.num_local_frames = args['num_local_frames']
        self.num_ref_frames = args['num_ref_frames']
        self.size = self.w, self.h = (args['w'], args['h'])

        json_path = os.path.join(args['data_root'], args['name'], 'train.json')
        with open(json_path, 'r') as f:
            self.video_dict = json.load(f)
        self.video_names = list(self.video_dict.keys())
        if debug:
            self.video_names = self.video_names[:100]

        self._to_tensors = transforms.Compose([
            Stack(),
            ToTorchFormatTensor(),
        ])

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, index):
        item = self.load_item(index)
        return item
    

    def _sample_index(self, length, sample_length, video_name, num_ref_frame=3):
        complete_idx_set = list(range(length))
        pivot = random.randint(0, length - sample_length)
        local_idx = complete_idx_set[pivot:pivot + sample_length]
        remain_idx = list(set(complete_idx_set) - set(local_idx))
        ref_index = sorted(random.sample(remain_idx, num_ref_frame))

        return local_idx + ref_index

    def load_item(self, index):
        video_name = self.video_names[index]
        # create masks
        # all_masks = create_random_shape_with_random_motion(
        #     self.video_dict[video_name], imageHeight=self.h, imageWidth=self.w) 
        
        # create sample index
        selected_index = self._sample_index(self.video_dict[video_name],
                                            self.num_local_frames,
                                            video_name,
                                            self.num_ref_frames)

        # read video frames
        frames = []
        masks = []
        corrupts = []
        for idx in selected_index:
            video_path = os.path.join(self.args['data_root'],
                                      self.args['name'], 'JPEGImages',
                                      f'{video_name}.zip')
            corr_video_path = os.path.join(self.args['data_root'],
                                           self.args['name'], 'BSCJPEGImages',
                                             f'{video_name}.zip')
            gt_img = TrainZipReader.imread(video_path, idx).convert('RGB')
            corr_img = TrainZipReader.imread(corr_video_path, idx).convert('RGB')
            # if the imread result is None, then skip this frame
            if gt_img is None:
                continue
            if corr_img is None:
                continue
            gt_img = gt_img.resize(self.size)
            corr_img = corr_img.resize(self.size)
            # masks.append(all_masks[idx])
            
            mask_path = os.path.join(self.args['data_root'], 
                                     self.args['name'], 'train_mask', 
                                     video_name, str(idx).zfill(5) + '.png')
            if not os.path.exists(mask_path):
                continue
            
            mask = Image.open(mask_path).resize(self.size,
                                                Image.NEAREST).convert('L')

            # origin: 0 indicates missing. now: 1 indicates missing
            mask = np.asarray(mask)
            m = np.array(mask > 0).astype(np.uint8)
            m = cv2.dilate(m,
                        cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
                            iterations=4)
            mask = Image.fromarray(m*255)
            frames.append(gt_img)
            masks.append(mask)
            corrupts.append(corr_img)

        # normalizate, to tensors — apply ONE shared flip to keep gt/bsc/mask aligned
        if random.random() < 0.5:
            frames   = [f.transpose(Image.FLIP_LEFT_RIGHT) for f in frames]
            corrupts = [c.transpose(Image.FLIP_LEFT_RIGHT) for c in corrupts]
            masks    = [m.transpose(Image.FLIP_LEFT_RIGHT) for m in masks]
        frame_tensors = self._to_tensors(frames) * 2.0 - 1.0
        corrupt_tensors = self._to_tensors(corrupts) * 2.0 - 1.0
        mask_tensors = self._to_tensors(masks)
        return frame_tensors, mask_tensors, corrupt_tensors, video_name


class DirTrainDataset(torch.utils.data.Dataset):
    """Dir-format dataset: reads gt/bsc/mask from plain image directories.
    Returns (frames, masks, corrupts, video_name) matching TrainDataset interface.
    """

    def __init__(self, args: dict):
        self.gt_root       = args['gt_root']
        self.bsc_root      = args['bsc_root']
        self.mask_root     = args['mask_root']
        self.num_local     = args['num_local_frames']
        self.num_ref       = args['num_ref_frames']
        self.h, self.w     = args['h'], args['w']
        self.min_mask_ratio = args.get('min_mask_ratio', 0.0)
        total_needed       = self.num_local + self.num_ref

        videos = sorted(os.listdir(self.gt_root))
        self.videos      = []
        self.frame_count = {}
        for v in videos:
            frames = sorted(glob.glob(os.path.join(self.gt_root, v, '*.jpg')) +
                            glob.glob(os.path.join(self.gt_root, v, '*.png')))
            if len(frames) >= total_needed + 1:
                self.videos.append(v)
                self.frame_count[v] = len(frames)

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
                    gt = cv2.imread(os.path.join(self.gt_root,  video, stem + '.png'))
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

            n_local = min(self.num_local, len(mask_imgs))
            local_has_mask = any(mask_imgs[i].max() > 0 for i in range(n_local))
            local_mask_ratio = (sum(mask_imgs[i].mean() for i in range(n_local))
                                / (n_local * 255.0)) if n_local > 0 else 0.0
            passes_ratio = (self.min_mask_ratio == 0.0
                            or local_mask_ratio >= self.min_mask_ratio)
            if len(gt_imgs) >= needed and local_has_mask and passes_ratio:
                break
            idx = random.randint(0, len(self.videos) - 1)
        else:
            # 10 retries exhausted without meeting filter; hard-resample a new index
            if not (len(gt_imgs) >= needed and local_has_mask and passes_ratio):
                return self.__getitem__(random.randint(0, len(self.videos) - 1))

        # shared horizontal flip — keeps gt/bsc/mask aligned
        if random.random() < 0.5:
            gt_imgs   = [np.ascontiguousarray(img[:, ::-1, :]) for img in gt_imgs]
            bsc_imgs  = [np.ascontiguousarray(img[:, ::-1, :]) for img in bsc_imgs]
            mask_imgs = [np.ascontiguousarray(img[:, ::-1])    for img in mask_imgs]

        def to_rgb(imgs):
            arr = np.stack(imgs[:needed], 0).astype(np.float32) / 255.0  # (T,H,W,3)
            arr = arr.transpose(0, 3, 1, 2)                              # (T,3,H,W)
            return torch.from_numpy(arr) * 2.0 - 1.0                    # [-1,1]

        def to_mask(imgs):
            arr = (np.stack(imgs[:needed], 0).astype(np.float32) > 0).astype(np.float32)
            return torch.from_numpy(arr).unsqueeze(1)                   # (T,1,H,W)

        return to_rgb(gt_imgs), to_mask(mask_imgs), to_rgb(bsc_imgs), video


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        self.size = self.w, self.h = args.size

        with open(os.path.join(args.data_root, args.dataset, 'test.json'),
                  'r') as f:
            self.video_dict = json.load(f)
        self.video_names = list(self.video_dict.keys())

        self._to_tensors = transforms.Compose([
            Stack(),
            ToTorchFormatTensor(),
        ])

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, index):
        item = self.load_item(index)
        return item

    def load_item(self, index):
        video_name = self.video_names[index]
        ref_index = list(range(self.video_dict[video_name]))

        # read video frames
        frames = []
        masks = []
        for idx in ref_index:
            video_path = os.path.join(self.args.data_root, self.args.dataset,
                                      'JPEGImages_test', f'{video_name}.zip')
            img = TestZipReader.imread(video_path, idx).convert('RGB')
            img = img.resize(self.size)
            frames.append(img)
            mask_path = os.path.join(self.args.data_root, self.args.dataset,
                                     'test_masks', video_name,
                                     str(idx).zfill(5) + '.png')
            mask = Image.open(mask_path).resize(self.size,
                                                Image.NEAREST).convert('L')
            # origin: 0 indicates missing. now: 1 indicates missing
            mask = np.asarray(mask)
            m = np.array(mask > 0).astype(np.uint8)
            m = cv2.dilate(m,
                           cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
                           iterations=4)
            mask = Image.fromarray(m * 255)
            masks.append(mask)

        # to tensors
        frames_PIL = [np.array(f).astype(np.uint8) for f in frames]
        frame_tensors = self._to_tensors(frames) * 2.0 - 1.0
        mask_tensors = self._to_tensors(masks)
        return frame_tensors, mask_tensors, video_name, frames_PIL
