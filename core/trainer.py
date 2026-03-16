import os
import glob
import logging
import importlib
import re
import csv
import numpy as np
from tqdm import tqdm

import cv2
from PIL import Image
from skimage.metrics import structural_similarity as ssim_func

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

from core.lr_scheduler import MultiStepRestartLR, CosineAnnealingRestartLR
from core.loss import AdversarialLoss
from core.dataset import TrainDataset, DirTrainDataset
from core.utils import to_tensors
from model.modules.flow_comp import FlowCompletionLoss
import torch.nn.functional as F

class Trainer:
    def __init__(self, config):
        self.config = config
        self.epoch = 0
        self.iteration = 0
        self.num_local_frames = config['train_data_loader']['num_local_frames']
        self.num_ref_frames = config['train_data_loader']['num_ref_frames']
        self.spynet_lr = config['trainer'].get('spynet_lr', 1.0)
        self.empty_mask_skip_count = 0

        # setup data set and data loader
        dl_cfg = config['train_data_loader']
        if dl_cfg.get('format') == 'dir':
            self.train_dataset = DirTrainDataset(dl_cfg)
        else:
            self.train_dataset = TrainDataset(dl_cfg)

        self.train_sampler = None
        self.train_args = config['trainer']
        self.train_args.setdefault('save_freq', 500)
        if config['distributed']:
            self.train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=config['world_size'],
                rank=config['global_rank'])

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.train_args['batch_size'] // config['world_size'],
            shuffle=(self.train_sampler is None),
            num_workers=self.train_args['num_workers'],
            sampler=self.train_sampler)

        # set loss functions
        self.adversarial_loss = AdversarialLoss(
            type=self.config['losses']['GAN_LOSS'])
        self.adversarial_loss = self.adversarial_loss.to(self.config['device'])
        self.l1_loss = nn.L1Loss()
        self.flow_comp_loss = FlowCompletionLoss().to(self.config['device'])

        # setup models including generator and discriminator
        net = importlib.import_module('model.' + config['model']['net'])
        self.netG = net.InpaintGenerator()
        
        self.netG = self.netG.to(self.config['device'])
        if not self.config['model']['no_dis']:
            self.netD = net.Discriminator(
                in_channels=3,
                use_sigmoid=config['losses']['GAN_LOSS'] != 'hinge')
            self.netD = self.netD.to(self.config['device'])

        # setup optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()
        self.load()

        if config['distributed']:
            self.netG = DDP(self.netG,
                            device_ids=[self.config['local_rank']],
                            output_device=self.config['local_rank'],
                            broadcast_buffers=True,
                            find_unused_parameters=True)
            if not self.config['model']['no_dis']:
                self.netD = DDP(self.netD,
                                device_ids=[self.config['local_rank']],
                                output_device=self.config['local_rank'],
                                broadcast_buffers=True,
                                find_unused_parameters=False)
        
        for param in self.netG.parameters():
            param.data = param.data.contiguous()

        self.use_ema = bool(self.train_args.get('use_ema', True))
        self.ema_decay = float(self.train_args.get('ema_decay', 0.999))
        self.ema_state = {}
        if self.use_ema:
            netG_ref = self.netG.module if isinstance(self.netG, (nn.DataParallel, DDP)) else self.netG
            for name, param in netG_ref.named_parameters():
                if param.requires_grad:
                    self.ema_state[name] = param.detach().clone()

        # ── mixed precision (optional) ───────────────────────────────────────
        device = self.config.get('device', None)
        device_type = getattr(device, "type", str(device))
        self.use_amp = bool(self.train_args.get('use_amp', False)) and (device_type == 'cuda')
        self.scaler = GradScaler(enabled=self.use_amp)

        # set summary writer
        self.dis_writer = None
        self.gen_writer = None
        self.summary = {}
        if self.config['global_rank'] == 0 or (not config['distributed']):
            self.dis_writer = SummaryWriter(
                os.path.join(config['save_dir'], 'dis'))
            self.gen_writer = SummaryWriter(
                os.path.join(config['save_dir'], 'gen'))

        # ── inline val-eval setup ─────────────────────────────────────────────
        self.best_psnr = -1.0
        val_cfg = config.get('val_data', {})
        self.val_video_dir = val_cfg.get('video_dir', '')
        self.val_mask_dir  = val_cfg.get('mask_dir',  '')
        self.val_gt_dir    = val_cfg.get('gt_dir',    '')
        self.val_w = config['train_data_loader'].get('w', 432)
        self.val_h = config['train_data_loader'].get('h', 240)
        self.val_enabled = bool(
            self.val_video_dir
            and os.path.isdir(self.val_video_dir)
            and os.path.isdir(self.val_mask_dir)
            and os.path.isdir(self.val_gt_dir))
        self.val_csv_path = os.path.join(config['save_dir'], 'val_psnr.csv')
        if self.val_enabled and not os.path.isfile(self.val_csv_path):
            with open(self.val_csv_path, 'w') as f:
                f.write('iteration,mean_psnr,mean_ssim\n')
        # Resume best_psnr from a previous run if available
        best_txt = os.path.join(config['save_dir'], 'best.txt')
        if os.path.isfile(best_txt):
            try:
                self.best_psnr = float(
                    open(best_txt).read().strip().split('psnr=')[1])
            except Exception:
                pass

        # ── train loss CSV ──────────────────────────────────────────────────
        self.train_csv_path = os.path.join(config['save_dir'], 'train_losses.csv')
        os.makedirs(config['save_dir'], exist_ok=True)
        if not os.path.isfile(self.train_csv_path):
            with open(self.train_csv_path, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow([
                    'iteration', 'lr',
                    'hole_loss', 'valid_loss', 'l1_stab', 'td_loss', 'gan_loss',
                    'total_gen', 'total_dis',
                    'mask_ratio', 'w_td', 'w_l1', 'w_gan',
                ])

    def setup_optimizers(self):
        """Set up optimizers."""
        backbone_params = []
        spynet_params = []
        prompt_params = []
        complete_params = []
        for name, param in self.netG.named_parameters():
            if 'update_spynet' in name:
                spynet_params.append(param)
            # elif 'prompt' in name or 'sam_fuser' in name or 'enhance' in name:
            elif 'prompt' in name or 'sam_fuser' in name:
                prompt_params.append(param)
            elif 'enhance' in name:
                complete_params.append(param)
            else:
                backbone_params.append(param)
                
        # freeze the flow completion and content recovery modules
        for param in backbone_params:
            param.requires_grad = False
        for param in spynet_params:
            param.requires_grad = False

        optim_params = [
            {   # HA and MoRE modules learning rate
                'params': prompt_params,
                'lr': self.config['trainer']['lr']
            },
            {  # finetuning learning rate for feature completion to adapt sam feature
                'params': complete_params,
                'lr': self.config['trainer']['lr'] * 0.1
            },
        ]

        self.optimG = torch.optim.Adam(optim_params,
                                       weight_decay=self.config['trainer'].get('weight_decay', 1e-4),
                                       betas=(self.config['trainer']['beta1'],
                                              self.config['trainer']['beta2']))

        if not self.config['model']['no_dis']:
            self.optimD = torch.optim.Adam(
                self.netD.parameters(),
                lr=self.config['trainer']['lr'],
                betas=(self.config['trainer']['beta1'],
                       self.config['trainer']['beta2']))

    def setup_schedulers(self):
        """Set up schedulers."""
        scheduler_opt = self.config['trainer']['scheduler']
        scheduler_type = scheduler_opt.pop('type')

        if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR']:
            self.scheG = MultiStepRestartLR(
                self.optimG,
                milestones=scheduler_opt['milestones'],
                gamma=scheduler_opt['gamma'])
            if not self.config['model']['no_dis']:
                self.scheD = MultiStepRestartLR(
                    self.optimD,
                    milestones=scheduler_opt['milestones'],
                    gamma=scheduler_opt['gamma'])
        elif scheduler_type == 'CosineAnnealingRestartLR':
            self.scheG = CosineAnnealingRestartLR(
                self.optimG,
                periods=scheduler_opt['periods'],
                restart_weights=scheduler_opt['restart_weights'])
            if not self.config['model']['no_dis']:
                self.scheD = CosineAnnealingRestartLR(
                    self.optimD,
                    periods=scheduler_opt['periods'],
                    restart_weights=scheduler_opt['restart_weights'])
        else:
            raise NotImplementedError(
                f'Scheduler {scheduler_type} is not implemented yet.')

    def update_learning_rate(self):
        """Update learning rate."""
        self.scheG.step()
        if not self.config['model']['no_dis']:
            self.scheD.step()

    def get_lr(self):
        """Get current learning rate."""
        return self.optimG.param_groups[0]['lr']

    def add_summary(self, writer, name, val):
        """Add tensorboard summary."""
        if name not in self.summary:
            self.summary[name] = 0
        self.summary[name] += val
        if writer is not None and self.iteration % 100 == 0:
            writer.add_scalar(name, self.summary[name] / 100, self.iteration)
            self.summary[name] = 0

    def _update_ema(self):
        if not self.use_ema:
            return
        netG_ref = self.netG.module if isinstance(self.netG, (nn.DataParallel, DDP)) else self.netG
        with torch.no_grad():
            for name, param in netG_ref.named_parameters():
                if not param.requires_grad or name not in self.ema_state:
                    continue
                self.ema_state[name].mul_(self.ema_decay).add_(
                    param.detach(), alpha=1.0 - self.ema_decay)

    def _build_ema_state_dict(self, netG_ref):
        ema_dict = {k: v.detach().clone() for k, v in netG_ref.state_dict().items()}
        for name, tensor in self.ema_state.items():
            if name in ema_dict:
                ema_dict[name] = tensor.detach().to(
                    device=ema_dict[name].device, dtype=ema_dict[name].dtype).clone()
        return ema_dict

    def _swap_in_ema(self, netG_ref):
        if not self.use_ema:
            return {}
        backup = {}
        with torch.no_grad():
            for name, param in netG_ref.named_parameters():
                if not param.requires_grad or name not in self.ema_state:
                    continue
                backup[name] = param.data.detach().clone()
                param.data.copy_(self.ema_state[name].to(device=param.device, dtype=param.dtype))
        return backup

    def _restore_from_backup(self, netG_ref, backup):
        if not backup:
            return
        with torch.no_grad():
            for name, param in netG_ref.named_parameters():
                if name in backup:
                    param.data.copy_(backup[name].to(device=param.device, dtype=param.dtype))

    def _cleanup_old_checkpoints(self):
        keep_last_k = int(self.train_args.get('keep_last_k', 0))
        if keep_last_k <= 0:
            return
        save_dir = self.config['save_dir']
        ids = []
        for p in glob.glob(os.path.join(save_dir, 'gen_*.pth')):
            m = re.match(r'^gen_(\d{6})\.pth$', os.path.basename(p))
            if m:
                ids.append(int(m.group(1)))
        ids = sorted(set(ids))
        if len(ids) <= keep_last_k:
            return
        for old_id in ids[:-keep_last_k]:
            for name in (
                f'gen_{old_id:06d}.pth',
                f'gen_{old_id:06d}_ema.pth',
                f'opt_{old_id:06d}.pth',
                f'dis_{old_id:06d}.pth',
            ):
                path = os.path.join(save_dir, name)
                if os.path.isfile(path):
                    os.remove(path)

    def load(self):
        """Load netG (and netD)."""
        # get the latest checkpoint
        model_path = self.config['save_dir']
        if os.path.isfile(os.path.join(model_path, 'latest.ckpt')):
            latest_epoch = open(os.path.join(model_path, 'latest.ckpt'),
                                'r').read().splitlines()[-1]
        else:
            ckpts = sorted(glob.glob(os.path.join(model_path, 'gen_[0-9]*.pth')))
            # strip _ema suffix and extract iteration number
            iters = []
            for c in ckpts:
                base = os.path.basename(c).replace('_ema', '').split('.pth')[0]
                try:
                    iters.append(int(base.replace('gen_', '')))
                except ValueError:
                    pass
            latest_epoch = str(max(iters)) if iters else None

        if latest_epoch is not None:
            gen_path = os.path.join(model_path,
                                    f'gen_{int(latest_epoch):06d}.pth')
            dis_path = os.path.join(model_path,
                                    f'dis_{int(latest_epoch):06d}.pth')
            opt_path = os.path.join(model_path,
                                    f'opt_{int(latest_epoch):06d}.pth')

            if self.config['global_rank'] == 0:
                print(f'Loading model from {gen_path}...')
            dataG = torch.load(gen_path, map_location=self.config['device'])
            self.netG.load_state_dict(dataG)
            if not self.config['model']['no_dis']:
                dataD = torch.load(dis_path,
                                   map_location=self.config['device'])
                self.netD.load_state_dict(dataD)

            data_opt = torch.load(opt_path, map_location=self.config['device'])
            self.optimG.load_state_dict(data_opt['optimG'])
            self.scheG.load_state_dict(data_opt['scheG'])
            if not self.config['model']['no_dis']:
                self.optimD.load_state_dict(data_opt['optimD'])
                self.scheD.load_state_dict(data_opt['scheD'])
            self.epoch = data_opt['epoch']
            self.iteration = data_opt['iteration']

        else:
            gen_path = self.config['model'].get('pretrain', '')
            if gen_path and os.path.isfile(gen_path):
                if self.config['global_rank'] == 0:
                    print(f'Loading pretrain model from {gen_path}...')
                dataG = torch.load(gen_path, map_location=self.config['device'])
                self.netG.load_state_dict(dataG, strict=False)
            else:
                if self.config['global_rank'] == 0:
                    print('Warning: No pretrain model found. Using random initialisation.')

    def save(self, it):
        """Save parameters every eval_epoch"""
        if self.config['global_rank'] == 0:
            # configure path
            gen_path = os.path.join(self.config['save_dir'],
                                    f'gen_{it:06d}.pth')
            gen_ema_path = os.path.join(self.config['save_dir'],
                                        f'gen_{it:06d}_ema.pth')
            dis_path = os.path.join(self.config['save_dir'],
                                    f'dis_{it:06d}.pth')
            opt_path = os.path.join(self.config['save_dir'],
                                    f'opt_{it:06d}.pth')
            print(f'\nsaving model to {gen_path} ...')

            # remove .module for saving
            if isinstance(self.netG, torch.nn.DataParallel) \
               or isinstance(self.netG, DDP):
                netG = self.netG.module
                if not self.config['model']['no_dis']:
                    netD = self.netD.module
            else:
                netG = self.netG
                if not self.config['model']['no_dis']:
                    netD = self.netD

            # save checkpoints
            torch.save(netG.state_dict(), gen_path)
            if self.use_ema and self.ema_state:
                torch.save(self._build_ema_state_dict(netG), gen_ema_path)
            if not self.config['model']['no_dis']:
                torch.save(netD.state_dict(), dis_path)
                torch.save(
                    {
                        'epoch': self.epoch,
                        'iteration': self.iteration,
                        'optimG': self.optimG.state_dict(),
                        'optimD': self.optimD.state_dict(),
                        'scheG': self.scheG.state_dict(),
                        'scheD': self.scheD.state_dict()
                    }, opt_path)
            else:
                torch.save(
                    {
                        'epoch': self.epoch,
                        'iteration': self.iteration,
                        'optimG': self.optimG.state_dict(),
                        'scheG': self.scheG.state_dict()
                    }, opt_path)

            latest_path = os.path.join(self.config['save_dir'], 'latest.ckpt')
            os.system(f"echo {it:06d} > {latest_path}")
            self._cleanup_old_checkpoints()

    def _val_quick(self, it):
        """
        Forward-only (no TTA, no boundary head) PSNR eval on val_video_dir.
        Saves gen_best.pth if PSNR improves. Appends to val_psnr.csv.
        Called on rank-0 only, right after self.save().
        """
        if not self.val_enabled:
            return

        device = self.config['device']
        w, h   = self.val_w, self.val_h
        val_cfg = self.config.get('val_data', {})
        use_train_frame_counts = bool(val_cfg.get('use_train_frame_counts', True))
        val_use_amp = bool(val_cfg.get('use_amp', self.use_amp)) and getattr(device, "type", str(device)) == 'cuda'

        # Keep val-time memory bounded by matching the frame counts used in training by default.
        # The previous sliding-window settings could use many more frames per forward pass,
        # which is much more memory-hungry (especially for attention/softmax).
        NEIGHBOR_STRIDE = int(val_cfg.get('neighbor_stride', 5))
        REF_STEP        = int(val_cfg.get('ref_step', 10))
        FRAMESTRIDE     = int(val_cfg.get('frame_stride', 30))
        if use_train_frame_counts:
            VAL_LOCAL_FRAMES = int(val_cfg.get('num_local_frames', self.num_local_frames))
            VAL_REF_FRAMES   = int(val_cfg.get('num_ref_frames', self.num_ref_frames))

        if getattr(device, "type", str(device)) == 'cuda':
            torch.cuda.empty_cache()

        netG = self.netG.module if isinstance(self.netG, (nn.DataParallel, DDP)) else self.netG
        ema_backup = self._swap_in_ema(netG)
        netG.eval()

        videos = sorted([d for d in os.listdir(self.val_video_dir)
                         if os.path.isdir(os.path.join(self.val_video_dir, d))])
        all_psnrs = []
        all_ssims = []

        with torch.inference_mode():
            for video in videos:
                vpath = os.path.join(self.val_video_dir, video)
                mpath = os.path.join(self.val_mask_dir,  video)
                gpath = os.path.join(self.val_gt_dir,    video)
                if not os.path.isdir(mpath) or not os.path.isdir(gpath):
                    continue

                # Read BSC frames
                rframes = []
                for name in sorted(os.listdir(vpath)):
                    img = cv2.imread(os.path.join(vpath, name))
                    if img is None:
                        continue
                    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).resize((w, h))
                    rframes.append(img)
                if not rframes:
                    continue

                # Read masks (same pre-processing as test.py)
                rmasks = []
                for mp in sorted(os.listdir(mpath)):
                    m = Image.open(os.path.join(mpath, mp)).resize((w, h), Image.NEAREST)
                    m = np.array(m.convert('L'))
                    m = np.array(m > 0).astype(np.uint8)
                    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
                                   iterations=4)
                    rmasks.append(Image.fromarray(m * 255))
                if not rmasks:
                    continue

                video_length = len(rframes)
                sum_frames   = [None] * video_length
                count_frames = [0]    * video_length

                x_frames = [rframes[i:i + FRAMESTRIDE] for i in range(0, video_length, FRAMESTRIDE)]
                x_masks  = [rmasks[i:i + FRAMESTRIDE]  for i in range(0, video_length, FRAMESTRIDE)]

                for itern in range(len(x_frames)):
                    stride_length = len(x_frames[itern])
                    imgs  = to_tensors()(x_frames[itern]).unsqueeze(0) * 2 - 1
                    raw_f = [np.array(f).astype(np.uint8) for f in x_frames[itern]]
                    bmasks = [
                        np.expand_dims((np.array(m) != 0).astype(np.uint8), 2)
                        for m in x_masks[itern]
                    ]
                    masks = to_tensors()(x_masks[itern]).unsqueeze(0)
                    imgs, masks = imgs.to(device), masks.to(device)

                    for f in range(0, stride_length, NEIGHBOR_STRIDE):
                        if use_train_frame_counts:
                            local_n = min(VAL_LOCAL_FRAMES, stride_length)
                            half = local_n // 2
                            start = max(0, min(f - half, stride_length - local_n))
                            neighbor_ids = list(range(start, start + local_n))

                            ref_candidates = [i for i in range(0, stride_length, max(1, REF_STEP))
                                              if i not in neighbor_ids]
                            if len(ref_candidates) < VAL_REF_FRAMES:
                                ref_candidates += [i for i in range(stride_length) if i not in neighbor_ids]
                            ref_ids = ref_candidates[:VAL_REF_FRAMES]
                        else:
                            neighbor_ids = [i for i in range(
                                max(0, f - NEIGHBOR_STRIDE),
                                min(stride_length, f + NEIGHBOR_STRIDE + 1))]
                            ref_ids = [i for i in range(0, stride_length, REF_STEP)
                                       if i not in neighbor_ids]

                        sel_imgs  = imgs[:1, neighbor_ids + ref_ids]
                        sel_masks = masks[:1, neighbor_ids + ref_ids]

                        with autocast(enabled=val_use_amp, dtype=torch.float16):
                            pred_out, _ = netG(sel_imgs, sel_imgs, sel_masks, len(neighbor_ids))
                            pred_out = pred_out[:, :, :h, :w]
                            pred_out = (pred_out + 1) / 2
                            pred_np  = pred_out.float().cpu().permute(0, 2, 3, 1).numpy() * 255

                        for i, idx in enumerate(neighbor_ids):
                            gidx   = itern * FRAMESTRIDE + idx
                            mask2d = bmasks[idx][:, :, 0:1].astype(np.float32)
                            img    = (pred_np[i].astype(np.float32) * mask2d +
                                      raw_f[idx].astype(np.float32) * (1.0 - mask2d)).astype(np.uint8)
                            if sum_frames[gidx] is None:
                                sum_frames[gidx] = img.astype(np.float32)
                            else:
                                sum_frames[gidx] += img.astype(np.float32)
                            count_frames[gidx] += 1

                        del sel_imgs, sel_masks, pred_out

                    del imgs, masks
                    if getattr(device, "type", str(device)) == 'cuda':
                        torch.cuda.empty_cache()

                # Compute masked PSNR on corrupted frames only (PSNR-aligned with training)
                vid_psnrs = []
                vid_ssims = []
                for i, frame in enumerate(sum_frames):
                    if frame is None:
                        continue
                    comp = (frame / count_frames[i]).astype(np.uint8)

                    mask_path = os.path.join(mpath, f'{i:05d}.png')
                    mask_raw = None
                    if os.path.exists(mask_path):
                        mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if mask_raw is None or not np.any(mask_raw > 0):
                        continue

                gt_path = None
                for ext in ('.jpg', '.png'):
                    cand = os.path.join(gpath, f'{i:05d}{ext}')
                    if os.path.exists(cand):
                        gt_path = cand
                        break
                if gt_path is None:
                    continue

                gt_bgr = cv2.imread(gt_path)
                if gt_bgr is None:
                    continue

                pred_bgr = cv2.cvtColor(comp, cv2.COLOR_RGB2BGR)
                if pred_bgr.shape[:2] != gt_bgr.shape[:2]:
                    pred_bgr = cv2.resize(pred_bgr,
                                          (gt_bgr.shape[1], gt_bgr.shape[0]),
                                          interpolation=cv2.INTER_CUBIC)
                pred_f = cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2RGB).astype(np.float64)
                gt_f   = cv2.cvtColor(gt_bgr,   cv2.COLOR_BGR2RGB).astype(np.float64)

                # mask -> GT resolution, apply same dilation used elsewhere
                m = (mask_raw > 0).astype(np.uint8)
                if m.shape[:2] != gt_bgr.shape[:2]:
                    m = cv2.resize(m,
                                   (gt_bgr.shape[1], gt_bgr.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
                mse  = float(np.mean((pred_f - gt_f) ** 2))
                psnr   = 20.0 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')
                vid_psnrs.append(psnr)

                # SSIM is computed on the full RGB frame (still only on corrupted frames)
                try:
                    vid_ssims.append(float(ssim_func(pred_f, gt_f, data_range=255.0, channel_axis=2)))
                except Exception:
                    pass

                if vid_psnrs:
                    all_psnrs.append(float(np.mean(vid_psnrs)))
                if vid_ssims:
                    all_ssims.append(float(np.mean(vid_ssims)))

        self._restore_from_backup(netG, ema_backup)
        netG.train()

        if not all_psnrs:
            return

        mean_psnr = float(np.mean(all_psnrs))
        mean_ssim = float(np.mean(all_ssims)) if all_ssims else 0.0

        # CSV log (backward compatible with older 2-col files)
        want_ssim = True
        if os.path.isfile(self.val_csv_path):
            try:
                header = open(self.val_csv_path, 'r').readline().strip().lower()
                want_ssim = ('ssim' in header)
            except Exception:
                want_ssim = False
        with open(self.val_csv_path, 'a') as f:
            if want_ssim:
                f.write(f'{it},{mean_psnr:.4f},{mean_ssim:.6f}\n')
            else:
                f.write(f'{it},{mean_psnr:.4f}\n')

        # Tensorboard
        if self.gen_writer is not None:
            self.gen_writer.add_scalar('val/psnr', mean_psnr, it)

        # Save gen_best.pth if improved
        is_best = mean_psnr > self.best_psnr + 1e-4
        if is_best:
            self.best_psnr = mean_psnr
            best_path = os.path.join(self.config['save_dir'], 'gen_best.pth')
            if self.use_ema and self.ema_state:
                torch.save(self._build_ema_state_dict(netG), best_path)
            else:
                torch.save(netG.state_dict(), best_path)
            with open(os.path.join(self.config['save_dir'], 'best.txt'), 'w') as f:
                f.write(f'iter={it}  psnr={mean_psnr:.4f}\n')

        marker = '  ← NEW BEST' if is_best else ''
        print(f'\n[Val @ {it}] mean PSNR: {mean_psnr:.4f} dB  '
              f'mean SSIM: {mean_ssim:.4f}  (best: {self.best_psnr:.4f} dB){marker}')

    def train(self):
        """training entry"""
        pbar = range(int(self.train_args['iterations']))
        if self.config['global_rank'] == 0:
            pbar = tqdm(pbar,
                        initial=self.iteration,
                        dynamic_ncols=True,
                        smoothing=0.01)

        os.makedirs('logs', exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(filename)s[line:%(lineno)d]"
            "%(levelname)s %(message)s",
            datefmt="%a, %d %b %Y %H:%M:%S",
            filename=f"logs/{self.config['save_dir'].split('/')[-1]}.log",
            filemode='w')

        while True:
            self.epoch += 1
            if self.config['distributed']:
                self.train_sampler.set_epoch(self.epoch)

            self._train_epoch(pbar)
            if self.iteration > self.train_args['iterations']:
                break
        print('\nEnd training....')

    def _train_epoch(self, pbar):
        """Process input and calculate loss every training epoch"""
        device = self.config['device']
        # print trainable parameters
        for name, param in self.netG.named_parameters():
            if param.requires_grad:
                print(name)

        for frames, masks, corrupts, _ in self.train_loader:
            frames, masks, corrupts = frames.to(device), masks.to(device), corrupts.to(device)
            
            # in the first iterations, we freeze the discriminator
            if not self.config['model']['no_dis']:
                if self.iteration < 1:
                    for param in self.netD.parameters():
                        param.requires_grad = False
                else:
                    for param in self.netD.parameters():
                        param.requires_grad = True
                        # re-init the optimizer
                        self.optimD = torch.optim.Adam(
                            self.netD.parameters(),
                            lr=self.config['trainer']['lr'],
                            betas=(self.config['trainer']['beta1'],
                                self.config['trainer']['beta2']))

            # Skip truly empty-mask batches and track how often this occurs.
            if masks.sum() < 1.0:
                self.empty_mask_skip_count += 1
                if self.config['global_rank'] == 0 and self.empty_mask_skip_count % 50 == 0:
                    logging.info(f"[EmptyMaskSkip] count={self.empty_mask_skip_count}")
                continue
            
            l_t = self.num_local_frames
            b, t, c, h, w = frames.size()
            
            with autocast(enabled=self.use_amp, dtype=torch.float16):
                pred_imgs, _ = self.netG(frames, corrupts, masks, l_t)
                pred_imgs = pred_imgs.view(b, -1, c, h, w)
                comp_imgs = corrupts * (1. - masks) + masks * pred_imgs

            # ── Temporal difference loss ──────────────────────────────────────
            # w_td ramp: 0 for iter<500, linear 0→td_max for iter 500-2000, td_max after
            # Set losses.td_weight: 0.0 in config to disable TD entirely (e.g. specialist)
            td_max = self.config['losses'].get('td_weight', 0.1)
            it = self.iteration
            if td_max == 0.0 or it < 500:
                w_td = 0.0
            elif it < 2000:
                w_td = td_max * (it - 500) / 1500.0
            else:
                w_td = td_max

            if w_td > 0.0 and l_t >= 2:
                comp_loc   = comp_imgs[:, :l_t]              # (B, l_t, 3, H, W)
                gt_loc     = frames[:, :l_t]
                m_loc      = masks[:, :l_t]                  # (B, l_t, 1, H, W)

                d_comp     = comp_loc[:, 1:] - comp_loc[:, :-1]   # (B, l_t-1, 3, H, W)
                d_gt       = gt_loc[:, 1:]   - gt_loc[:, :-1]

                union      = (m_loc[:, 1:] + m_loc[:, :-1]).clamp(0.0, 1.0)
                union_flat = union.reshape(b * (l_t - 1), 1, h, w)
                w_flat     = F.max_pool2d(union_flat, kernel_size=3, stride=1, padding=1)
                w_mask     = w_flat.reshape(b, l_t - 1, 1, h, w)

                td_err     = (d_comp - d_gt).abs() * w_mask
                td_loss    = td_err.float().mean() / (w_mask.float().mean() + 1e-6)
                td_loss_item = td_loss.item()
            else:
                td_loss      = torch.tensor(0.0).to(device)
                td_loss_item = 0.0

            mask_ratio = masks[:, :l_t].mean().item()

            gen_loss = 0
            dis_loss = 0

            if not self.config['model']['no_dis']:
                if self.iteration < 1:
                    dis_real_loss = torch.tensor(0).to(device)
                    dis_fake_loss = torch.tensor(0).to(device)
                    dis_loss = torch.tensor(0).to(device)
                    gan_loss = torch.tensor(0).to(device)
                    pass
                else:
                    # discriminator adversarial loss
                    with autocast(enabled=self.use_amp, dtype=torch.float16):
                        real_clip = self.netD(frames)
                        fake_clip = self.netD(comp_imgs.detach())
                    dis_real_loss = self.adversarial_loss(real_clip, True, True)
                    dis_fake_loss = self.adversarial_loss(fake_clip, False, True)
                    dis_loss += (dis_real_loss + dis_fake_loss) / 2       
                    
                    self.optimD.zero_grad()
                    if self.use_amp:
                        self.scaler.scale(dis_loss).backward()
                        self.scaler.step(self.optimD)
                    else:
                        dis_loss.backward()
                        self.optimD.step()

                    # generator adversarial loss
                    with autocast(enabled=self.use_amp, dtype=torch.float16):
                        gen_clip = self.netD(comp_imgs)
                    gan_loss = self.adversarial_loss(gen_clip, True, False)
                    gan_loss = gan_loss \
                        * self.config['losses']['adversarial_weight']
                    gen_loss += gan_loss

            # masked MSE reconstruction losses (PSNR-aligned)
            eps        = 1e-6
            diff       = (pred_imgs - frames).float()
            sq         = diff ** 2
            masks_f    = masks.float()

            hole_loss  = (sq * masks_f).mean()       / (masks_f.mean()        + eps) \
                         * self.config['losses']['hole_weight']
            valid_loss = (sq * (1 - masks_f)).mean() / ((1-masks_f).mean()    + eps) \
                         * self.config['losses']['valid_weight']
            w_l1       = float(self.config['losses'].get('l1_weight', 0.1))
            l1_stab    = torch.mean(torch.abs(diff))   # L1 stabilizer (weighted separately)

            gen_loss += hole_loss
            gen_loss += valid_loss
            gen_loss += w_l1 * l1_stab
            gen_loss += w_td * td_loss

            self.optimG.zero_grad()
            if self.use_amp:
                self.scaler.scale(gen_loss).backward()
                self.scaler.step(self.optimG)
                self.scaler.update()
            else:
                gen_loss.backward()
                self.optimG.step()
            self._update_ema()
            
            if not self.config['model']['no_dis']:
                self.add_summary(self.dis_writer, 'loss/dis_vid_fake',
                                    dis_fake_loss.item())
                self.add_summary(self.dis_writer, 'loss/dis_vid_real',
                                    dis_real_loss.item())
                self.add_summary(self.gen_writer, 'loss/gan_loss',
                                    gan_loss.item())
            self.add_summary(self.gen_writer, 'loss/hole_loss',
                             hole_loss.item())
            self.add_summary(self.gen_writer, 'loss/valid_loss',
                             valid_loss.item())
            self.add_summary(self.gen_writer, 'loss/l1_stab',
                             (w_l1 * l1_stab).item())
            self.add_summary(self.gen_writer, 'loss/td_loss', td_loss_item)
            self.add_summary(self.gen_writer, 'stat/mask_ratio', mask_ratio)
            self.add_summary(self.gen_writer, 'stat/w_td', w_td)
            self.add_summary(self.gen_writer, 'stat/w_l1', w_l1)
            if not self.config['model']['no_dis']:
                self.add_summary(self.gen_writer, 'stat/w_gan',
                                 float(self.config['losses'].get('adversarial_weight', 0.0)))
            
            self.update_learning_rate()      
            self.iteration += 1
            
            # console logs
            if self.config['global_rank'] == 0:
                pbar.update(1)
                if not self.config['model']['no_dis']:
                    pbar.set_description((f"d: {dis_loss.item():.3f}; "
                                          f"hole: {hole_loss.item():.3f}; "
                                          f"valid: {valid_loss.item():.3f}; "
                                          f"td: {td_loss_item:.4f}; "
                                          f"mask: {mask_ratio:.3f}"
                                         ))
                else:
                    pbar.set_description((f"hole: {hole_loss.item():.3f}; "
                                          f"valid: {valid_loss.item():.3f}; "
                                          f"td: {td_loss_item:.4f}; "
                                          f"mask: {mask_ratio:.3f}"
                                         ))

                if self.iteration % self.train_args['log_freq'] == 0:
                    # CSV log (every log_freq; avoids per-iter I/O)
                    gan_item = gan_loss.item() if (not self.config['model']['no_dis']) else 0.0
                    dis_item = dis_loss.item() if (not self.config['model']['no_dis']) else 0.0
                    lr_item  = self.get_lr()
                    w_gan    = float(self.config['losses'].get('adversarial_weight', 0.0))
                    with open(self.train_csv_path, 'a', newline='') as f:
                        w = csv.writer(f)
                        w.writerow([
                            int(self.iteration), float(lr_item),
                            float(hole_loss.item()), float(valid_loss.item()),
                            float((w_l1 * l1_stab).item()), float(td_loss_item), float(gan_item),
                            float(gen_loss.item()), float(dis_item),
                            float(mask_ratio), float(w_td), float(w_l1), float(w_gan),
                        ])
                    if not self.config['model']['no_dis']:
                        logging.info(f"[Iter {self.iteration}] "
                                     f"d: {dis_loss.item():.4f}; "
                                     f"hole: {hole_loss.item():.4f}; "
                                     f"valid: {valid_loss.item():.4f}; "
                                     f"l1: {(w_l1 * l1_stab).item():.4f}; "
                                     f"td: {td_loss_item:.4f}; "
                                     f"mask_ratio: {mask_ratio:.3f}; "
                                     f"w_td: {w_td:.4f}; "
                                     f"w_l1: {w_l1:.3f}; "
                                     f"w_gan: {float(self.config['losses'].get('adversarial_weight', 0.0)):.3f}"
                                        )
                    else:
                        logging.info(f"[Iter {self.iteration}] "
                                     f"hole: {hole_loss.item():.4f}; "
                                     f"valid: {valid_loss.item():.4f}; "
                                     f"l1: {(w_l1 * l1_stab).item():.4f}; "
                                     f"td: {td_loss_item:.4f}; "
                                     f"mask_ratio: {mask_ratio:.3f}; "
                                     f"w_td: {w_td:.4f}; "
                                     f"w_l1: {w_l1:.3f}"
                                     )

            # saving models + inline val eval
            if self.iteration % self.train_args['save_freq'] == 0:
                self.save(int(self.iteration))
                if self.config['global_rank'] == 0:
                    self._val_quick(int(self.iteration))

            if self.iteration > self.train_args['iterations']:
                break
