#!/bin/bash
cd /media/ajeet/data/MINI_BTP/B2SCVR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python tto_test.py --ckpt /media/ajeet/data/MINI_BTP/B2SCVR/checkpoints/ema_cons_run1/gen_best.pth --lora_ckpt /media/ajeet/data/MINI_BTP/B2SCVR/checkpoints/lora_sam2_best.pth --lora_transformer_ckpt /media/ajeet/data/MINI_BTP/B2SCVR/checkpoints/lora_transformer_on_sam2_best.pth --boundary_ckpt /media/ajeet/data/MINI_BTP/B2SCVR/checkpoints/boundary_head_best.pth --refiner_ckpt /media/ajeet/data/MINI_BTP/B2SCVR/checkpoints/refiner_best.pth --video_dir /media/ajeet/data/MINI_BTP/data/validation/bsc_imgs --mask_dir /media/ajeet/data/MINI_BTP/data/validation/masks --width 432 --height 240 --tta_steps 60 --tta_lr 2e-5 --min_mask_ratio 0.08
python make_submission.py --input_dir ./outputs --zip_name ./Vroom_tto.zip
python psnr.py --input_dir ./Vroom
