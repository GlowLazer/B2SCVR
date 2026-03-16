python test.py \
  --ckpt checkpoints/psnr_run5/gen_best.pth \
  --boundary_ckpt checkpoints/boundary_head_best.pth \
  --refiner_ckpt  checkpoints/refiner_1500_best.pth \
  --lora_ckpt     checkpoints/lora_sam2_run3_best.pth \
  --lora_transformer_ckpt checkpoints/lora_transformer_on_sam2_best.pth \
  --video_dir /media/ajeet/data/MINI_BTP/data/ntire_test_set_25/bsc_imgs \
  --mask_dir  /media/ajeet/data/MINI_BTP/data/ntire_test_set_25/masks \
  --width 432 --height 240 \
  --out_dir outputs
 # --ckpt checkpoints/ema_cons_run1/gen_best.pt