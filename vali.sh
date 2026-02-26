cd /media/ajeet/data/MINI_BTP/B2SCVR

python test.py \
    --ckpt checkpoints/B2SCVR_ckpts/B2SCVR.pth \
    --video_dir /media/ajeet/data/MINI_BTP/data/validation/bsc_imgs \
    --mask_dir  /media/ajeet/data/MINI_BTP/data/validation/masks \
    --width 432 \
    --height 240
