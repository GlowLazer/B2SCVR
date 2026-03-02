cd /media/ajeet/data/MINI_BTP/B2SCVR

python test.py \
    --ckpt /media/ajeet/data/MINI_BTP/B2SCVR/checkpoints/main_best.pth \
    --boundary_ckpt /media/ajeet/data/MINI_BTP/B2SCVR/checkpoints/boundary_head_best.pth \
    --video_dir /media/ajeet/data/MINI_BTP/data/validation/bsc_imgs \
    --mask_dir  /media/ajeet/data/MINI_BTP/data/validation/masks \
    --width 432 \
    --height 240
