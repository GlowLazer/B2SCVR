cd /media/ajeet/data/MINI_BTP/B2SCVR

python test.py \
    --ckpt /media/ajeet/data/MINI_BTP/B2SCVR/checkpoints/B2SCVR_ckpts/B2SCVR.pth \
    --boundary_ckpt /media/ajeet/data/MINI_BTP/B2SCVR/checkpoints/boundary_head_20000.pth \
    --video_dir /media/ajeet/data/MINI_BTP/data/validation/bsc_imgs \
    --mask_dir  /media/ajeet/data/MINI_BTP/data/validation/masks \
    --width 432 \
    --height 240
