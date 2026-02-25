cd /media/ajeet/data/MINI_BTP/B2SCVR

# Count total videos upfront
video_dirs=(/media/ajeet/data/MINI_BTP/data/validation/bsc_imgs/*/)
total=${#video_dirs[@]}
current=0

# Loop through each video folder
for video_dir in "${video_dirs[@]}"; do
    video_name=$(basename "$video_dir")
    current=$((current + 1))
    echo "[$current/$total] Processing $video_name..."

    python test.py \
        --ckpt checkpoints/B2SCVR_ckpts/B2SCVR.pth \
        -v "$video_dir" \
        --dac_mask /media/ajeet/data/MINI_BTP/data/validation/masks/$video_name \
        --width 432 \
        --height 240

    echo "[$current/$total] ✓ Done: $video_name"
done
echo "All $total videos processed."
