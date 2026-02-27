#!/bin/bash
set -e

cd /media/ajeet/data/MINI_BTP/B2SCVR

echo "========================================="
echo " Step 1/3: Running inference (vali.sh)"
echo "========================================="
bash vali.sh

echo ""
echo "========================================="
echo " Step 2/3: Building submission ZIP"
echo "========================================="
python make_submission.py

echo ""
echo "========================================="
echo " Step 3/3: Evaluating PSNR / SSIM"
echo "========================================="
python psnr.py

echo ""
echo "========================================="
echo " Done."
echo "========================================="
