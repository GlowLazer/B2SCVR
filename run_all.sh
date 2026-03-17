#!/bin/bash
set -e

echo "========================================="
echo " Step 1/3: Running inference (vali.sh)"
echo "========================================="
bash vali.sh

echo ""
echo "========================================="
echo " Step 2/3: Post-processing outputs (run_model.py)"
echo "========================================="
python run_model.py

echo ""
echo "========================================="
echo " Step 3/3: Evaluating PSNR / SSIM"
echo "========================================="
python psnr.py

echo ""
echo "========================================="
echo " Done."
echo "========================================="

