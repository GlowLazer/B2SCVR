# analyze_catastrophic_failure.py
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path

VIDEO_ID = "24947a9f29"

# Paths
input_dir = f"/media/ajeet/data/MINI_BTP/data/validation/bsc_imgs/{VIDEO_ID}/"
output_dir = f"./outputs/{VIDEO_ID}/frame_seq/"
mask_dir = f"/media/ajeet/data/MINI_BTP/data/validation/masks/{VIDEO_ID}/"

# Load all frames
input_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.jpg')])
output_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.jpg')])
mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])

print(f"Found {len(input_files)} input frames")
print(f"Found {len(output_files)} output frames")
print(f"Found {len(mask_files)} mask frames")

# Analyze each frame
frame_psnrs = []

for i, (inp_f, out_f, mask_f) in enumerate(zip(input_files, output_files, mask_files)):
    # Load frames
    inp = cv2.imread(os.path.join(input_dir, inp_f))
    out = cv2.imread(os.path.join(output_dir, out_f))
    mask = cv2.imread(os.path.join(mask_dir, mask_f), 0)
    
    if inp is None or out is None or mask is None:
        print(f"⚠️ Frame {i}: Failed to load")
        continue
    
    # Compute PSNR for this frame
    mse = np.mean((inp.astype(float) - out.astype(float)) ** 2)
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    
    frame_psnrs.append(psnr)
    
    # Create detailed visualization for worst frame
    if psnr < 15:  # If catastrophically bad
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'CATASTROPHIC FAILURE - Frame {i} | PSNR: {psnr:.2f} dB', 
                     fontsize=16, color='red')
        
        # Row 1
        axes[0, 0].imshow(cv2.cvtColor(inp, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Corrupted Input')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(mask, cmap='gray')
        axes[0, 1].set_title(f'Mask (Coverage: {np.mean(mask > 0)*100:.1f}%)')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title('Model Output (FAILED)')
        axes[0, 2].axis('off')
        
        # Row 2: Error analysis
        error = np.abs(inp.astype(float) - out.astype(float))
        error_map = np.mean(error, axis=2)
        
        axes[1, 0].imshow(error_map, cmap='hot', vmin=0, vmax=255)
        axes[1, 0].set_title(f'Error Map (Avg: {np.mean(error_map):.1f})')
        axes[1, 0].axis('off')
        
        # Channel-wise analysis
        error_r = error[:, :, 2]  # Red channel
        error_g = error[:, :, 1]  # Green
        error_b = error[:, :, 0]  # Blue
        
        axes[1, 1].hist([error_r.flatten(), error_g.flatten(), error_b.flatten()], 
                       bins=50, color=['red', 'green', 'blue'], alpha=0.5,
                       label=['R', 'G', 'B'])
        axes[1, 1].set_title('Per-Channel Error Distribution')
        axes[1, 1].set_xlabel('Error Magnitude')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Spatial error profile
        axes[1, 2].plot(np.mean(error_map, axis=1), label='Horizontal avg')
        axes[1, 2].plot(np.mean(error_map, axis=0), label='Vertical avg')
        axes[1, 2].set_title('Spatial Error Profile')
        axes[1, 2].set_xlabel('Position')
        axes[1, 2].set_ylabel('Error')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'FAILURE_frame_{i:03d}_{psnr:.1f}dB.png', dpi=150)
        print(f"⚠️ Frame {i}: CATASTROPHIC {psnr:.2f} dB - Saved analysis")
    else:
        print(f"✓ Frame {i}: {psnr:.2f} dB")

# Overall analysis
print("\n" + "="*60)
print(f"VIDEO {VIDEO_ID} - FAILURE ANALYSIS")
print("="*60)
print(f"Average PSNR: {np.mean(frame_psnrs):.2f} dB")
print(f"Min PSNR: {np.min(frame_psnrs):.2f} dB (frame {np.argmin(frame_psnrs)})")
print(f"Max PSNR: {np.max(frame_psnrs):.2f} dB (frame {np.argmax(frame_psnrs)})")
print(f"Std Dev: {np.std(frame_psnrs):.2f} dB")

# Plot PSNR over time
plt.figure(figsize=(12, 6))
plt.plot(frame_psnrs, marker='o', linewidth=2, markersize=8)
plt.axhline(y=np.mean(frame_psnrs), color='r', linestyle='--', 
            label=f'Mean: {np.mean(frame_psnrs):.2f} dB')
plt.axhline(y=20, color='orange', linestyle='--', alpha=0.5, label='Poor quality threshold')
plt.xlabel('Frame Number', fontsize=12)
plt.ylabel('PSNR (dB)', fontsize=12)
plt.title(f'Video {VIDEO_ID} - Per-Frame PSNR', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(f'FAILURE_timeline_{VIDEO_ID}.png', dpi=150)

print(f"\nSaved: FAILURE_timeline_{VIDEO_ID}.png")
print("\nCheck FAILURE_*.png files for detailed analysis")