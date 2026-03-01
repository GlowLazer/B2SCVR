"""
Boundary Refinement Head for B2SCVR.

A lightweight post-processing module that refines the seam between the
model-restored region (inside mask) and the original frame (outside mask).

Architecture:
    Input  : [restored(3) | original(3) | mask(1) | band(1)] = 8ch
    Body   : Conv(8→64) + 3 ResBlocks + ChannelAttention
    Output : delta(3) + alpha(1)  →  soft boundary blend

Final composition (per pixel):
    outside mask   → original           (untouched)
    inside band    → (1-α)*original + α*(restored + Δ)   (learned blend)
    inside mask    → restored           (untouched)

All tensors in [-1, 1] range (matching tanh model output).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.gap(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(x + self.block(x))


class BoundaryRefinementHead(nn.Module):
    """
    Lightweight boundary refinement head.
    Operates at inference resolution (e.g. 432x240), frame by frame.
    """

    def __init__(self, channels=64, dil_k=5, ero_k=5):
        super().__init__()
        self.dil_k = dil_k
        self.ero_k = ero_k

        # 8ch in: restored(3) + original(3) + mask(1) + band(1)
        self.conv_in = nn.Sequential(
            nn.Conv2d(8, channels, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.res1 = ResBlock(channels)
        self.res2 = ResBlock(channels)
        self.res3 = ResBlock(channels)
        self.ca   = ChannelAttention(channels)
        # 4ch out: delta(3) + alpha(1)
        self.conv_out = nn.Conv2d(channels, 4, 3, 1, 1)

    def compute_band(self, mask):
        """Boundary band = dilation(mask) - erosion(mask)."""
        dil = F.max_pool2d(mask, kernel_size=self.dil_k, stride=1,
                           padding=self.dil_k // 2)
        ero = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=self.ero_k,
                                  stride=1, padding=self.ero_k // 2)
        return (dil - ero).clamp(0.0, 1.0)

    def forward(self, restored, original, mask):
        """
        Args:
            restored : (B, 3, H, W)  model output,    range [-1, 1]
            original : (B, 3, H, W)  input/BSC frame,  range [-1, 1]
            mask     : (B, 1, H, W)  binary mask,       range [0, 1]
        Returns:
            refined  : (B, 3, H, W)  boundary-refined output, range [-1, 1]
            band     : (B, 1, H, W)  boundary band (for loss weighting)
        """
        band = self.compute_band(mask)   # (B, 1, H, W)

        x    = torch.cat([restored, original, mask, band], dim=1)   # (B,8,H,W)
        feat = self.conv_in(x)
        feat = self.res1(feat)
        feat = self.res2(feat)
        feat = self.res3(feat)
        feat = self.ca(feat)
        out  = self.conv_out(feat)                    # (B, 4, H, W)

        delta = torch.tanh(out[:, :3])               # RGB correction in [-1,1]
        alpha = torch.sigmoid(out[:, 3:4])           # blend weight in [0,1]

        # Inside band: learned soft blend
        blended = (1.0 - alpha) * original + alpha * (restored + delta)

        # Compose: band region → blended, rest → restored (trainer adds hard composite)
        refined = restored * (1.0 - band) + blended * band

        return refined.clamp(-1.0, 1.0), band
