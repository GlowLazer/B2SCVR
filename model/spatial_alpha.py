"""
SpatialAlphaNet — per-pixel blending weight for TTA forward/backward merge.

Takes the forward (F) and backward (B) predictions as 6-channel input and
outputs a spatial alpha map α(x,y) ∈ [0,1]:

    Final(x,y) = α(x,y) · F(x,y) + (1 - α(x,y)) · B(x,y)

Architecture: 3 conv layers, no heavy attention — keeps inference fast.
Input normalised to [0, 1].
"""

import torch
import torch.nn as nn


class SpatialAlphaNet(nn.Module):
    """Lightweight spatial alpha estimator for TTA fwd/bwd merge."""

    def __init__(self, mid_ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, mid_ch, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        # Initialise final bias to 0 → default output ≈ 0.5 (unbiased blend)
        nn.init.zeros_(self.net[-2].bias)

    def forward(self, fwd, bwd):
        """
        Args:
            fwd : (B, 3, H, W)  forward-pass prediction, values in [0, 1]
            bwd : (B, 3, H, W)  backward-pass prediction, values in [0, 1]
        Returns:
            alpha : (B, 1, H, W)  per-pixel blend weight in [0, 1]
        """
        x = torch.cat([fwd, bwd], dim=1)   # (B, 6, H, W)
        return self.net(x)
