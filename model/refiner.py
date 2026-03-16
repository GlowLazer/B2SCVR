import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBlock(nn.Module):
    """Two conv layers with LeakyReLU — no normalisation (safer for PSNR)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResidualRefiner(nn.Module):
    """
    Small UNet that predicts a residual correction on the TTA-merged output.

    Input  : concat([bsc, mask, pred])  — 7 channels, all in [0, 1]
               bsc  (3ch) : corrupted BSC frame
               mask (1ch) : binary inpainting mask
               pred (3ch) : TTA-merged model output

             Optional (in_ch=10):
               lap  (3ch) : Laplacian-ish feature from pred (pred - gaussian_blur(pred))
               concat([bsc, mask, pred, lap])  — 10 channels

    Output : out = (pred + delta).clamp(0, 1)
             out = out * mask + bsc * (1 - mask)
             — correction only inside mask and strict data consistency outside

    Parameters (base_ch=32) : ~768 K
    """

    def __init__(self, in_ch=7, base_ch=32):
        super().__init__()
        if in_ch not in (7, 10):
            raise ValueError(f'in_ch must be 7 or 10, got {in_ch}')
        self.in_ch = int(in_ch)
        bc = base_ch

        # Encoder
        self.enc1 = _ConvBlock(in_ch,   bc)       # (B, bc,   H,   W)
        self.enc2 = _ConvBlock(bc,      bc * 2)   # (B, bc*2, H/2, W/2)
        self.enc3 = _ConvBlock(bc * 2,  bc * 4)   # (B, bc*4, H/4, W/4)

        # Bottleneck
        self.mid  = _ConvBlock(bc * 4,  bc * 4)

        # Decoder  (skip + up)
        self.dec2 = _ConvBlock(bc * 4 + bc * 2, bc * 2)
        self.dec1 = _ConvBlock(bc * 2 + bc,     bc)

        # Output residual — initialised to zero → safe no-op at training start
        self.out  = nn.Conv2d(bc, 3, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def _gaussian_blur5(self, x):
        # Separable 5x5 Gaussian-ish blur. x: (B,3,H,W) -> (B,3,H,W)
        k = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0],
                         device=x.device, dtype=x.dtype)
        k = k / k.sum()
        kh = k.view(1, 1, 1, 5).repeat(3, 1, 1, 1)
        kv = k.view(1, 1, 5, 1).repeat(3, 1, 1, 1)
        x = F.conv2d(x, kh, padding=(0, 2), groups=3)
        x = F.conv2d(x, kv, padding=(2, 0), groups=3)
        return x

    def forward(self, bsc, mask, pred):
        """
        bsc  : (B, 3, H, W) in [0, 1]
        mask : (B, 1, H, W) in [0, 1]
        pred : (B, 3, H, W) in [0, 1]
        returns: (B, 3, H, W) in [0, 1]
        """
        if self.in_ch == 10:
            lap = pred - self._gaussian_blur5(pred)
            x = torch.cat([bsc, mask, pred, lap], dim=1)   # (B, 10, H, W)
        else:
            x = torch.cat([bsc, mask, pred], dim=1)        # (B, 7, H, W)

        e1 = self.enc1(x)                             # (B, bc,   H,   W)
        e2 = self.enc2(F.max_pool2d(e1, 2))           # (B, bc*2, H/2, W/2)
        e3 = self.enc3(F.max_pool2d(e2, 2))           # (B, bc*4, H/4, W/4)

        m  = self.mid(e3)                             # (B, bc*4, H/4, W/4)

        d2 = self.dec2(torch.cat([
            F.interpolate(m,  scale_factor=2, mode='bilinear', align_corners=False), e2], 1))
        d1 = self.dec1(torch.cat([
            F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False), e1], 1))

        delta = self.out(d1)                          # (B, 3, H, W)
        out = (pred + delta).clamp(0, 1)
        # Strict data consistency: never touch clean pixels.
        return (out * mask + bsc * (1.0 - mask)).clamp(0, 1)
