"""
LoRA utilities shared across SAM2 and Transformer finetuning scripts.

Classes / functions exported:
  LoRALinear           — wraps an nn.Linear with a low-rank adapter
  inject_lora_sam2     — inject into SAM2 Hiera-T trunk (attn.qkv only)
  inject_lora_transformer — inject into WindowAttention blocks (qkv, optionally proj)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Wraps an existing nn.Linear with a low-rank adapter (A·B).

    The wrapped weight stays frozen; only lora_A and lora_B are trained.
    Output = W(x) + (x @ A.T @ B.T) * scale
    """

    def __init__(self, linear: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.linear = linear
        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.randn(r, linear.in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(linear.out_features, r))
        self.scale  = alpha / r

    def forward(self, x):
        return self.linear(x) + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale


def inject_lora_sam2(trunk, r: int = 8, alpha: int = 16):
    """Inject LoRA into every MultiScaleAttention.qkv in a Hiera trunk (in-place).

    Args:
        trunk: SAM2Extractor.encoder  (model.image_encoder.trunk)
        r: LoRA rank
        alpha: LoRA alpha scaling
    """
    count = 0
    for block in trunk.blocks:
        block.attn.qkv = LoRALinear(block.attn.qkv, r=r, alpha=alpha)
        count += 1
    print(f'[LoRA] SAM2 trunk injected into {count} blocks (r={r}, alpha={alpha})')
    return trunk


def inject_lora_transformer(model, r: int = 8, alpha: int = 16, include_proj: bool = False):
    """Inject LoRA into WindowAttention blocks in all TemporalFocalTransformerBlocks (in-place).

    Args:
        model: full InpaintGenerator model
        r: LoRA rank
        alpha: LoRA alpha scaling
        include_proj: if True, also wrap proj (output projection). Default False (qkv only).
    """
    from model.modules.tfocal_transformer_hq import WindowAttention
    count = 0
    for m in model.modules():
        if isinstance(m, WindowAttention):
            m.qkv = LoRALinear(m.qkv, r=r, alpha=alpha)
            if include_proj:
                m.proj = LoRALinear(m.proj, r=r, alpha=alpha)
            count += 1
    print(f'[LoRA] Transformer injected into {count} WindowAttention blocks '
          f'(r={r}, alpha={alpha}, proj={include_proj})')
    return model
