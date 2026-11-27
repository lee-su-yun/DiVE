"""Unified UA+UB 3D UNet — the decomposed visibility estimator f_DiVE.

Input  : (B, 3, D, H, W) = [belief_A_t, belief_B_t, voxel_t]
Output : (B, 2, D, H, W) = [logit_UA, logit_UB] (raw logit; sigmoid -> belief in [0,1])

- The backbone (encoder / bottleneck / decoder) is shared; only the final
  out_conv carries the 2 output channels.
- prior_a and prior_b initialize the output bias of each channel independently.
- No swept_map input: the sequence is view-only.
"""
import math

import torch
import torch.nn as nn


class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        num_groups = min(8, out_ch)
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(num_groups, out_ch),
            nn.ReLU(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(num_groups, out_ch),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv(x)


class UNet3DPushUnified(nn.Module):
    """3D UNet with shared backbone and 2-channel output head."""

    def __init__(self, in_ch=3, prior_a=0.1, prior_b=0.1):
        super().__init__()
        self.enc1 = ConvBlock3D(in_ch, 16)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = ConvBlock3D(16, 32)
        self.pool2 = nn.MaxPool3d(2)

        self.bottleneck = ConvBlock3D(32, 64)

        self.up2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec2 = ConvBlock3D(64, 32)
        self.up1 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.dec1 = ConvBlock3D(32, 16)

        self.out_conv = nn.Conv3d(16, 2, 1)
        nn.init.zeros_(self.out_conv.weight)
        with torch.no_grad():
            self.out_conv.bias[0] = math.log(prior_a / (1.0 - prior_a))
            self.out_conv.bias[1] = math.log(prior_b / (1.0 - prior_b))

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))

        b = self.bottleneck(self.pool2(e2))

        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out_conv(d1)  # (B, 2, D, H, W)
