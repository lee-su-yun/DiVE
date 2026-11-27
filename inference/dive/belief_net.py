"""Unified UA + UB 3D UNet: shared backbone with a 2-channel output head.

Mirrors the training architecture in dive/train_UAB_unified.py.

Input  : (B, 3, D, H, W) = [belief_A_t, belief_B_t, voxel_t]   (no swept_map)
Output : (B, 2, D, H, W) = [logit_UA, logit_UB]
with (D, H, W) = (60, 120, 80).

Prior-aware bias init: sigmoid(logit_A) == prior_a, sigmoid(logit_B) == prior_b
at init (weight=0).

Checkpoint parity verified against the released
DiVE_decomposed_visibility_estimator/best.pth:
  enc1.conv.0.weight (16, 3, 3, 3, 3) ; out_conv (2, 16, 1, 1, 1).
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
    """Shared backbone, 2-channel head. Channel 0 = U_A logit, 1 = U_B logit."""

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
        return self.out_conv(d1)
