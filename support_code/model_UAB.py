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


class UNet3DPush(nn.Module):
    """3D UNet for direct soft-BCE visibility prediction.

    Input : (B, 3, D, H, W) = prev_belief + voxel view + swept_map
    Output: (B, 1, D, H, W) raw logit (sigmoid -> belief in [0,1])
    """

    def __init__(self, in_ch=3, prior=0.1):
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

        self.out_conv = nn.Conv3d(16, 1, 1)
        # Prior-aware init: weight=0, bias=logit(prior) -> sigmoid(logit)==prior everywhere at init.
        nn.init.zeros_(self.out_conv.weight)
        nn.init.constant_(self.out_conv.bias, math.log(prior / (1.0 - prior)))

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))

        b = self.bottleneck(self.pool2(e2))

        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out_conv(d1)
