"""
3D UNet backbone shared by BeliefUpdateNet and BeliefUpdateNetRecursive.

Voxel space: W=120 (x), H=60 (z-height), D=80 (y-depth) — the belief-map
modules work on the (W, H, D) transpose of the (60, 120, 80) grid used
elsewhere in the pipeline.
Self-contained — no external imports beyond torch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Building blocks ────────────────────────────────────────────────────────────

def conv3d_bn_relu(in_ch, out_ch, kernel=3, padding=1, stride=1):
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=False),
        nn.BatchNorm3d(out_ch),
        nn.ReLU(inplace=True),
    )


class ResBlock3d(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv3d(ch, ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm3d(ch)
        self.conv2 = nn.Conv3d(ch, ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm3d(ch)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        r = self.relu(self.bn1(self.conv1(x)))
        r = self.bn2(self.conv2(r))
        return self.relu(x + r)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = conv3d_bn_relu(in_ch, out_ch, kernel=3, padding=1, stride=2)
        self.res  = ResBlock3d(out_ch)

    def forward(self, x):
        return self.res(self.down(x))


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.proj = conv3d_bn_relu(in_ch, out_ch)
        self.fuse = nn.Sequential(
            conv3d_bn_relu(out_ch + skip_ch, out_ch),
            ResBlock3d(out_ch),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=True)
        x = self.proj(x)
        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)


# ── 3D UNet backbone ───────────────────────────────────────────────────────────

class UNet3D(nn.Module):
    """
    Generic 3D UNet.  Input/Output: (B, c_in/c_out, W, H, D)

    Spatial resolution per stage (W=120, H=60, D=80, base_ch=32):
      enc0 : (B, 32,  120, 60, 80)
      enc1 : (B, 64,   60, 30, 40)
      enc2 : (B, 128,  30, 15, 20)
      enc3 : (B, 256,  15,  8, 10)  ← bottleneck
    """
    def __init__(self, c_in: int, c_out: int, base_ch: int = 32, cond_dim: int = 0):
        super().__init__()
        ch = base_ch
        self.cond_dim = cond_dim

        self.enc0 = nn.Sequential(conv3d_bn_relu(c_in, ch), ResBlock3d(ch))
        self.enc1 = DownBlock(ch,   ch * 2)
        self.enc2 = DownBlock(ch*2, ch * 4)
        self.enc3 = DownBlock(ch*4, ch * 8)

        btn_in = ch * 8 + cond_dim
        self.bottleneck = nn.Sequential(
            conv3d_bn_relu(btn_in, ch * 8),
            ResBlock3d(ch * 8),
        )

        self.dec2 = UpBlock(ch*8, ch*4, ch*4)
        self.dec1 = UpBlock(ch*4, ch*2, ch*2)
        self.dec0 = UpBlock(ch*2, ch,   ch)

        self.head = nn.Conv3d(ch, c_out, 1)

    def forward(self, x, cond=None):
        e0 = self.enc0(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        if cond is not None and self.cond_dim > 0:
            c  = cond.view(cond.shape[0], -1, 1, 1, 1).expand(-1, -1, *e3.shape[2:])
            e3 = torch.cat([e3, c], dim=1)

        b  = self.bottleneck(e3)
        d2 = self.dec2(b,  e2)
        d1 = self.dec1(d2, e1)
        d0 = self.dec0(d1, e0)
        return self.head(d0)   # (B, c_out, W, H, D)

    def encode(self, x):
        """Return the decoder feature map (before the output head), for
        dual-head models that attach their own heads."""
        e0 = self.enc0(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b  = self.bottleneck(e3)
        d2 = self.dec2(b,  e2)
        d1 = self.dec1(d2, e1)
        d0 = self.dec0(d1, e0)
        return d0   # (B, base_ch, W, H, D)
