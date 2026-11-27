"""Reward model: belief map (1,60,120,80) -> (98,) logits over candidate cams.

Architecture:
  3D conv encoder (3 downsamples) -> collapse D -> bilinear interpolate to
  (7,14) camera grid -> 2D conv head -> (1,7,14) -> flatten row-major -> (98,)

Loss is flat KL / softmax CE over the 98 logits; the 7x14 shape is purely an
inductive bias for spatial weight-sharing across neighboring cameras
(row = v // 14, col = v % 14).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        g = min(8, out_ch)
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(g, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(g, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ConvBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        g = min(8, out_ch)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(g, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(g, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class RewardNet(nn.Module):
    """Input  : (B, 1, 60, 120, 80) belief map
                + optional seen_mask (B, 7, 14) of cameras already observed
       Output : (B, 98)              raw logits, row-major cam idx = row*14 + col

    When use_seen_mask=True, after the 3D encoder collapses to a (B, c4, 7, 14)
    feature map matching the camera grid, the binary seen mask is concatenated
    as an extra channel before the 2D head. This gives the model explicit
    "which cams were used" info without forcing it to recover that from the
    belief map's spatial pattern.
    """

    def __init__(self, base_ch=16, use_seen_mask=False):
        super().__init__()
        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8
        self.use_seen_mask = use_seen_mask

        self.enc1 = ConvBlock3D(1, c1)
        self.enc2 = ConvBlock3D(c1, c2)
        self.enc3 = ConvBlock3D(c2, c3)
        self.enc4 = ConvBlock3D(c3, c4)
        self.pool = nn.MaxPool3d(2)

        head_in = c4 + (1 if use_seen_mask else 0)
        self.head2d = nn.Sequential(
            ConvBlock2D(head_in, c3),
            ConvBlock2D(c3, c2),
            nn.Conv2d(c2, 1, 1),
        )

    def forward(self, x, seen_mask=None):
        x = self.pool(self.enc1(x))   # (B, c1, 30, 60, 40)
        x = self.pool(self.enc2(x))   # (B, c2, 15, 30, 20)
        x = self.pool(self.enc3(x))   # (B, c3,  7, 15, 10)
        x = self.enc4(x)              # (B, c4,  7, 15, 10)
        x = x.mean(dim=2)             # collapse D: (B, c4, 15, 10)
        x = F.interpolate(x, size=(7, 14), mode='bilinear', align_corners=False)
        if self.use_seen_mask:
            assert seen_mask is not None, 'use_seen_mask=True but no mask given'
            x = torch.cat([x, seen_mask.unsqueeze(1)], dim=1)  # (B, c4+1, 7, 14)
        x = self.head2d(x)            # (B, 1, 7, 14)
        return x.flatten(1)           # (B, 98)


if __name__ == '__main__':
    for use_mask in [False, True]:
        m = RewardNet(base_ch=16, use_seen_mask=use_mask)
        x = torch.randn(2, 1, 60, 120, 80)
        mask = torch.zeros(2, 7, 14)
        mask[:, 0, 0] = 1.0
        y = m(x, mask) if use_mask else m(x)
        n_params = sum(p.numel() for p in m.parameters())
        print(f'use_seen_mask={use_mask}: '
              f'input={tuple(x.shape)} -> output={tuple(y.shape)}  '
              f'params={n_params/1e6:.2f}M')
