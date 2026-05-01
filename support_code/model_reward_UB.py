"""UB Reward model: belief + 29 swept_maps -> (29,) logits over candidate pushes.

Input  : (B, 1, 60, 120, 80) belief + (B, 29, 60, 120, 80) swept_maps
         (concatenated to (B, 30, 60, 120, 80) inside forward)
Output : (B, 29) raw logits (push idx 0..28)

Architecture mirrors the UA RewardNet:
  3D conv encoder (3 downsamples) -> collapse D -> bilinear interp to (1, 29)
  -> 2D conv head -> (1, 1, 29) -> flatten -> (29,)

The (1, 29) interp gives a 1D spatial inductive bias for left-to-right push
smoothness across col_idx.
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
    def __init__(self, in_ch, out_ch, kernel=3):
        super().__init__()
        g = min(8, out_ch)
        pad = kernel // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=pad),
            nn.GroupNorm(g, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel, padding=pad),
            nn.GroupNorm(g, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class RewardNetUB(nn.Module):
    """30-channel 3D input -> (B, 29) push logits."""

    NUM_PUSHES = 29

    def __init__(self, base_ch=16):
        super().__init__()
        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8
        in_ch = 1 + self.NUM_PUSHES   # belief + 29 swept

        self.enc1 = ConvBlock3D(in_ch, c1)
        self.enc2 = ConvBlock3D(c1, c2)
        self.enc3 = ConvBlock3D(c2, c3)
        self.enc4 = ConvBlock3D(c3, c4)
        self.pool = nn.MaxPool3d(2)

        # 2D head over (1, 29) grid.  Use kernel (1,3) so receptive field grows
        # only along the push axis (W).
        self.head2d = nn.Sequential(
            ConvBlock2D(c4, c3, kernel=3),
            ConvBlock2D(c3, c2, kernel=3),
            nn.Conv2d(c2, 1, 1),
        )

    def forward(self, belief, swept_maps):
        """belief: (B, 1, D, H, W), swept_maps: (B, 29, D, H, W)."""
        x = torch.cat([belief, swept_maps], dim=1)   # (B, 30, D, H, W)
        x = self.pool(self.enc1(x))   # (B, c1, 30, 60, 40)
        x = self.pool(self.enc2(x))   # (B, c2, 15, 30, 20)
        x = self.pool(self.enc3(x))   # (B, c3,  7, 15, 10)
        x = self.enc4(x)              # (B, c4,  7, 15, 10)
        x = x.mean(dim=2)             # collapse D -> (B, c4, 15, 10)
        x = F.interpolate(x, size=(1, self.NUM_PUSHES),
                          mode='bilinear', align_corners=False)
        x = self.head2d(x)            # (B, 1, 1, 29)
        return x.flatten(1)           # (B, 29)


if __name__ == '__main__':
    m = RewardNetUB(base_ch=16)
    belief = torch.randn(2, 1, 60, 120, 80)
    swept = torch.randn(2, 29, 60, 120, 80)
    y = m(belief, swept)
    n_params = sum(p.numel() for p in m.parameters())
    print(f'belief={tuple(belief.shape)}  swept={tuple(swept.shape)} '
          f'-> output={tuple(y.shape)}  params={n_params/1e6:.2f}M')
