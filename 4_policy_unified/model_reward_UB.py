"""UB Reward model (siamese + overlap-residual, optional action attention).

  final_score = alpha * overlap_norm + beta * learned_residual

  overlap_norm     : per-sample z-score of sum_DHW(belief * swept_i) over
                     the N action axis. No-learning baseline diagnostic.
  learned_residual : siamese 3D-CNN over concat(belief, swept_i), pooled
                     with mean+max concat. Optionally, a transformer block
                     attends across the N action tokens so the per-action
                     scores can be informed by the OTHER candidates
                     (key for top1 discrimination among near-tied actions).
                     Enable with use_attn=True.

Input  : belief (B, 1, D, H, W) + swept (B, N, D, H, W)
Output : (B, N) scores.
"""
import torch
import torch.nn as nn


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


class ActionAttention(nn.Module):
    """One transformer-encoder block over the N action tokens.

    Input  : (B, N, dim)
    Output : (B, N, dim)  — same shape, but each token has attended to others.
    Pre-norm with residual + a 2x FF expansion.
    """

    def __init__(self, dim, n_heads=4, ff_mult=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Linear(dim * ff_mult, dim),
        )

    def forward(self, x, key_padding_mask=None):
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask,
                         need_weights=False)
        x = x + a
        x = x + self.ff(self.norm2(x))
        return x


class RewardNetUB(nn.Module):
    """Per-action scorer with overlap-residual structure."""

    def __init__(self, base_ch=16, alpha=1.0, beta=0.1, learnable_ab=False,
                 use_attn=False, attn_heads=4):
        super().__init__()
        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8

        self.enc1 = ConvBlock3D(2, c1)
        self.enc2 = ConvBlock3D(c1, c2)
        self.enc3 = ConvBlock3D(c2, c3)
        self.enc4 = ConvBlock3D(c3, c4)
        self.pool = nn.MaxPool3d(2)

        feat_dim = 2 * c4
        self.use_attn = bool(use_attn)
        if self.use_attn:
            self.attn = ActionAttention(feat_dim, n_heads=attn_heads)

        self.head = nn.Sequential(
            nn.Linear(feat_dim, c3),
            nn.ReLU(inplace=True),
            nn.Linear(c3, c2),
            nn.ReLU(inplace=True),
            nn.Linear(c2, 1),
        )

        self.learnable_ab = bool(learnable_ab)
        if self.learnable_ab:
            self.alpha = nn.Parameter(torch.tensor(float(alpha)))
            self.beta = nn.Parameter(torch.tensor(float(beta)))
        else:
            self.alpha = float(alpha)
            self.beta = float(beta)

    def current_ab(self):
        """Return (alpha, beta) as floats for logging."""
        a = self.alpha.item() if isinstance(self.alpha, torch.Tensor) else self.alpha
        b = self.beta.item() if isinstance(self.beta, torch.Tensor) else self.beta
        return a, b

    @staticmethod
    def _overlap_score(belief, swept):
        """Per-sample z-scored sum_DHW(belief * swept_i).

        Computed in fp32 (autocast disabled) — the spatial sum over ~5e5
        voxels can otherwise overflow fp16. Result is then re-cast to
        whatever dtype caller is in.
        """
        with torch.amp.autocast('cuda', enabled=False):
            ov = (belief.float() * swept.float()).sum(dim=(2, 3, 4))   # (B, N)
            m = ov.mean(dim=-1, keepdim=True)
            s = ov.std(dim=-1, keepdim=True) + 1e-6
            return ((ov - m) / s).to(swept.dtype)

    def forward(self, belief, swept, valid_mask=None):
        B, N, D, H, W = swept.shape

        overlap_norm = self._overlap_score(belief, swept)   # (B, N)

        belief_rep = belief.unsqueeze(1).expand(B, N, 1, D, H, W)
        swept_rep = swept.unsqueeze(2)
        x = torch.cat([belief_rep, swept_rep], dim=2)
        x = x.reshape(B * N, 2, D, H, W)

        x = self.pool(self.enc1(x))   # (B*N, c1, 30, 60, 40)
        x = self.pool(self.enc2(x))   # (B*N, c2, 15, 30, 20)
        x = self.pool(self.enc3(x))   # (B*N, c3,  7, 15, 10)
        x = self.enc4(x)              # (B*N, c4,  7, 15, 10)

        flat = x.flatten(2)
        xm = flat.mean(dim=-1)
        xM = flat.amax(dim=-1)
        x = torch.cat([xm, xM], dim=-1)               # (B*N, 2*c4)

        if self.use_attn:
            x = x.view(B, N, -1)                      # (B, N, 2*c4)
            kpm = None
            if valid_mask is not None:
                # key_padding_mask: True = ignore. So invert valid_mask.
                kpm = ~valid_mask                     # (B, N) bool
            x = self.attn(x, key_padding_mask=kpm)    # attend across N actions
            x = x.reshape(B * N, -1)

        learned = self.head(x).squeeze(-1).view(B, N)

        ov_term = self.alpha * overlap_norm
        lr_term = self.beta * learned
        self._last_ov_term = ov_term.detach()
        self._last_lr_term = lr_term.detach()
        return ov_term + lr_term


if __name__ == '__main__':
    belief = torch.randn(2, 1, 60, 120, 80)
    swept = torch.randn(2, 20, 60, 120, 80)
    for use_attn in (False, True):
        m = RewardNetUB(base_ch=16, learnable_ab=True, use_attn=use_attn)
        y = m(belief, swept)
        n_params = sum(p.numel() for p in m.parameters())
        a, b = m.current_ab()
        print(f'use_attn={use_attn}  output={tuple(y.shape)}  '
              f'params={n_params/1e6:.4f}M  alpha={a:.3f} beta={b:.3f}')
