"""
BeliefUpdateNet
===============
(b_t, occ_t[, swept][, sem_obs]) → (α, β)[, sem_logits]

Input (channels are concatenated):
  α/50, β/50  : (B, 2, W, H, D)  — previous occupancy belief (normalized)
  occ_t       : (B, 1, W, H, D)  — new binary observation
  swept_map   : (B, 1, W, H, D)  — push swept region (with_sweep=True)
  sem_obs     : (B, 1, W, H, D)  — per-view semantic observation, encoded
                                   through the class_scalar lookup
                                   (with_semantic=True)

Output:
  new_α, new_β : (B, W, H, D)  each ≥ 1+ε  — updated occupancy belief
  sem_logits   : (B, n_sem, W, H, D)  raw logits — semantic prediction (n_sem > 0)

c_in / c_out:
  sweep=F, sem=F, n_sem=0  →  c_in=3,  c_out=2
  sweep=T, sem=T, n_sem=12 →  c_in=5,  c_out=14   (YCB v3: 0=free + 11 objects)

Loss (see belief_map/train_finetune.py):
  MSE(α/(α+β), gt_occ) + w_sem * CrossEntropy(sem_logits, gt_sem)

This is the non-recursive variant. The released `estimator.pt` checkpoint is the
recursive one — see `net_recursive.BeliefUpdateNetRecursive`.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from belief_map.unet3d import UNet3D


MAX_CLASSES   = 100
ENCODING_SEED = 31337


def _build_class_scalar(max_classes: int = MAX_CLASSES,
                        seed: int       = ENCODING_SEED) -> torch.Tensor:
    """
    Returns (max_classes + 1,) float32 lookup table.
      table[0]   = 0.0                    # free / unobserved
      table[1..] = linspace(0.05, 0.95, max_classes) permuted by `seed`
    """
    linspace = np.linspace(0.05, 0.95, max_classes, dtype=np.float32)
    rng      = np.random.default_rng(seed)
    perm     = rng.permutation(max_classes)
    table    = np.zeros(max_classes + 1, dtype=np.float32)
    table[1:] = linspace[perm]
    return torch.from_numpy(table)


class BeliefUpdateNet(nn.Module):
    """
    (b_t, occ_t[, swept_map][, semantic]) → b_{t+1}[, sem_logits]

    with_sweep=True    → +1ch input: swept_map
    with_semantic=True → +1ch input: class_scalar[semantic_map]
                                     (scalable lookup, class 0 → 0.0)
    n_sem > 0          → +n_sem ch output: per-class semantic logits

    c_in combinations (with_free adds one more channel on top of these):
      sweep=F, sem=F → 3
      sweep=T, sem=F → 4
      sweep=F, sem=T → 4
      sweep=T, sem=T → 5

    Returns:
      n_sem == 0: (new_alpha, new_beta)
      n_sem  > 0: (new_alpha, new_beta, sem_logits)  sem_logits: (B, n_sem, W, H, D)
    """
    EPSILON = 0.01

    def __init__(self, base_ch: int = 32, with_sweep: bool = False,
                 with_semantic: bool = False, with_free: bool = False,
                 n_sem: int = 0,
                 max_classes: int  = MAX_CLASSES,
                 encoding_seed: int = ENCODING_SEED):
        super().__init__()
        self.with_sweep    = with_sweep
        self.with_semantic = with_semantic
        self.with_free     = with_free
        self.n_sem         = n_sem
        self.max_classes   = max_classes
        c_in  = 3 + int(with_sweep) + int(with_semantic) + int(with_free)
        c_out = 2 + n_sem
        self.unet = UNet3D(c_in=c_in, c_out=c_out, base_ch=base_ch, cond_dim=0)
        self.register_buffer(
            'class_scalar',
            _build_class_scalar(max_classes, encoding_seed),
        )

    def forward(
        self,
        alpha:        torch.Tensor,                # (B, W, H, D)
        beta:         torch.Tensor,                # (B, W, H, D)
        occ:          torch.Tensor,                # (B, W, H, D)  0/1
        swept_map:    torch.Tensor | None = None,  # (B, W, H, D)  0/1, None=zeros
        semantic_map: torch.Tensor | None = None,  # (B, W, H, D)  int, None=zeros
        free:         torch.Tensor | None = None,  # (B, W, H, D)  0/1, None=zeros
    ):
        """
        Returns:
          n_sem == 0: (new_alpha, new_beta)               each (B, W, H, D) ≥ 1+ε
          n_sem  > 0: (new_alpha, new_beta, sem_logits)   sem_logits raw, (B,n_sem,W,H,D)
        """
        a = alpha.unsqueeze(1) / 50.0    # (B, 1, W, H, D)
        b = beta .unsqueeze(1) / 50.0
        o = occ  .unsqueeze(1).float()   # (B, 1, W, H, D)

        channels = [a, b, o]

        if self.with_free:
            f = free.unsqueeze(1).float() if free is not None \
                else torch.zeros_like(o)
            channels.append(f)

        if self.with_sweep:
            s = swept_map.unsqueeze(1).float() if swept_map is not None \
                else torch.zeros_like(o)
            channels.append(s)

        if self.with_semantic:
            if semantic_map is not None:
                idx = semantic_map.long().clamp(0, self.max_classes)
                sem = self.class_scalar[idx].unsqueeze(1)  # (B, 1, W, H, D)
            else:
                sem = torch.zeros_like(o)
            channels.append(sem)

        x = torch.cat(channels, dim=1)

        out = self.unet(x)                          # (B, 2+n_sem, W, H, D)

        occ_out    = F.softplus(out[:, :2]) + 1.0 + self.EPSILON
        new_alpha  = occ_out[:, 0]                  # (B, W, H, D)
        new_beta   = occ_out[:, 1]

        if self.n_sem > 0:
            sem_logits = out[:, 2:]                 # (B, n_sem, W, H, D)  raw logits
            return new_alpha, new_beta, sem_logits

        return new_alpha, new_beta

    @property
    def occupancy(self):
        """Use after forward: occ = α/(α+β)"""
        pass  # computed externally


def partial_completion_loss(
    pred_alpha: torch.Tensor,   # (B, W, H, D)
    pred_beta:  torch.Tensor,   # (B, W, H, D)
    gt:         torch.Tensor,   # (B, W, H, D) bool — full GT occupancy
    vis_mask:   torch.Tensor,   # (B, W, H, D) bool — voxels of objects seen at least partially
    gamma: float = 2.0,
    pos_weight: float = 4.0,
) -> torch.Tensor:
    """
    Partial shape completion loss (Focal BCE + class weight).

    - vis_mask=True  (object was observed): Focal BCE against the GT, so the
      network learns to complete the object's boundary.
    - vis_mask=False (never observed): no loss at all, which prevents
      hallucinating unseen geometry.

    Focal + pos_weight: occupied voxels are far rarer than free ones, so the
    class imbalance has to be compensated.
    """
    prob = pred_alpha / (pred_alpha + pred_beta)   # (B, W, H, D)

    p_sel = prob[vis_mask]        # (N,)
    t_sel = gt[vis_mask].float()  # (N,)

    if p_sel.numel() == 0:
        return pred_alpha.sum() * 0.0

    p_sel = p_sel.clamp(1e-6, 1 - 1e-6)
    logit = torch.log(p_sel / (1 - p_sel))

    bce    = F.binary_cross_entropy_with_logits(logit, t_sel, reduction='none')
    prob_t = torch.where(t_sel > 0.5, p_sel, 1 - p_sel)
    focal  = (1 - prob_t) ** gamma
    weight = torch.where(t_sel > 0.5,
                         torch.full_like(t_sel, pos_weight),
                         torch.ones_like(t_sel))
    return (focal * weight * bce).mean()
