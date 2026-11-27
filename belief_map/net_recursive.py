"""
BeliefUpdateNetRecursive
========================
Semantic-recursive version of BeliefUpdateNet (softmax probability feedback),
with the per-view semantic observation expanded into an n_sem-channel one-hot.

Differences from the original BeliefUpdateNet:
  - The input channels (α/50, β/50, occ_t, swept_map, sem_obs) are kept, except
    that sem_obs goes from a 1-channel scalar lookup to an n_sem-channel one-hot.
  - Extra input: sem_belief (B, n_sem, W, H, D) — the accumulated semantic
    belief, i.e. softmax of the previous step's sem_logits, detached and fed
    back as the next step's input (the same recursive pattern as occupancy α/β).
  - The output is unchanged: (new_α, new_β, sem_logits). sem_logits keeps its
    shape, dtype and meaning, so the existing CE loss code needs no changes.

Channel counts (with_sweep=True, with_semantic=True, n_sem=15):
  c_in  = 3 (a, b, occ) + 1 (sweep) + n_sem (sem_obs one-hot) + n_sem (sem_belief) = 34
  c_out = 2 + n_sem = 17

Channel layout:
  Ch 0           : alpha / 50.0                   (recursive occupancy belief α)
  Ch 1           : beta  / 50.0                   (recursive occupancy belief β)
  Ch 2           : occ                            (per-view occupancy obs)
  Ch 3           : swept_map                      (push action region)
  Ch 4..4+n_sem-1: one_hot(semantic_map, n_sem)   (per-view semantic obs, channel = class id)
  Ch 4+n_sem..   : sem_belief                     (recursive semantic belief softmax probs)

Why n_sem=15:
  The current data only uses 0=free + 11 objects = 12 classes, but 3 slots
  (12, 13, 14) are reserved so that finetuning on additional classes needs no
  change to the architecture.

Benefits of the one-hot semantic input:
  - sem_obs and sem_belief share one channel layout, so Ch[4+c] and Ch[4+n_sem+c]
    refer to the same class c; the convolutions can then learn a channel-wise
    comparison in the shape of a Bayesian update.
  - It removes the arbitrary mapping imposed by the permuted scalar lookup.
  - When a class is added, the channel position itself carries the meaning,
    which gives a clear learning signal during finetuning.

Training / inference usage pattern:
  sem_belief = torch.full((B, n_sem, *VOL_SHAPE), 1.0/n_sem, device=device)  # uniform prior
  for t in range(N_STEPS):
      out = net(alpha, beta, occ_t, sweep_t, sem_t, sem_belief)
      alpha, beta, sem_logits = out
      sem_belief = F.softmax(sem_logits, dim=1).detach()
      alpha = alpha.detach(); beta = beta.detach()
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from belief_map.unet3d import UNet3D


class BeliefUpdateNetRecursive(nn.Module):
    """
    (b_t, occ_t, swept, sem_obs, sem_belief_t) → b_{t+1}, sem_logits_{t+1}

    sem_obs: per-view semantic observation (class id) — converted to an
        n_sem-channel one-hot.
    sem_belief_t: accumulated belief softmax(sem_logits) from the previous step,
        passed in detached. The initial value is either uniform (1/n_sem) or 0,
        decided by the caller.
    """
    EPSILON = 0.01

    def __init__(self, base_ch: int = 32,
                 with_sweep: bool    = True,
                 with_semantic: bool = True,
                 n_sem: int          = 15):
        super().__init__()
        assert n_sem > 0, ('BeliefUpdateNetRecursive only works with n_sem > 0 '
                           '(the recursive semantic channels are mandatory)')
        self.with_sweep    = with_sweep
        self.with_semantic = with_semantic
        self.n_sem         = n_sem
        # c_in: a, b, occ (3) + [sweep] (1) + [sem_obs one-hot] (n_sem) + sem_belief (n_sem)
        c_in  = 3 + int(with_sweep) + (n_sem if with_semantic else 0) + n_sem
        c_out = 2 + n_sem
        self.unet = UNet3D(c_in=c_in, c_out=c_out, base_ch=base_ch, cond_dim=0)

    def forward(
        self,
        alpha:        torch.Tensor,                # (B, W, H, D)
        beta:         torch.Tensor,                # (B, W, H, D)
        occ:          torch.Tensor,                # (B, W, H, D) 0/1
        swept_map:    torch.Tensor | None = None,  # (B, W, H, D) 0/1
        semantic_map: torch.Tensor | None = None,  # (B, W, H, D) int (per-view sem obs)
        sem_belief:   torch.Tensor | None = None,  # (B, n_sem, W, H, D) recursive belief
    ):
        a = alpha.unsqueeze(1) / 50.0
        b = beta .unsqueeze(1) / 50.0
        o = occ  .unsqueeze(1).float()

        channels = [a, b, o]

        if self.with_sweep:
            s = swept_map.unsqueeze(1).float() if swept_map is not None \
                else torch.zeros_like(o)
            channels.append(s)

        if self.with_semantic:
            if semantic_map is not None:
                # (B, W, H, D) class id → (B, n_sem, W, H, D) one-hot float
                idx = semantic_map.long().clamp(0, self.n_sem - 1)
                sem = F.one_hot(idx, num_classes=self.n_sem)            # (B, W, H, D, n_sem)
                sem = sem.permute(0, sem.dim() - 1,
                                  *range(1, sem.dim() - 1)).float()      # (B, n_sem, W, H, D)
            else:
                sem = torch.zeros(
                    alpha.shape[0], self.n_sem, *alpha.shape[1:],
                    device=alpha.device, dtype=alpha.dtype,
                )
            channels.append(sem)

        # sem_belief: (B, n_sem, W, H, D) — fall back to a uniform prior if absent
        if sem_belief is None:
            B = alpha.shape[0]
            sem_belief = torch.full(
                (B, self.n_sem, *alpha.shape[1:]),
                1.0 / self.n_sem,
                device=alpha.device, dtype=alpha.dtype,
            )
        channels.append(sem_belief)

        x   = torch.cat(channels, dim=1)
        out = self.unet(x)                          # (B, 2+n_sem, W, H, D)

        occ_out   = F.softplus(out[:, :2]) + 1.0 + self.EPSILON
        new_alpha = occ_out[:, 0]
        new_beta  = occ_out[:, 1]
        sem_logits = out[:, 2:]                     # (B, n_sem, W, H, D) raw logits

        return new_alpha, new_beta, sem_logits
