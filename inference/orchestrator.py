"""
DiVE + RECURSIVE semantic belief map parallel tracker — orchestrator.

Sibling of inference/orchestrator_base.py. The DiVE half (UAB unified belief +
RewardNet UA / UB + push provider) is identical. The central belief half is
swapped to use `BeliefUpdateNetRecursive`, which:

  - one-hot encodes per-view `sem_obs` internally (n_sem channels)
  - takes an additional `sem_belief` input (n_sem ch, recursive semantic belief)
  - feeds back `softmax(sem_logits).detach()` as next-step `sem_belief`

Channel layout (n_sem=15):
  Ch 0,1    : alpha/50, beta/50    (occupancy belief, recursive)
  Ch 2      : occ                  (per-view occupancy obs)
  Ch 3      : swept_map            (push action region)
  Ch 4..18  : sem_obs one-hot      (per-view semantic obs, 15ch)
  Ch 19..33 : sem_belief           (recursive semantic belief softmax, 15ch)
  → c_in = 34

The recursive checkpoint was trained with `n_sem=15` (12 active YCB classes +
3 reserved slots for finetune). Inference + metric evaluation continue to use
n_sem=12 active classes; the model architecture is built with n_sem=15 to
match the checkpoint, but downstream argmax over sem_logits still only ever
produces labels in [0, 11] for current data.

Public API:
  - CentralBeliefTrackRecursive (replaces CentralBeliefTrack)
  - OrchestratorRecursive       (replaces Orchestrator)

The rest of the module re-exports the constants / helpers of the non-recursive
sibling (CENTRAL_SHAPE, DIVE_SHAPE, CENTER_CAM_IDX, pick_post_push_cam, …), so
`from inference.orchestrator import ...` is a drop-in replacement for
`from inference.orchestrator_base import ...`.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

# ── Path setup ───────────────────────────────────────────────────────────────
_PKG_ROOT = Path(__file__).resolve().parents[1]   # repo root
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from belief_map.net_recursive import BeliefUpdateNetRecursive

# DiVE models
from inference.dive.belief_net import UNet3DPushUnified
from inference.dive.reward_net_ua import RewardNet
from inference.dive.reward_net_ub import RewardNetUB

# Re-use everything that didn't change from the non-recursive tracker so this
# stays a thin diff rather than a fork.
from inference.orchestrator_base import (
    N_CAMS, DIVE_SHAPE, CENTRAL_SHAPE, CENTER_CAM_IDX,
    N_CAM_COLS, N_CAM_ROWS, TOP_ZI_CHOICES,
    cam_xi_from_x_voxel, pick_post_push_cam, post_view_from_swept,
    central_to_dive, dive_to_central,
    UABUnifiedBeliefTrack, _BeliefView, StepLog,
    DEFAULT_UA_BELIEF_CKPT, DEFAULT_UB_BELIEF_CKPT,
    DEFAULT_UA_REWARD_CKPT, DEFAULT_UB_REWARD_CKPT,
)

DEFAULT_CENTRAL_RECURSIVE_CKPT = (
    'checkpoints/central/belief_update_scratch_ycb_v3_semantic_recursive_n15_onehot_ddp.pt'
)


# ── Central belief tracker (recursive semantic) ──────────────────────────────
class CentralBeliefTrackRecursive:
    """Semantic belief map track: (α, β, sem_logits) + BeliefUpdateNetRecursive.

    Holds an additional `sem_belief` (1, n_sem, W, H, D) recursive state,
    initialized to uniform 1/n_sem and replaced by softmax(sem_logits).detach()
    after each forward — matching the recursive training loop (see
    belief_map/train_finetune.py and belief_map/net_recursive.py).
    """

    def __init__(self, model: BeliefUpdateNetRecursive, device: str = 'cuda'):
        assert model.with_semantic and model.n_sem > 0, (
            'CentralBeliefTrackRecursive requires with_semantic=True and n_sem>0'
        )
        self.device = device
        self.model = model.to(device).eval()
        self.with_sweep    = model.with_sweep
        self.with_semantic = model.with_semantic
        self.n_sem         = model.n_sem
        self.alpha = torch.ones((1, *CENTRAL_SHAPE), device=device)
        self.beta  = torch.ones((1, *CENTRAL_SHAPE), device=device)
        self.last_sem_logits = None
        self.sem_belief = torch.full(
            (1, self.n_sem, *CENTRAL_SHAPE), 1.0 / self.n_sem, device=device,
        )

    def reset(self):
        """Fresh-episode reset (α=β=1, sem_belief=uniform)."""
        self.alpha = torch.ones((1, *CENTRAL_SHAPE), device=self.device)
        self.beta  = torch.ones((1, *CENTRAL_SHAPE), device=self.device)
        self.last_sem_logits = None
        self.sem_belief = torch.full(
            (1, self.n_sem, *CENTRAL_SHAPE), 1.0 / self.n_sem, device=self.device,
        )

    @torch.no_grad()
    def update(self, occ_view_central, swept_central, sem_voxel_central):
        """Inputs (W, H, D) on host or device.

        Recursive semantic belief is carried internally; caller doesn't pass it.
        """
        def _t(x, dtype):
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            return x.to(self.device, dtype=dtype).view(1, *CENTRAL_SHAPE)

        occ = _t(occ_view_central, torch.float32)
        swept_map = (_t(swept_central, torch.float32)
                     if self.with_sweep else None)
        # per-view semantic comes in as int label (W,H,D); model handles one-hot
        semantic_map = (_t(sem_voxel_central, torch.float32)
                        if self.with_semantic else None)

        self.alpha, self.beta, self.last_sem_logits = self.model(
            self.alpha, self.beta, occ,
            swept_map=swept_map,
            semantic_map=semantic_map,
            sem_belief=self.sem_belief,
        )
        # feed back softmax for next step (matches training)
        self.sem_belief = F.softmax(self.last_sem_logits, dim=1).detach()
        return self.alpha, self.beta, self.last_sem_logits


# ── Orchestrator (recursive central) ─────────────────────────────────────────
class OrchestratorRecursive:
    """Same wiring as Orchestrator, but the central tracker uses
    BeliefUpdateNetRecursive (+ recursive sem_belief state)."""

    def __init__(
        self,
        uab_belief_ckpt:   Optional[str] = None,
        ua_reward_ckpt:    Optional[str] = None,
        ub_reward_ckpt:    Optional[str] = None,
        central_ckpt:      Optional[str] = None,
        with_sweep_central:    bool = True,
        with_semantic_central: bool = True,
        # Model architecture n_sem (must match checkpoint; recursive ckpt = 15).
        n_sem_central:         int  = 15,
        n_steps:           int  = 30,
        device:            str  = 'cuda',
        ua_use_seen_mask:  bool = True,
        prior_a:           float = 0.1,
        prior_b:           float = 0.1,
        init_belief_a:     float = 0.0,
        init_belief_b:     float = 0.0,
        ub_reward_alpha:   float = 1.0,
        ub_reward_beta:    float = 0.5,
        ub_reward_use_attn: bool = True,
        ub_reward_attn_heads: int = 4,
    ):
        self.device = device if torch.cuda.is_available() or device == 'cpu' else 'cpu'
        self.n_steps = n_steps
        self.ua_use_seen_mask = ua_use_seen_mask

        # ── Unified UAB belief model ────────────────────────────────
        uab_net = UNet3DPushUnified(prior_a=prior_a, prior_b=prior_b)
        if uab_belief_ckpt is not None:
            self._load_state(uab_net, uab_belief_ckpt)
        self.uab_belief = UABUnifiedBeliefTrack(
            uab_net,
            init_belief_a=init_belief_a,
            init_belief_b=init_belief_b,
            device=self.device,
        )
        self.ua_belief = _BeliefView(self.uab_belief, 'belief_a')
        self.ub_belief = _BeliefView(self.uab_belief, 'belief_b')

        # ── semantic belief map, recursive (f_scene) ────────────────
        central_net = BeliefUpdateNetRecursive(
            with_sweep=with_sweep_central,
            with_semantic=with_semantic_central,
            n_sem=n_sem_central,
        )
        if central_ckpt is not None:
            self._load_state(central_net, central_ckpt)
        self.central = CentralBeliefTrackRecursive(central_net, device=self.device)

        # ── DiVE RewardNets ─────────────────────────────────────────
        self.reward_ua = RewardNet(use_seen_mask=ua_use_seen_mask).to(self.device).eval()
        if ua_reward_ckpt is not None:
            self._load_state(self.reward_ua, ua_reward_ckpt)

        self.reward_ub = RewardNetUB(
            base_ch=16,
            alpha=ub_reward_alpha,
            beta=ub_reward_beta,
            learnable_ab=False,
            use_attn=ub_reward_use_attn,
            attn_heads=ub_reward_attn_heads,
        ).to(self.device).eval()
        if ub_reward_ckpt is not None:
            self._load_state(self.reward_ub, ub_reward_ckpt)

        # ── per-episode state ───────────────────────────────────────
        self.seen_mask = torch.zeros(1, 7, 14, device=self.device)
        self.t = 0
        self.log: list[StepLog] = []

        # Candidate-viewpoint restriction for the |C| sensitivity study.
        # None = all 98 (paper setting). Otherwise a bool (98,) mask applied to
        # the view logits before the argmax; the pretrained head is untouched.
        self.allowed_cams: Optional[torch.Tensor] = None

    def set_allowed_cams(self, cam_indices) -> None:
        """Restrict view selection to `cam_indices` (None / all 98 = no-op)."""
        if cam_indices is None or len(cam_indices) == 98:
            self.allowed_cams = None
            return
        mask = torch.zeros(98, dtype=torch.bool, device=self.device)
        mask[torch.as_tensor(list(cam_indices), dtype=torch.long)] = True
        self.allowed_cams = mask

    @staticmethod
    def _load_state(model: torch.nn.Module, ckpt_path: str):
        from inference.utils.ckpt_io import load_ckpt_into
        load_ckpt_into(model, ckpt_path, tag=ckpt_path)

    # `step` is identical to the non-recursive sibling; `select_action` adds the
    # `allowed_cams` mask (|C| sensitivity study) and the `force_mode` override
    # (mode-switching ablations). Both are re-stated here rather than inherited
    # because this class replaces the type of self.central; the logic touches
    # only public state (uab_belief, central, seen_mask, reward_*).
    @torch.no_grad()
    def select_action(self, swept_dive_candidates: torch.Tensor,
                      force_view: bool = False,
                      selector: str = 'reward_logit',
                      force_mode: str = None):
        ua = self.uab_belief.belief_a
        ub = self.uab_belief.belief_b

        if self.ua_use_seen_mask:
            logit_ua = self.reward_ua(ua, self.seen_mask).squeeze(0)
        else:
            logit_ua = self.reward_ua(ua).squeeze(0)
        sw_5d = swept_dive_candidates.unsqueeze(0).to(self.device, dtype=torch.float32)
        logit_ub = self.reward_ub(ub, sw_5d).squeeze(0)

        # Restrict the argmax to the allowed candidate viewpoints. The mode-level
        # gain G_view is a sum over voxels of the visibility state and does not
        # depend on |C|, so only the within-mode ranking is masked here.
        logit_ua_sel = logit_ua
        if self.allowed_cams is not None:
            logit_ua_sel = logit_ua.masked_fill(~self.allowed_cams, float('-inf'))

        ua_max = float(logit_ua_sel.max())
        ub_max = float(logit_ub.max())
        ua_arg = int(logit_ua_sel.argmax())
        ub_arg = int(logit_ub.argmax())

        step_norm = self.t + 1
        score_ua_bern = float((2.0 * ua * (1.0 - ua)).sum() / step_norm)
        score_ub_bern = float((2.0 * ub * (1.0 - ub)).sum() / step_norm)
        score_ua_sum  = float(ua.sum())
        score_ub_sum  = float(ub.sum())

        if selector == 'reward_logit':
            A, B = ua_max, ub_max
        elif selector == 'bernoulli_vs_ub_sum':
            A, B = score_ua_bern, score_ub_sum
        elif selector == 'bernoulli_legacy':
            A, B = score_ua_bern, score_ub_bern
        else:
            raise ValueError(f'unknown selector: {selector}')

        # `force_mode` overrides the learned comparison with an externally
        # supplied mode (heuristic mode-switching ablations). The mode-specific
        # policies still choose *which* viewpoint or push, so the ablation
        # isolates the mode decision alone.
        if force_mode == 'push':
            action = ('push', ub_arg)
        elif force_mode == 'view' or force_view or B <= A:
            action = ('view', ua_arg)
        else:
            action = ('push', ub_arg)

        return action, dict(
            force_mode=force_mode,
            score_ua=score_ua_bern, score_ub=score_ub_bern,
            score_ua_bern=score_ua_bern, score_ub_bern=score_ub_bern,
            score_ua_sum =score_ua_sum,  score_ub_sum =score_ub_sum,
            selector=selector,
            ua_max=ua_max, ub_max=ub_max,
            ua_arg=ua_arg, ub_arg=ub_arg,
            logit_ua_all=logit_ua.detach().cpu().numpy(),
            logit_ub_all=logit_ub.detach().cpu().numpy(),
        )

    @torch.no_grad()
    def step(self, sim_obs: dict, action=None, sel_meta=None):
        if action is None:
            action, sel_meta = self.select_action()

        occ_d   = sim_obs['occ_view_dive']
        swept_c = sim_obs['swept_central']
        sem_c   = sim_obs['sem_voxel_central']
        if 'occ_view_central' in sim_obs and sim_obs['occ_view_central'] is not None:
            occ_c = sim_obs['occ_view_central']
        else:
            occ_c = dive_to_central(
                occ_d if isinstance(occ_d, np.ndarray) else occ_d.cpu().numpy())

        occ_d_t = torch.as_tensor(occ_d, device=self.device).float()

        self.uab_belief.update(occ_d_t)
        self.central.update(occ_c, swept_c, sem_c)

        if action[0] == 'view':
            cam = action[1]
            self.seen_mask[0, cam // 14, cam % 14] = 1.0

        self.log.append(StepLog(
            t=self.t, action_type=action[0], action_idx=action[1],
            ua_max=sel_meta['ua_max'], ub_max=sel_meta['ub_max'],
            ua_argmax=sel_meta['ua_arg'], ub_argmax=sel_meta['ub_arg'],
        ))
        self.t += 1
        return action
