"""
DiVE + semantic belief map parallel tracker — orchestrator.

Three belief tracks run in parallel, all consuming the same per-step observation:
  - UA_belief : DiVE view-resolvable track, channel 0 of the unified UAB net.
  - UB_belief : DiVE push-resolvable track, channel 1 of the same net.
  - central   : (α, β) + optional sem_logits from BeliefUpdateNet. This is the
                track the reported metrics are computed from.

Action selection: max(RewardNet_UA(UA_belief)) vs max(RewardNetUB(UB_belief))
  → the larger of the two picks the action type, the argmax within that head
  picks the concrete viewpoint or push.

Push action space: 20 belief-driven frontier candidates from
sim.push_provider.FrontierPushSpecProvider — freeform (x_voxel, y_voxel,
angle_deg), regenerated every step because the frontier depends on the belief.

The observations come from inference/sim_wrapper.py (SimWrapper), which exposes
  reset()               -> initial scene state
  capture_view(cam_idx) -> (occ_view_dive, sem_voxel_central)
  apply_push(spec)      -> executes the push; a post-push view is then captured
  get_gt_id_map()       -> (W, H, D) int, used only for the monitoring metric

The DEFAULT_*_CKPT constants below are legacy paths from the research tree and
are never used by the released pipeline: run_eval.py always passes explicit
checkpoint paths (see scripts/download_checkpoints.sh).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# ── Path setup ───────────────────────────────────────────────────────────────
_PKG_ROOT = Path(__file__).resolve().parents[1]   # repo root
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

# semantic belief map estimator (f_scene)
from belief_map.net import BeliefUpdateNet

# DiVE models
from inference.dive.belief_net import UNet3DPushUnified
from inference.dive.reward_net_ua import RewardNet
from inference.dive.reward_net_ub import RewardNetUB


# ── Constants ────────────────────────────────────────────────────────────────
N_CAMS    = 98
DIVE_SHAPE    = (60, 120, 80)    # (D=z, H=x, W=y)         DiVE convention
CENTRAL_SHAPE = (120, 60, 80)    # (W=x, H=z, D=y)         belief_map convention
CENTER_CAM_IDX = 49              # 7×14 grid → row 3, col 7 → 3*14+7 = 49


N_CAM_COLS = 14   # xi axis (azim) — 14 cams uniformly span voxel x ∈ [0, VOXEL_W=120]
N_CAM_ROWS = 7    # zi axis (elev)
TOP_ZI_CHOICES = (4, 5, 6)   # upper 3 elevation rows (higher zi = higher cam_idx = higher view)


def cam_xi_from_x_voxel(x_voxel: int, voxel_w: int = 120,
                        n_cols: int = N_CAM_COLS) -> int:
    """Map a push's x_voxel ∈ [0, voxel_w) to the closest cam-grid column xi.

    Cameras are placed at xs[xi] = linspace(-SHELF_W/2, SHELF_W/2, n_cols),
    which lands on voxel x = xi * voxel_w / (n_cols - 1) (continuous). Inverse
    is xi = x_voxel * (n_cols - 1) / voxel_w, rounded + clamped to [0, n_cols-1].
    """
    xi = int(round(x_voxel * (n_cols - 1) / voxel_w))
    return max(0, min(n_cols - 1, xi))


def pick_post_push_cam(x_voxel: int, rng, top_zi=TOP_ZI_CHOICES,
                       n_cols: int = N_CAM_COLS) -> int:
    """Post-push view = (closest xi to push x_voxel, random zi ∈ top 3 rows).

    Higher zi → higher cam_idx (cam = zi*14 + xi) → higher physical elevation,
    so picking from the top 3 rows means looking down onto the just-pushed area.
    """
    xi = cam_xi_from_x_voxel(x_voxel, n_cols=n_cols)
    zi = int(rng.choice(list(top_zi)))
    return zi * n_cols + xi


def post_view_from_swept(swept_dive, rng, zi=3):
    """[Deprecated] Legacy swept-map-based post-view picker. Kept only for
    back-compat with older eval scripts. New code should call
    pick_post_push_cam(spec.x_voxel, rng) directly.
    """
    import numpy as np
    occ_vx = np.where(swept_dive.sum(axis=(0, 2)) > 0)[0]
    if len(occ_vx) == 0:
        return CENTER_CAM_IDX
    vx_min, vx_max = int(occ_vx.min()), int(occ_vx.max())
    xi_min = max(0,  round(vx_min / 119 * 13))
    xi_max = min(13, round(vx_max / 119 * 13))
    xi = int(rng.integers(xi_min, xi_max + 1))
    return zi * 14 + xi

# Legacy checkpoint paths from the research tree. Kept only so that older
# callers importing these names keep working; run_eval.py overrides all of them.
DEFAULT_UA_BELIEF_CKPT = 'checkpoints_zoo/dive/UA_ycb_v3_TFanneal/best.pth'
DEFAULT_UB_BELIEF_CKPT = 'checkpoints_zoo/dive/UB_ycb_v3_marginal/best.pth'
DEFAULT_UA_REWARD_CKPT = 'checkpoints_zoo/dive/reward_UA_sigz_v1/best.pth'
DEFAULT_UB_REWARD_CKPT = 'checkpoints_zoo/dive/reward_UB_full_siamese_sigz_b05_attn/best.pth'
DEFAULT_CENTRAL_CKPT   = 'checkpoints/central/belief_update_scratch_ycb_v3_sem_ddp4_b64eff.pt'


# ── Axis adapters ────────────────────────────────────────────────────────────
def central_to_dive(arr_whD):
    """(..., W=120, H=60, D=80) → (..., D=60, H=120, W=80).
    Permute spatial axes (-3, -2, -1) → (-2, -3, -1)."""
    if isinstance(arr_whD, np.ndarray):
        ax = list(range(arr_whD.ndim))
        ax[-3], ax[-2] = ax[-2], ax[-3]
        return np.transpose(arr_whD, ax)
    elif isinstance(arr_whD, torch.Tensor):
        ax = list(range(arr_whD.ndim))
        ax[-3], ax[-2] = ax[-2], ax[-3]
        return arr_whD.permute(*ax).contiguous()
    raise TypeError(type(arr_whD))


def dive_to_central(arr_DHw):
    """Reverse: same permutation (involution)."""
    return central_to_dive(arr_DHw)


# ── Push spec provider ───────────────────────────────────────────────────────
# DiVE inference uses sim.push_provider.FrontierPushSpecProvider directly: the
# same belief-driven frontier sampling the scene data was collected with, and
# the same action set the push policy was trained on (policy/push,
# NUM_PUSHES=20). Swept maps for the RewardNetUB / BeliefUpdateNet inputs are
# sim-generated gripper-volume sweeps via SimWrapper.predict_swept_map_for(spec)
# — matching action.npz['swept_map'] in training — recomputed every step
# because the frontier depends on the belief state.


# ── Belief tracks ────────────────────────────────────────────────────────────
class UABUnifiedBeliefTrack:
    """Unified UA + UB belief (shared backbone, 2-channel head).

    Mirrors dive/train_UAB_unified.py exactly:
      input  : cat([belief_A, belief_B, voxel], dim=1)        (1, 3, D, H, W)
      logits = model(x)                                       (1, 2, D, H, W)
      belief_A = sigmoid(logits[:, 0:1])
      belief_B = sigmoid(logits[:, 1:2])

    Note: training is VIEW-ONLY (no swept_map channel). At inference the
    belief update is also driven by `voxel` = per-step pre-view occ
    (raycast first-surface, occ_view_dive). Swept maps are used only by
    the central tracker + RewardNetUB, never as a belief-net input.

    Self-feed update matches training's eval mode (validate(): both
    belief_a/belief_b feed back as sigmoid of own logit each step).
    """

    def __init__(self, model: UNet3DPushUnified,
                 init_belief_a: float = 0.0,
                 init_belief_b: float = 0.0,
                 device: str = 'cuda'):
        self.device = device
        self.model = model.to(device).eval()
        self.init_belief_a = float(init_belief_a)
        self.init_belief_b = float(init_belief_b)
        self.belief_a = torch.full((1, 1, *DIVE_SHAPE), init_belief_a, device=device)
        self.belief_b = torch.full((1, 1, *DIVE_SHAPE), init_belief_b, device=device)

    def reset(self):
        self.belief_a = torch.full(
            (1, 1, *DIVE_SHAPE), self.init_belief_a, device=self.device)
        self.belief_b = torch.full(
            (1, 1, *DIVE_SHAPE), self.init_belief_b, device=self.device)

    @torch.no_grad()
    def update(self, voxel_dive: torch.Tensor):
        """voxel_dive: (D, H, W) ∈ {0,1} — per-step ray-cast first-surface occ."""
        v = voxel_dive.to(self.device).view(1, 1, *DIVE_SHAPE).float()
        x = torch.cat([self.belief_a, self.belief_b, v], dim=1)   # (1, 3, D, H, W)
        logits = self.model(x)                                    # (1, 2, D, H, W)
        self.belief_a = torch.sigmoid(logits[:, 0:1].float())
        self.belief_b = torch.sigmoid(logits[:, 1:2].float())
        return self.belief_a, self.belief_b


# Back-compat shim: expose .belief as a per-channel view so that callers which
# dump belief slices keep working through `orch.ua_belief.belief` and
# `orch.ub_belief.belief`.
class _BeliefView:
    """Read/write proxy onto a single channel of UABUnifiedBeliefTrack."""

    def __init__(self, track: 'UABUnifiedBeliefTrack', attr: str):
        self._track = track
        self._attr = attr

    @property
    def belief(self):
        return getattr(self._track, self._attr)

    @belief.setter
    def belief(self, val):
        setattr(self._track, self._attr, val)


class CentralBeliefTrack:
    """Semantic belief map track: (α, β [, sem_logits]) + BeliefUpdateNet.

    Updated every step regardless of action type.
    Stores (1, W, H, D) tensors per spec.
    """

    def __init__(self, model: BeliefUpdateNet, device: str = 'cuda'):
        self.device = device
        self.model = model.to(device).eval()
        self.alpha = torch.ones((1, *CENTRAL_SHAPE), device=device)
        self.beta  = torch.ones((1, *CENTRAL_SHAPE), device=device)
        self.last_sem_logits = None                       # (1, n_sem, W, H, D) or None
        self.with_sweep    = model.with_sweep
        self.with_semantic = model.with_semantic
        self.n_sem         = model.n_sem

    @torch.no_grad()
    def update(self, occ_view_central, swept_central, sem_voxel_central):
        """All inputs (W, H, D) on host or device.
        occ_view_central : float in {0,1}
        swept_central    : float in {0,1} (zeros for view step)
        sem_voxel_central: int (per-view 2D→3D sparse semantic, class_scalar lookup inside model)
        """
        def _t(x, dtype):
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            return x.to(self.device, dtype=dtype).view(1, *CENTRAL_SHAPE)

        occ = _t(occ_view_central, torch.float32)
        kwargs = {}
        if self.with_sweep:
            kwargs['swept_map'] = _t(swept_central, torch.float32)
        if self.with_semantic:
            kwargs['semantic_map'] = _t(sem_voxel_central, torch.float32)

        out = self.model(self.alpha, self.beta, occ, **kwargs)
        if self.n_sem > 0:
            self.alpha, self.beta, self.last_sem_logits = out
        else:
            self.alpha, self.beta = out
            self.last_sem_logits = None
        return self.alpha, self.beta, self.last_sem_logits


# ── Orchestrator ─────────────────────────────────────────────────────────────
@dataclass
class StepLog:
    t: int
    action_type: str       # 'view' | 'push'
    action_idx: int
    ua_max: float
    ub_max: float
    ua_argmax: int
    ub_argmax: int


class Orchestrator:
    def __init__(
        self,
        uab_belief_ckpt:   Optional[str] = None,
        ua_reward_ckpt:    Optional[str] = None,
        ub_reward_ckpt:    Optional[str] = None,
        central_ckpt:      Optional[str] = None,
        with_sweep_central:    bool = True,
        with_semantic_central: bool = True,
        n_sem_central:         int  = 12,
        n_steps:           int  = 30,
        device:            str  = 'cuda',
        ua_use_seen_mask:  bool = True,    # the released view policy was trained with use_seen_mask=True
        # Unified belief priors (must match train_UAB_unified.py defaults).
        prior_a:           float = 0.1,
        prior_b:           float = 0.1,
        init_belief_a:     float = 0.0,
        init_belief_b:     float = 0.0,
        # RewardNetUB constructor knobs — must match training (see policy/push/train_reward_UB.py).
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
        # Back-compat shims: orch.ua_belief.belief / orch.ub_belief.belief
        # still work for downstream consumers (visualization, slice dumps).
        self.ua_belief = _BeliefView(self.uab_belief, 'belief_a')
        self.ub_belief = _BeliefView(self.uab_belief, 'belief_b')

        # ── semantic belief map (f_scene) ───────────────────────────
        central_net = BeliefUpdateNet(
            with_sweep=with_sweep_central,
            with_semantic=with_semantic_central,
            n_sem=n_sem_central,
        )
        if central_ckpt is not None:
            self._load_state(central_net, central_ckpt)
        self.central = CentralBeliefTrack(central_net, device=self.device)

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
        self.seen_mask = torch.zeros(1, 7, 14, device=self.device)   # cam-grid 'has been viewed' bitmap
        self.t = 0
        self.log: list[StepLog] = []

    # ── helpers for ckpt loading ────────────────────────────────────
    @staticmethod
    def _load_state(model: torch.nn.Module, ckpt_path: str):
        # delegate to the shared ckpt extractor (handles {epoch, model, ...} wrappers)
        from inference.utils.ckpt_io import load_ckpt_into
        load_ckpt_into(model, ckpt_path, tag=ckpt_path)

    # ── action selection ────────────────────────────────────────────
    @torch.no_grad()
    def select_action(self, swept_dive_candidates: torch.Tensor,
                      force_view: bool = False,
                      selector: str = 'reward_logit'):
        """Chooses 'view' vs 'push', then picks the specific cam / push via
        RewardNet argmax within the chosen head.

        Three selectors implemented (compare scalar A vs B at each step):

          'reward_logit'  (default) — RewardNet logit max comparison.
              A = ua_max  = max_c  RewardNet_UA(belief_ua)
              B = ub_max  = max_c  RewardNet_UB(belief_ub, swept_candidates)
              action = 'push' if B > A else 'view'
              Rationale: both logits are distilled z-scored L1-gain predictions —
              same space, directly comparable.

          'bernoulli_vs_ub_sum'   — the default used by run_eval.py.
              A = score_ua_bern  = Σ 2·UA·(1-UA) / (t+1)
              B = score_ub_sum   = Σ UB
              action = 'push' if B > A else 'view'
              Rationale: Bernoulli total uncertainty for view; raw UB belief
              mass for push (UB itself predicts marginal push visibility).

          'bernoulli_legacy'      — both Bernoulli (original).
              A = score_ua_bern, B = score_ub_bern = Σ 2·UB·(1-UB) / (t+1)
              (For backward compatibility / ablation; previously gave push=0.)

        All four diagnostic scalars are returned in sel_meta regardless of
        which selector is active.
        """
        ua = self.uab_belief.belief_a                              # (1, 1, D, H, W) ∈ [0,1]
        ub = self.uab_belief.belief_b

        # ── RewardNet logits (= predicted z-scored L1 gain) ────────────────
        if self.ua_use_seen_mask:
            logit_ua = self.reward_ua(ua, self.seen_mask).squeeze(0)
        else:
            logit_ua = self.reward_ua(ua).squeeze(0)               # (98,)
        sw_5d = swept_dive_candidates.unsqueeze(0).to(self.device, dtype=torch.float32)
        logit_ub = self.reward_ub(ub, sw_5d).squeeze(0)            # (N,)
        ua_max = float(logit_ua.max())
        ub_max = float(logit_ub.max())
        ua_arg = int(logit_ua.argmax())
        ub_arg = int(logit_ub.argmax())

        # ── All four diagnostic scalars (independent of selector) ──────────
        step_norm = self.t + 1
        score_ua_bern = float((2.0 * ua * (1.0 - ua)).sum() / step_norm)
        score_ub_bern = float((2.0 * ub * (1.0 - ub)).sum() / step_norm)
        score_ua_sum  = float(ua.sum())
        score_ub_sum  = float(ub.sum())

        # ── decision rule ───────────────────────────────────────────────────
        if selector == 'reward_logit':
            A, B = ua_max, ub_max
        elif selector == 'bernoulli_vs_ub_sum':
            A, B = score_ua_bern, score_ub_sum
        elif selector == 'bernoulli_legacy':
            A, B = score_ua_bern, score_ub_bern
        else:
            raise ValueError(f'unknown selector: {selector}')

        if force_view or B <= A:
            action = ('view', ua_arg)
        else:
            action = ('push', ub_arg)

        return action, dict(
            # legacy keys (kept for back-compat with existing log parsers)
            score_ua=score_ua_bern, score_ub=score_ub_bern,
            # new diagnostic keys
            score_ua_bern=score_ua_bern, score_ub_bern=score_ub_bern,
            score_ua_sum =score_ua_sum,  score_ub_sum =score_ub_sum,
            selector=selector,
            ua_max=ua_max, ub_max=ub_max,
            ua_arg=ua_arg, ub_arg=ub_arg,
            logit_ua_all=logit_ua.detach().cpu().numpy(),
            logit_ub_all=logit_ub.detach().cpu().numpy(),
        )

    # ── step ────────────────────────────────────────────────────────
    @torch.no_grad()
    def step(self, sim_obs: dict, action=None, sel_meta=None):
        """sim_obs is built by the caller, normally from SimWrapper:
            'occ_view_dive'    : (D=60, H=120, W=80) float in {0,1}  — raycast
                                  first-surface occ, used as `voxel_t` input to
                                  the unified UAB belief net (matches training:
                                  common/dataset_UAB_unified.py — voxel_maps[:, t]).
            'occ_view_central' : (W=120, H=60, D=80) float in {0,1}  — OPTIONAL.
                                  Used for BeliefUpdateNet update. Should be
                                  computed via _depth_view_to_voxel_occ (first-surface
                                  raycast) — that is what BeliefUpdateNet was trained on.
                                  If absent, falls back to dive→central axis permutation
                                  of occ_view_dive.
            'sem_voxel_central': (W=120, H=60, D=80) int — sparse, class_scalar lookup inside central model
            'swept_dive'       : (D, H, W) float — NOT used by UAB belief net
                                  (training is view-only, no swept channel).
                                  Carried in sim_obs only for symmetry / logging.
            'swept_central'    : (W, H, D) float — central tracker's swept input.

        Pass `action` and `sel_meta` to skip internal select_action.
        """
        if action is None:
            action, sel_meta = self.select_action()

        occ_d  = sim_obs['occ_view_dive']
        swept_c = sim_obs['swept_central']
        sem_c  = sim_obs['sem_voxel_central']
        if 'occ_view_central' in sim_obs and sim_obs['occ_view_central'] is not None:
            occ_c = sim_obs['occ_view_central']
        else:
            occ_c = dive_to_central(
                occ_d if isinstance(occ_d, np.ndarray) else occ_d.cpu().numpy())

        # to tensor on device
        occ_d_t  = torch.as_tensor(occ_d, device=self.device).float()

        # ── update all 3 tracks (every step, regardless of action) ──
        # UAB unified belief: one forward, NO swept input (training=view-only).
        self.uab_belief.update(occ_d_t)
        self.central.update(occ_c, swept_c, sem_c)

        # update seen_mask if view action
        if action[0] == 'view':
            cam = action[1]
            self.seen_mask[0, cam // 14, cam % 14] = 1.0

        # log
        self.log.append(StepLog(
            t=self.t, action_type=action[0], action_idx=action[1],
            ua_max=sel_meta['ua_max'], ub_max=sel_meta['ub_max'],
            ua_argmax=sel_meta['ua_arg'], ub_argmax=sel_meta['ub_arg'],
        ))
        self.t += 1
        return action


# Module is library-only. The end-to-end entry point is inference/run_eval.py.
