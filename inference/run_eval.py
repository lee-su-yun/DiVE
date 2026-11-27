"""DiVE evaluation runner — Isaac Lab adapter.

Drives the full DiVE pipeline on top of inference/sim_wrapper.py: the unified
UA/UB visibility belief plus the UA/UB RewardNets pick an action, while the
recursive semantic belief map (BeliefUpdateNetRecursive, wrapped by
OrchestratorRecursive) runs alongside and carries the reported metrics.

Invoke from the repo root with PYTHONPATH=$PWD:

    python -m inference.run_eval --uab_belief ... --out_dir ...

Model architecture n_sem (= --n_sem_model, default 15) must match the
checkpoint. Metric / evaluation n_sem (= --n_sem, default 12) covers the
active YCB classes; the 3 reserved slots are part of the network's output
channels but never scored.

Action space matches training (policy/push, NUM_PUSHES=20):
  - 98 fixed cameras (UA RewardNet → argmax over the 7x14 cam grid).
  - 20 BELIEF-DRIVEN FRONTIER push candidates (FrontierPushSpecProvider), with
    swept_map computed per step from the sim's gripper geometry
    (sw.predict_swept_map_for(spec) — a gripper-volume sweep matching
    action.npz['swept_map'] in training).

Per step t:
  1. Build the 20 candidate FreeformPushSpecs from the current central belief.
     Stack their sim-style gripper-volume swept maps → swept_dive (20, D, H, W).
  2. Orchestrator.select_action(swept_dive): the mode is chosen by comparing two
     scalars, per --action_selector (default 'bernoulli_vs_ub_sum': the UA
     Bernoulli uncertainty sum 2p(1-p)/(t+1) against the UB belief mass sum).
     Within the chosen mode, the RewardNet argmax (UA: 98 cams, UB: 20 pushes)
     picks the concrete action. With --force_view the push branch is disabled
     and every step is a 'view'.
  3. Execute in sim (capture_view, or apply_push + post-view capture).
  4. Orchestrator.step() updates all three tracks (UA/UB DiVE beliefs + central).
     Central uses _depth_view_to_voxel_occ (training-distribution first-surface
     raycast occ), NOT the σ_o-style height-map occ that goes to the DiVE belief.
  5. central_metrics vs GT — the same definitions the CNABU baseline uses, so
     the two runs are directly comparable.

Per-step output schema:
  miou / iou_occ / iou_free / sem_miou / sem_acc / mapped_frac
plus DiVE-specific:
  score_ua / score_ub / ua_arg / ub_arg / ua_max / ub_max / ua_topk / ub_topk
  / push payload (push step only).

Test-set sharding: --test_set_dir + --episode_start / --episode_end +
--summary_suffix. inference/run_eval.sh drives the reference run as 4 parallel
shards of 25 episodes each and then calls inference/aggregate.py.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# ── 1) parse our args, then init Isaac Lab BEFORE any sim_wrapper-derived imports
parser = argparse.ArgumentParser(description='DiVE + central tracker pipeline runner')
parser.add_argument('--uab_belief',   required=True,
                    help='Unified UA+UB visibility belief ckpt (dive/train_UAB_unified.py '
                         'output, e.g. checkpoints/DiVE_decomposed_visibility_estimator/best.pth).')
parser.add_argument('--ua_reward',    required=True)
parser.add_argument('--ub_reward',    required=True)
parser.add_argument('--central_ckpt', required=True)
parser.add_argument('--out_dir',      required=True)
parser.add_argument('--num_episodes', type=int, default=1)
parser.add_argument('--action_budget', type=int, default=30)
parser.add_argument('--seed',         type=int, default=0)
parser.add_argument('--push_dist',    type=float, default=0.25)
parser.add_argument('--n_sem',        type=int, default=12,
                    help='Active class count for metric/eval (YCB v3 = 12).')
parser.add_argument('--n_sem_model',  type=int, default=15,
                    help='Model architecture n_sem (must match the checkpoint; the '
                         'released belief map estimator was trained with n_sem=15).')
parser.add_argument('--scene_mode',   choices=['high', 'low', 'super_high'], default='low')
parser.add_argument('--mode_selector', type=str, default='learned',
                    help='Mode-switching rule (heuristic ablations). "learned" '
                         '(default) compares G_view vs G_push as in the paper. '
                         '"periodic:K" pushes on every K-th action step. '
                         '"random:P" pushes with probability P. In every case '
                         'pi_view / pi_push still choose the concrete action, '
                         'so only the mode decision changes.')
parser.add_argument('--push_selector', type=str, default='policy',
                    choices=['policy', 'random'],
                    help='Which push to execute once the push mode is chosen. '
                         '"policy" (default) = pi_push argmax. "random" = a '
                         'uniformly sampled candidate, isolating the value of '
                         'the push-ranking policy from the mode decision.')
parser.add_argument('--push_beyond_m', type=float, default=None,
                    help='Push length: metres travelled past the frontier '
                         'surface. None (default) = the scene module\'s '
                         'PUSH_BEYOND_M = 0.15, the paper setting. Applied to '
                         'both the executed push and the predicted swept map.')
parser.add_argument('--angle_max_deg', type=float, default=None,
                    help='Half-width of the push direction range about the '
                         'shelf depth axis. None (default) = 30 deg, the paper '
                         'setting. Still clamped near the shelf edges.')
parser.add_argument('--view_subset',  type=str, default='all',
                    help='Restrict the candidate viewpoint set C for the |C| '
                         'sensitivity study. "all" (default) = the 98-view grid '
                         'used in the paper. Also: uniform:RxC, narrow:RxC, '
                         'random:K[:SEED]. See inference/view_subsets.py. The '
                         'view head is not retrained; its logits are masked.')
parser.add_argument('--n_push_cands', type=int, default=20,
                    help='Number of frontier push candidates per step. Training '
                         'NUM_PUSHES=20 — leave at 20 to match training.')
parser.add_argument('--occ_thresh',   type=float, default=0.5,
                    help='Frontier occupancy threshold on central prob map.')
parser.add_argument('--n_initial_random_views', type=int, default=1,
                    help='Training-aligned: force the first N steps to be views '
                         'with a random cam index. UA/UB beliefs were trained on '
                         'rollouts whose t=0 is always a random pre-push view '
                         '(common/dataset_UAB_unified.py:_sample_views).')
parser.add_argument('--action_selector', type=str, default='bernoulli_vs_ub_sum',
                    choices=['reward_logit', 'bernoulli_vs_ub_sum', 'bernoulli_legacy'],
                    help='Compare scalar A vs B at each step (see Orchestrator.select_action).')
parser.add_argument('--no_headless',  dest='headless', action='store_false')
parser.add_argument('--keep_open',    action='store_true')
parser.add_argument('--force_view',   action='store_true',
                    help='Disable push selection: every step is a view (view-only pipeline).')
parser.add_argument('--ua_use_seen_mask', action='store_true', default=True,
                    help='Pass seen_mask to the UA RewardNet. The released view '
                         'selection policy was trained with use_seen_mask=True, '
                         'so the default is True.')
parser.add_argument('--no_ua_use_seen_mask', dest='ua_use_seen_mask',
                    action='store_false')
parser.add_argument('--ub_reward_alpha', type=float, default=1.0,
                    help='Match training (--alpha; train default 1.0).')
parser.add_argument('--ub_reward_beta',  type=float, default=0.5,
                    help='Match training (--beta; the released push policy used 0.5).')
parser.add_argument('--ub_reward_no_attn', action='store_true',
                    help='Disable ActionAttention. Default = enabled, matching how '
                         'the released push selection policy was trained.')
parser.add_argument('--ub_reward_attn_heads', type=int, default=4)
parser.add_argument('--sigma_o',      type=str, default='',
                    help='CNABU σ_o ckpt for v2 sidecar (DiVE plan + σ_o belief metric).')
parser.add_argument('--sigma_m',      type=str, default='',
                    help='CNABU σ_m ckpt for v2 sidecar (used on push).')
parser.add_argument('--viz_dir',      type=str, default='')
parser.add_argument('--viz_episodes', type=int, default=0)
parser.add_argument('--test_set_dir', type=str, default='')
parser.add_argument('--episode_start', type=int, default=0)
parser.add_argument('--episode_end',   type=int, default=-1)
parser.add_argument('--summary_suffix', type=str, default='')
parser.add_argument('--dump_sem', action='store_true',
                    help='also save per-step (pred_occ, sem_3d_pred) and gt_id '
                         'as sem_dump_ep<NNN>.npz for semantic viz.')
parser.add_argument('--save_figure_dump', action='store_true',
                    help='save per-step full-3D UA/UB belief, central alpha/beta, '
                         'and the post-view RGB image to figure_dump_ep<NNN>.npz '
                         'for paper-figure rendering.')
parser.add_argument('--warmup_sleep', type=float, default=0.0,
                    help='seconds to sleep after sim/scene is ready and before '
                         'the first inference step (so you can frame the GUI '
                         'window for screenshots).')
parser.add_argument('--force_action_json', type=str, default='',
                    help='JSON file with a per-step action override. Format: '
                         '[{"step":N, "kind":"view|push", "idx":INT}, ...]. '
                         'Used to replay a recorded trajectory exactly.')
parser.set_defaults(headless=True)
args, remaining = parser.parse_known_args()
sys.argv = [sys.argv[0]] + remaining

# ── 2) Init Isaac Lab
_PKG_ROOT = Path(__file__).resolve().parents[1]   # repo root
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from inference.sim_wrapper import init_isaac_lab, SimWrapper        # noqa: E402
init_isaac_lab(headless=args.headless)

# ── 3) Now safe to import everything else
import torch                                              # noqa: E402

from inference.orchestrator import (         # noqa: E402
    OrchestratorRecursive as Orchestrator,
    CENTER_CAM_IDX, CENTRAL_SHAPE, DIVE_SHAPE,
    pick_post_push_cam,
)
from inference.sim.push_provider import FrontierPushSpecProvider     # noqa: E402
from inference.utils.voxel_projection import depth_semantic_to_voxel  # noqa: E402
from inference.utils.bev_projection import project_semantic_3d_to_2d  # noqa: E402

# σ_o sidecar (v2: DiVE plan + CNABU belief metric) is optional and NOT shipped
# in this package (it needs run_mem_pipeline.py + the CNABU paper repo). It is
# imported lazily below only if both --sigma_o and --sigma_m are passed.


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPS    = 1e-8


# ── helpers ──────────────────────────────────────────────────────────────────
def view_occ_and_sem(obs, env_origin_np, cd_module):
    """Per-view occ + semantic for the CENTRAL path, matching its training.

    BeliefUpdateNet was trained on `occ_pre.npz['occupied']` from
    inference/sim/scene_low_occlusion.py:depth_stack_to_occ, which uses
    `_depth_view_to_voxel_occ` (raycast first-surface). Using sim_wrapper's
    occ_view_dive (sem-masked + height-map column-fill) for the central path
    instead drops mIoU ~0.85 → 0.32.

    Returns:
        occ_view_central : (W=120, H=60, D=80) float32 0/1
        sem_voxel_central: (W=120, H=60, D=80) int32
    """
    depth_v = np.nan_to_num(obs.depth, nan=0.0)
    occ_hwd = cd_module._depth_view_to_voxel_occ(
        depth_v, obs.cam_pos, obs.cam_quat_wxyz, env_origin_np,
    )                                                                # (H=60, W=120, D=80) = DiVE frame
    occ_view_central = occ_hwd.transpose(1, 0, 2).astype(np.float32) # (W, H, D)        central frame
    occ_view_dive    = occ_hwd.astype(np.float32)                    # (H, W, D)        DiVE frame
    # NOTE: occ_view_dive (raycast first-surface) is what the unified UA/UB
    # belief net was trained on — see preprocessing/visibility/data_preprocessing.py
    # (data["occupied"] = depth_stack_to_occ → _depth_view_to_voxel_occ). The
    # previously used obs.occ_view_dive (sem-masked + height_map column-fill) is
    # a different representation and is OOD for the UA/UB belief.

    sem_voxel_central = depth_semantic_to_voxel(
        depth_v, obs.semantic_2d, obs.cam_pos, obs.cam_quat_wxyz, env_origin_np,
    ).astype(np.int32)
    return occ_view_central, sem_voxel_central, occ_view_dive


def central_metrics(orch: Orchestrator, gt_id_whd: np.ndarray, n_sem: int) -> dict:
    """Metric definitions shared with the CNABU baseline runner, so that
    cross-run comparison is direct.
    """
    alpha = orch.central.alpha[0]
    beta  = orch.central.beta[0]
    prob_whD = (alpha / (alpha + beta + EPS)).cpu().numpy()        # (W, H, D)
    pred_occ = prob_whD > 0.5
    gt_occ   = gt_id_whd > 0
    inter = np.logical_and(pred_occ, gt_occ).sum()
    union = np.logical_or(pred_occ, gt_occ).sum()
    iou_occ  = float(inter / union) if union > 0 else 0.0
    iou_free = float(
        np.logical_and(~pred_occ, ~gt_occ).sum() /
        (np.logical_or(~pred_occ, ~gt_occ).sum() + EPS)
    )

    if orch.central.last_sem_logits is not None:
        sem_3d_pred = orch.central.last_sem_logits[0].argmax(dim=0).cpu().numpy().astype(np.int32)
        gt3 = gt_id_whd.astype(np.int32)

        # ── 3D metrics (training-style) ────────────────────────────────
        # sem_acc_3d_occ: pred==gt over GT-occupied voxels only
        obj_mask = gt3 > 0
        if obj_mask.any():
            sem_acc_3d_occ = float((sem_3d_pred[obj_mask] == gt3[obj_mask]).mean())
        else:
            sem_acc_3d_occ = float('nan')
        # sem_miou_3d: per-class IoU over all 3D voxels (free + objects)
        ious_3d = []
        for c in range(n_sem):
            p = (sem_3d_pred == c); g = (gt3 == c)
            u = np.logical_or(p, g).sum()
            if u > 0:
                ious_3d.append(float(np.logical_and(p, g).sum()) / float(u))
        sem_miou_3d = float(np.mean(ious_3d)) if ious_3d else 0.0

        # ── 2D BEV metrics (cnabu-style) ───────────────────────────────
        # Standard BEV projection: pred uses pred_occ for height selection,
        # GT uses gt_occ. Same formula as the CNABU baseline runner.
        sem_2d_pred = project_semantic_3d_to_2d(sem_3d_pred, pred_occ.astype(np.float32))
        sem_2d_gt   = project_semantic_3d_to_2d(gt3, gt_occ.astype(np.float32))
        # Oracle-height BEV projection (Option A): pred semantic is queried
        # at the GT-occupied topmost voxel — isolates semantic accuracy from
        # occupancy prediction error. Reported alongside the standard metric.
        sem_2d_pred_oracle = project_semantic_3d_to_2d(
            sem_3d_pred, gt_occ.astype(np.float32))

        def _miou(pred2d, gt2d, n):
            ious = []
            for c in range(n):
                p = (pred2d == c); g = (gt2d == c)
                u = np.logical_or(p, g).sum()
                if u > 0:
                    ious.append(float(np.logical_and(p, g).sum()) / float(u))
            return float(np.mean(ious)) if ious else 0.0

        sem_miou        = _miou(sem_2d_pred, sem_2d_gt, n_sem)
        sem_miou_oracle = _miou(sem_2d_pred_oracle, sem_2d_gt, n_sem)
        sem_acc         = float((sem_2d_pred        == sem_2d_gt).mean())
        sem_acc_oracle  = float((sem_2d_pred_oracle == sem_2d_gt).mean())
    else:
        sem_miou = sem_miou_oracle = float('nan')
        sem_acc  = sem_acc_oracle  = float('nan')
        sem_miou_3d = sem_acc_3d_occ = float('nan')

    mapped_frac = float(np.logical_or(prob_whD > 0.7, prob_whD < 0.3).mean())
    out = {
        "iou_occ": iou_occ, "iou_free": iou_free,
        "miou":    (iou_occ + iou_free) / 2,
        "sem_miou": sem_miou, "sem_acc": sem_acc,
        "sem_miou_oracle": sem_miou_oracle, "sem_acc_oracle": sem_acc_oracle,
        "sem_miou_3d": sem_miou_3d, "sem_acc_3d_occ": sem_acc_3d_occ,
        "mapped_frac": mapped_frac,
    }
    # extra arrays for --dump_sem (ignored by _make_log_entry, used by run_episode)
    out["_pred_occ"]    = pred_occ.astype(bool)
    if orch.central.last_sem_logits is not None:
        out["_sem_3d_pred"] = sem_3d_pred.astype(np.uint8)
    else:
        out["_sem_3d_pred"] = None
    return out


def topk_pairs(arr, k=5):
    if arr is None:
        return None
    arr = np.asarray(arr).reshape(-1)
    k = min(k, arr.size)
    idx = np.argpartition(-arr, k - 1)[:k]
    idx = idx[np.argsort(-arr[idx])]
    return [{"idx": int(i), "score": float(arr[i])} for i in idx]


# ── per-episode loop ─────────────────────────────────────────────────────────
EPS_PROB = 1e-8


def _central_prob_whD(orch: Orchestrator) -> np.ndarray:
    """(W, H, D) numpy occupancy prob from central α / (α + β)."""
    a = orch.central.alpha[0]
    b = orch.central.beta[0]
    return (a / (a + b + EPS_PROB)).detach().cpu().numpy()


def _build_candidate_swepts(sw: SimWrapper, specs):
    """For each FreeformPushSpec, compute the sim's gripper-volume swept map
    (matches action.npz['swept_map'] in training). Returns:

      swept_central_np : (N, W, H, D) float32 0/1
      swept_dive_np    : (N, D=60, H=120, W=80) float32 0/1
        = central .transpose(0, 2, 1, 3) — same convention as
          sim.push_provider.FrontierPushSpecProvider.swept_maps_dive().
    """
    n = len(specs)
    swept_c = np.zeros((n, *CENTRAL_SHAPE), dtype=np.float32)
    for i, s in enumerate(specs):
        swept_c[i] = sw.predict_swept_map_for(s).astype(np.float32)
    swept_d = swept_c.transpose(0, 2, 1, 3).astype(np.float32)
    return swept_c, swept_d


def _make_log_entry(t, action, sel_meta, m, step_time,
                    push_payload=None, post_view_cam=None,
                    is_push_bundle_duplicate=False,
                    plan_time=0.0, sim_time=0.0,
                    belief_update_time=0.0, metric_time=0.0,
                    step_time_total=0.0):
    """Build one log entry dict. `is_push_bundle_duplicate=True` marks the
    2nd of the two budget slots that a push action consumes (same metric).

    Timing convention:
      step_time         = plan + belief update (+ sidecar) — sim and metric EXCLUDED
      step_time_total   = plan + sim + belief update + metric — full wall-clock"""
    return {
        "step":      int(t),
        "action":    action[0],
        "view":      int(action[1]) if action[0] == 'view' else -1,
        "push_idx":  int(action[1]) if action[0] == 'push' else -1,
        "push":      push_payload,
        "post_view_cam": int(post_view_cam) if post_view_cam is not None else -1,
        "is_push_bundle_duplicate": bool(is_push_bundle_duplicate),
        "score_ua":  float(sel_meta.get('score_ua', float('nan'))),
        "score_ub":  float(sel_meta.get('score_ub', float('nan'))),
        "score_ua_bern": float(sel_meta.get('score_ua_bern', float('nan'))),
        "score_ub_bern": float(sel_meta.get('score_ub_bern', float('nan'))),
        "score_ua_sum":  float(sel_meta.get('score_ua_sum',  float('nan'))),
        "score_ub_sum":  float(sel_meta.get('score_ub_sum',  float('nan'))),
        "selector":      sel_meta.get('selector', ''),
        "ua_arg":    int(sel_meta['ua_arg']),
        "ub_arg":    int(sel_meta['ub_arg']),
        "ua_max":    float(sel_meta.get('ua_max', float('nan'))),
        "ub_max":    float(sel_meta.get('ub_max', float('nan'))),
        "ua_topk":   topk_pairs(sel_meta.get('logit_ua_all'), 5),
        "ub_topk":   topk_pairs(sel_meta.get('logit_ub_all'), 5),
        "logit_ua_all": (sel_meta['logit_ua_all'].tolist()
                         if sel_meta.get('logit_ua_all') is not None else None),
        "logit_ub_all": (sel_meta['logit_ub_all'].tolist()
                         if sel_meta.get('logit_ub_all') is not None else None),
        "iou_occ":   m["iou_occ"],
        "iou_free":  m["iou_free"],
        "miou":      m["miou"],
        "sem_miou":  m["sem_miou"],
        "sem_acc":   m["sem_acc"],
        "sem_miou_oracle": m["sem_miou_oracle"],
        "sem_acc_oracle":  m["sem_acc_oracle"],
        "sem_miou_3d":     m.get("sem_miou_3d",   float('nan')),
        "sem_acc_3d_occ":  m.get("sem_acc_3d_occ", float('nan')),
        "mapped_frac": m["mapped_frac"],
        "step_time":           float(step_time),           # plan + bu (no sim, no metric)
        "step_time_total":     float(step_time_total),     # full wall (incl. sim + metric)
        "plan_time":           float(plan_time),
        "sim_time":            float(sim_time),
        "belief_update_time":  float(belief_update_time),
        "metric_time":         float(metric_time),
        "post_push_forced": bool(sel_meta.get('post_push_forced', False)),
    }


def _v2_metric_entry(t, action, sel_meta, m, step_time,
                     push_payload=None, post_view_cam=None,
                     is_push_bundle_duplicate=False,
                     plan_time=0.0, sim_time=0.0,
                     belief_update_time=0.0, sidecar_time=0.0,
                     metric_time=0.0, step_time_total=0.0):
    """Mirror _make_log_entry but for σ_o sidecar metrics (v2). σ_o produces
    only 2D BEV semantic so there's no _3d / _oracle entries."""
    return {
        "step":      int(t),
        "action":    action[0],
        "view":      int(action[1]) if action[0] == 'view' else -1,
        "push_idx":  int(action[1]) if action[0] == 'push' else -1,
        "push":      push_payload,
        "post_view_cam": int(post_view_cam) if post_view_cam is not None else -1,
        "is_push_bundle_duplicate": bool(is_push_bundle_duplicate),
        "ua_arg":    int(sel_meta.get('ua_arg', -1)),
        "ub_arg":    int(sel_meta.get('ub_arg', -1)),
        "iou_occ":   m["iou_occ"],
        "iou_free":  m["iou_free"],
        "miou":      m["miou"],
        "sem_miou":  m["sem_miou"],
        "sem_acc":   m["sem_acc"],
        "mapped_frac": m["mapped_frac"],
        "step_time":           float(step_time),
        "step_time_total":     float(step_time_total),
        "plan_time":           float(plan_time),
        "sim_time":            float(sim_time),
        "belief_update_time":  float(belief_update_time),
        "sidecar_time":        float(sidecar_time),
        "metric_time":         float(metric_time),
        "post_push_forced": bool(sel_meta.get('post_push_forced', False)),
    }


def heuristic_mode(spec: str, t: int, rng) -> str:
    """Mode for step `t` under a heuristic rule, or None to use the learned one.

    'learned'     -> None (paper behaviour: compare G_view vs G_push)
    'periodic:K'  -> 'push' on every K-th step (t = K-1, 2K-1, ...), else 'view'
    'random:P'    -> 'push' with probability P, else 'view'
    """
    spec = (spec or 'learned').strip()
    if spec in ('', 'learned'):
        return None
    kind, _, arg = spec.partition(':')
    if kind == 'periodic':
        k = int(arg)
        if k < 1:
            raise ValueError('periodic:K needs K >= 1')
        return 'push' if ((t + 1) % k == 0) else 'view'
    if kind == 'random':
        return 'push' if rng.random() < float(arg) else 'view'
    if kind == 'minview':
        # Force the mode to view for the first N action steps, then hand back to
        # the learned comparison. Unlike --n_initial_random_views the viewpoint
        # is still chosen by pi_view; only the mode is constrained, which
        # isolates "do not push too early" from "do not use the view policy".
        return 'view' if t < int(arg) else None
    raise ValueError(f'unknown --mode_selector: {spec!r}')


def _nearest_slot_to_contact(spec, positions: dict, sw) -> str:
    """Slot name of the object the push was aimed at.

    The push is parameterised by a contact voxel on the occupancy frontier; the
    intended target is the active object whose centre is closest to that contact
    point in the shelf plane. Used to separate "the object we meant to move
    moved" from "we moved something else".
    """
    if not positions:
        return ''
    cd = sw._cd
    origin = sw.env_origin_np
    cx = float(origin[0]) + cd.SHELF_X_OFFSET + (spec.x_voxel + 0.5) * cd.VOXEL_SIZE
    cy = float(origin[1]) + cd.SHELF_Y_OFFSET + (spec.y_voxel + 0.5) * cd.VOXEL_SIZE
    return min(positions,
               key=lambda n: (positions[n][0] - cx) ** 2 + (positions[n][1] - cy) ** 2)


def snap_to_allowed(cam: int, allowed: np.ndarray) -> int:
    """Nearest allowed viewpoint on the 7x14 grid (row/col Euclidean).

    The post-push view is a heuristic that looks down at the pushed column; when
    C is restricted that exact camera may not exist, so we take the closest one
    that does and keep the heuristic's intent.
    """
    if allowed is None or len(allowed) == 98:
        return int(cam)
    r, c = divmod(int(cam), 14)
    rows, cols = allowed // 14, allowed % 14
    d2 = (rows - r) ** 2 + (cols - c) ** 2
    return int(allowed[int(np.argmin(d2))])


@torch.no_grad()
def run_episode(sw: SimWrapper, push: FrontierPushSpecProvider,
                orch: Orchestrator,
                ep_seed: int, n_steps: int, n_sem: int,
                force_view: bool,
                n_initial_random_views: int = 1,
                action_selector: str = 'reward_logit',
                cnabu_sidecar=None,
                dump_sem: bool = False,
                save_figure_dump: bool = False,
                forced_actions: dict = None,
                allowed_cams: np.ndarray = None,
                mode_selector: str = 'learned',
                push_selector: str = 'policy') -> list:
    """Per-episode rollout.

    Action model:
      - 'view' step: orch.select_action picks (view, cam) or (push, push_idx).
        Consumes 1 action budget slot. Updates UA/UB belief + central.
      - 'push' step: the physical action is a push plus a post-push view in ONE
        sim sequence (apply_push, then capture_view at cam = zi*14 + xi, where
        xi is the cam column closest to spec.x_voxel and zi is drawn from the
        top rows {4, 5, 6}). Then:
          1. UA + UB belief + seen_mask + orch.t are RESET (scene changed).
          2. orch.step is called ONCE with the bundle
             (occ=post_view_occ, swept=actual_swept, sem=post_view_sem).
             This matches training's push-step input (swept and occ together
             in the same timestep).
          3. seen_mask marks the post-view cam.
        For budget fairness (head-to-head with view-only baselines), this
        consumes 2 budget slots — the 2nd slot is a duplicate log entry with
        the same metric (no further sim/model update).

    Training-aligned t=0: UB/UA belief models were trained on rollouts where
    voxel_maps[:, 0] is a randomly sampled pre-push view. We mimic that by
    forcing the first `n_initial_random_views` action slots to be random
    views.
    """
    log = []
    log_v2 = []                     # σ_o sidecar log (only if cnabu_sidecar)
    rng = np.random.default_rng(ep_seed)

    belief_slice_z = 15
    belief_slices_ua = []
    belief_slices_ub = []

    # --- per-step full-3D figure dump (only when save_figure_dump=True) ---
    fig_ua_full:    list = []   # (D=60, H=120, W=80) float16
    fig_ub_full:    list = []
    fig_alpha:      list = []   # (W=120, H=60, D=80) float32
    fig_beta:       list = []
    fig_rgb:        list = []   # (H, W, 3) uint8 (post-view image)
    fig_action:     list = []   # 'view' | 'push'
    fig_view_cam:   list = []   # cam index that produced fig_rgb at this step
    fig_push_idx:   list = []   # -1 for view, else push candidate idx

    # --- per-iteration push-candidate dump (planning side) ---
    fig_iter_t:        list = []   # action_count at start of iter
    fig_iter_swept_d:  list = []   # (N=20, D, H, W) bool
    fig_iter_logit_ub: list = []   # (N=20,) float32
    fig_iter_action:   list = []   # 'view' | 'push'
    fig_iter_chosen:   list = []   # chosen idx (view_cam if view, push_idx if push)

    # --- per-step semantic dump (only when dump_sem=True) ---
    sem_dump_pred_occ: list = []   # list of (W,H,D) bool
    sem_dump_sem_3d:   list = []   # list of (W,H,D) uint8
    sem_dump_step:     list = []   # list of action_count (t)
    sem_dump_gt:       list = []   # list of (W,H,D) uint8 (per-step; scene changes on push)

    last_metrics: Optional[dict] = None
    action_count = 0    # number of action budget slots consumed
    force_view_next = False   # set True after a push so the next step is a view

    if cnabu_sidecar is not None:
        cnabu_sidecar.reset()

    while action_count < n_steps:
        t = action_count
        t_full_start = time.time()

        # ── PLAN: candidate generation + action selection ────────────────
        t_plan_start = time.time()
        prob_whD = _central_prob_whD(orch)
        specs    = push.specs(prob_whD)
        swept_c_np, swept_d_np = _build_candidate_swepts(sw, specs)
        swept_d_t = torch.from_numpy(swept_d_np).to(DEVICE)

        if forced_actions is not None and t in forced_actions:
            f_kind, f_idx = forced_actions[t]
            # still call select_action so sel_meta has the standard fields,
            # then override the chosen action.
            _, sel_meta = orch.select_action(
                swept_d_t, force_view=(f_kind == 'view'),
                selector=action_selector)
            action = (f_kind, int(f_idx))
            sel_meta['forced_action'] = True
            sel_meta['initial_random'] = (t < n_initial_random_views)
            sel_meta['post_push_forced'] = False
            if f_kind == 'view':
                sel_meta['ua_arg'] = int(f_idx)
            else:
                sel_meta['ub_arg'] = int(f_idx)
        elif t < n_initial_random_views:
            cam = (int(rng.integers(0, 98)) if allowed_cams is None
                   else int(allowed_cams[rng.integers(0, len(allowed_cams))]))
            action = ('view', cam)
            _, sel_meta = orch.select_action(swept_d_t, force_view=True,
                                             selector=action_selector)
            sel_meta['initial_random'] = True
            sel_meta['post_push_forced'] = False
            sel_meta['ua_arg'] = cam
        elif force_view_next:
            # Previous step was a push: force a model-chosen view this step
            # so we never get push → push back-to-back. The UA RewardNet's
            # argmax over the 98 cams picks the cam (same as a normal view
            # step with force_view=True).
            action, sel_meta = orch.select_action(
                swept_d_t, force_view=True, selector=action_selector,
            )
            sel_meta['post_push_forced'] = True
            force_view_next = False
        else:
            action, sel_meta = orch.select_action(
                swept_d_t, force_view=force_view, selector=action_selector,
                force_mode=heuristic_mode(mode_selector, t, rng),
            )

        if action[0] == 'push' and push_selector == 'random':
            action = ('push', int(rng.integers(0, len(specs))))
            sel_meta['ub_arg'] = action[1]
            sel_meta['push_random'] = True

        plan_time = time.time() - t_plan_start

        if save_figure_dump:
            fig_iter_t.append(int(t))
            fig_iter_swept_d.append(swept_d_np.astype(bool).copy())
            _lub = sel_meta.get('logit_ub_all')
            fig_iter_logit_ub.append(
                np.asarray(_lub, dtype=np.float32).copy()
                if _lub is not None else np.zeros(len(specs), np.float32))
            fig_iter_action.append(action[0])
            fig_iter_chosen.append(int(action[1]))

        # ── EXECUTE ──────────────────────────────────────────────────────
        if action[0] == 'view':
            # SIM: capture_view + raycast
            t_sim_start = time.time()
            obs = sw.capture_view(action[1])
            occ_c, sem_c, occ_d = view_occ_and_sem(obs, sw.env_origin_np, sw._cd)
            sim_time = time.time() - t_sim_start

            # BELIEF UPDATE: orch.step (model forwards)
            t_bu_start = time.time()
            orch.step(
                dict(
                    occ_view_dive    = occ_d,
                    occ_view_central = occ_c,
                    sem_voxel_central = sem_c,
                    swept_dive       = np.zeros(DIVE_SHAPE,    dtype=np.float32),
                    swept_central    = np.zeros(CENTRAL_SHAPE, dtype=np.float32),
                ),
                action=action, sel_meta=sel_meta,
            )
            belief_slices_ua.append(
                orch.ua_belief.belief[0, 0, belief_slice_z].cpu().numpy().copy())
            belief_slices_ub.append(
                orch.ub_belief.belief[0, 0, belief_slice_z].cpu().numpy().copy())
            if save_figure_dump:
                fig_ua_full.append(
                    orch.ua_belief.belief[0, 0].cpu().numpy().astype(np.float16).copy())
                fig_ub_full.append(
                    orch.ub_belief.belief[0, 0].cpu().numpy().astype(np.float16).copy())
                fig_alpha.append(orch.central.alpha[0].cpu().numpy().copy())
                fig_beta.append(orch.central.beta[0].cpu().numpy().copy())
                fig_rgb.append(np.asarray(obs.rgb, dtype=np.uint8).copy())
                fig_action.append('view')
                fig_view_cam.append(int(action[1]))
                fig_push_idx.append(-1)
            belief_update_time = time.time() - t_bu_start

            # SIDECAR (σ_o) — view step: update_view only
            sidecar_time = 0.0
            m_v2 = None
            if cnabu_sidecar is not None:
                t_sidecar_start = time.time()
                cnabu_sidecar.update_view(obs, sw.env_origin_np)
                sidecar_time = time.time() - t_sidecar_start

            # METRIC (central — primary, + σ_o sidecar if active)
            t_metric_start = time.time()
            gt = sw.get_gt_id_map(verbose=(t == 0)).cpu().numpy()
            m  = central_metrics(orch, gt, n_sem)
            last_metrics = m
            if dump_sem:
                gt_u8 = gt.astype(np.uint8)
                sem_dump_gt.append(gt_u8)
                sem_dump_pred_occ.append(m["_pred_occ"])
                sem_dump_sem_3d.append(
                    m["_sem_3d_pred"] if m["_sem_3d_pred"] is not None
                    else np.zeros_like(gt_u8))
                sem_dump_step.append(int(t))
            if cnabu_sidecar is not None:
                m_v2 = cnabu_sidecar.metrics(gt)
            metric_time = time.time() - t_metric_start

            step_time       = plan_time + belief_update_time + sidecar_time
            step_time_total = time.time() - t_full_start
            log.append(_make_log_entry(
                t, action, sel_meta, m, step_time,
                plan_time=plan_time, sim_time=sim_time,
                belief_update_time=belief_update_time, metric_time=metric_time,
                step_time_total=step_time_total))
            if m_v2 is not None:
                log_v2.append(_v2_metric_entry(
                    t, action, sel_meta, m_v2, step_time,
                    plan_time=plan_time, sim_time=sim_time,
                    belief_update_time=belief_update_time,
                    sidecar_time=sidecar_time,
                    metric_time=metric_time,
                    step_time_total=step_time_total))
            action_count += 1

            forced = sel_meta.get('post_push_forced', False)
            init   = sel_meta.get('initial_random', False)
            kind   = 'init view' if init else 'view'
            print(f't={t:2d}  {kind} cam={action[1]:<3}'.ljust(48) +
                  f'  mIoU={m["miou"]:.3f} sem_mIoU={m["sem_miou"]:.3f} '
                  f'sem_acc={m["sem_acc"]:.3f} mapped={m["mapped_frac"]:.3f}',
                  flush=True)

        else:
            # ── PUSH (bundled push + post-view, training-aligned) ────────
            idx     = int(action[1])
            spec    = specs[idx]

            # SIM: apply_push + post-view capture
            t_sim_start = time.time()
            # Push-risk bookkeeping: only objects newly disturbed BY THIS push
            # count as a failure, so an object already toppled earlier in the
            # episode is not re-counted at every subsequent push.
            _out_before = sw.push_outcome()
            _pos_before = sw.object_positions()
            sw.apply_push(spec)
            _out_after = sw.push_outcome()
            _pos_after = sw.object_positions()
            push_risk = {
                'fell':   sorted(set(_out_after['fallen']) - set(_out_before['fallen'])),
                'tipped': sorted(set(_out_after['tipped']) - set(_out_before['tipped'])),
            }
            push_risk['ok'] = not (push_risk['fell'] or push_risk['tipped'])

            # Execution outcome by what actually moved. A geometrically valid
            # push can still miss, slip, or be blocked, in which case nothing
            # displaces even though no object toppled -- the safety criterion
            # above would call that a success, so it is scored separately.
            _disp = {n: float(np.linalg.norm(np.subtract(_pos_after[n], p)))
                     for n, p in _pos_before.items() if n in _pos_after}
            _target = _nearest_slot_to_contact(spec, _pos_before, sw)
            push_risk['disp_max']    = max(_disp.values()) if _disp else 0.0
            push_risk['disp_target'] = _disp.get(_target, 0.0)
            push_risk['target']      = _target
            push_risk['n_moved_2cm'] = int(sum(d > 0.02 for d in _disp.values()))
            push_risk['n_moved_2cm_other'] = int(sum(
                d > 0.02 for n, d in _disp.items() if n != _target))

            post_cam = snap_to_allowed(
                pick_post_push_cam(spec.x_voxel, rng), allowed_cams)
            obs      = sw.capture_view(post_cam)
            occ_c, sem_c, occ_d = view_occ_and_sem(obs, sw.env_origin_np, sw._cd)
            sim_time = time.time() - t_sim_start

            # BELIEF UPDATE: scene changed → reset UA/UB belief + seen_mask + orch.t.
            # Central state is preserved (carry-forward across the reset).
            # ONE central update with bundled (occ + swept + sem) follows.
            t_bu_start = time.time()
            orch.uab_belief.reset()
            orch.seen_mask.zero_()
            orch.t = 0
            orch.step(
                dict(
                    occ_view_dive    = occ_d,
                    occ_view_central = occ_c,
                    sem_voxel_central = sem_c,
                    swept_dive       = swept_d_np[idx],     # carried for logging only
                    swept_central    = swept_c_np[idx],     # ← real swept fed to central
                ),
                action=action, sel_meta=sel_meta,
            )
            orch.seen_mask[0, post_cam // 14, post_cam % 14] = 1.0
            belief_slices_ua.append(
                orch.ua_belief.belief[0, 0, belief_slice_z].cpu().numpy().copy())
            belief_slices_ub.append(
                orch.ub_belief.belief[0, 0, belief_slice_z].cpu().numpy().copy())
            if save_figure_dump:
                fig_ua_full.append(
                    orch.ua_belief.belief[0, 0].cpu().numpy().astype(np.float16).copy())
                fig_ub_full.append(
                    orch.ub_belief.belief[0, 0].cpu().numpy().astype(np.float16).copy())
                fig_alpha.append(orch.central.alpha[0].cpu().numpy().copy())
                fig_beta.append(orch.central.beta[0].cpu().numpy().copy())
                fig_rgb.append(np.asarray(obs.rgb, dtype=np.uint8).copy())
                fig_action.append('push')
                fig_view_cam.append(int(post_cam))
                fig_push_idx.append(int(idx))
            belief_update_time = time.time() - t_bu_start

            # SIDECAR (σ_o) — push step: σ_m predicts post-push belief, then
            # σ_o ingests the post-push view (paper-verbatim CnabuPaperSidecar).
            sidecar_time = 0.0
            m_v2 = None
            if cnabu_sidecar is not None:
                t_sidecar_start = time.time()
                cnabu_sidecar.update_push(
                    spec, swept_c_np[idx],
                    post_obs=obs, env_origin_np=sw.env_origin_np,
                )
                sidecar_time = time.time() - t_sidecar_start

            # METRIC
            t_metric_start = time.time()
            gt = sw.get_gt_id_map(verbose=(t == 0)).cpu().numpy()
            m  = central_metrics(orch, gt, n_sem)
            last_metrics = m
            if dump_sem:
                gt_u8 = gt.astype(np.uint8)
                sem_dump_gt.append(gt_u8)
                sem_dump_pred_occ.append(m["_pred_occ"])
                sem_dump_sem_3d.append(
                    m["_sem_3d_pred"] if m["_sem_3d_pred"] is not None
                    else np.zeros_like(gt_u8))
                sem_dump_step.append(int(t))
            if cnabu_sidecar is not None:
                m_v2 = cnabu_sidecar.metrics(gt)
            metric_time = time.time() - t_metric_start

            step_time       = plan_time + belief_update_time + sidecar_time
            step_time_total = time.time() - t_full_start

            _push_payload = spec.to_dict()
            _push_payload['risk'] = push_risk

            log.append(_make_log_entry(
                t, action, sel_meta, m, step_time,
                push_payload=_push_payload, post_view_cam=post_cam,
                plan_time=plan_time, sim_time=sim_time,
                belief_update_time=belief_update_time, metric_time=metric_time,
                step_time_total=step_time_total))
            if m_v2 is not None:
                log_v2.append(_v2_metric_entry(
                    t, action, sel_meta, m_v2, step_time,
                    push_payload=_push_payload, post_view_cam=post_cam,
                    plan_time=plan_time, sim_time=sim_time,
                    belief_update_time=belief_update_time,
                    sidecar_time=sidecar_time,
                    metric_time=metric_time,
                    step_time_total=step_time_total))
            action_count += 1
            print(f't={t:2d}  push cand={spec.cand_idx:>2} x={spec.x_voxel:>3} '
                  f'y={spec.y_voxel:>2} ang={spec.angle_deg:+6.1f}  '
                  f'+view cam={post_cam:>2}'.ljust(48) +
                  f'  mIoU={m["miou"]:.3f} sem_mIoU={m["sem_miou"]:.3f} '
                  f'sem_acc={m["sem_acc"]:.3f} mapped={m["mapped_frac"]:.3f}',
                  flush=True)

            # ── 2nd budget slot: duplicate entry, no further update ─────
            # For fairness vs view-only baselines (push+view physical bundle
            # costs 2 budget slots). Same metric, no orch.step / no capture.
            if action_count < n_steps:
                t2 = action_count
                dup_sel_meta = dict(sel_meta)
                dup_sel_meta['post_push_forced'] = True
                dup_sel_meta['initial_random'] = False
                dup_sel_meta['ua_arg'] = post_cam
                # repeat the same belief slice for index alignment
                belief_slices_ua.append(belief_slices_ua[-1].copy())
                belief_slices_ub.append(belief_slices_ub[-1].copy())
                if save_figure_dump and fig_ua_full:
                    fig_ua_full.append(fig_ua_full[-1].copy())
                    fig_ub_full.append(fig_ub_full[-1].copy())
                    fig_alpha.append(fig_alpha[-1].copy())
                    fig_beta.append(fig_beta[-1].copy())
                    fig_rgb.append(fig_rgb[-1].copy())
                    fig_action.append('push_dup')
                    fig_view_cam.append(int(post_cam))
                    fig_push_idx.append(int(idx))
                if dump_sem and sem_dump_pred_occ:
                    sem_dump_pred_occ.append(sem_dump_pred_occ[-1].copy())
                    sem_dump_sem_3d.append(sem_dump_sem_3d[-1].copy())
                    sem_dump_gt.append(sem_dump_gt[-1].copy())
                    sem_dump_step.append(int(t2))
                log.append(_make_log_entry(
                    t2, ('view', post_cam), dup_sel_meta, m, 0.0,
                    post_view_cam=post_cam, is_push_bundle_duplicate=True))
                if m_v2 is not None:
                    log_v2.append(_v2_metric_entry(
                        t2, ('view', post_cam), dup_sel_meta, m_v2, 0.0,
                        post_view_cam=post_cam, is_push_bundle_duplicate=True))
                action_count += 1
                print(f't={t2:2d}  push-bundle dup cam={post_cam:<3}'.ljust(48) +
                      f'  mIoU={m["miou"]:.3f} sem_mIoU={m["sem_miou"]:.3f} '
                      f'sem_acc={m["sem_acc"]:.3f} mapped={m["mapped_frac"]:.3f}',
                      flush=True)

            # After the (push, dup) bundle the next step used to be forced to a
            # view. That is disabled: the policy chooses freely after a push, so
            # the `force_view_next` branch above is currently unreachable.
            # force_view_next = True
            force_view_next = False

    return log, log_v2, {
        'belief_slice_z': belief_slice_z,
        'ua_slices': np.stack(belief_slices_ua) if belief_slices_ua else None,
        'ub_slices': np.stack(belief_slices_ub) if belief_slices_ub else None,
        'sem_dump': ({
            'pred_occ':  np.stack(sem_dump_pred_occ),     # (T, W, H, D) bool
            'sem_3d':    np.stack(sem_dump_sem_3d),       # (T, W, H, D) uint8
            'gt_id':     np.stack(sem_dump_gt),           # (T, W, H, D) uint8
            'steps':     np.asarray(sem_dump_step, np.int32),
        } if (dump_sem and sem_dump_pred_occ) else None),
        'figure_dump': ({
            'ua':        np.stack(fig_ua_full),           # (T, D=60, H=120, W=80) float16
            'ub':        np.stack(fig_ub_full),
            'alpha':     np.stack(fig_alpha),             # (T, W=120, H=60, D=80) float32
            'beta':      np.stack(fig_beta),
            'rgb':       np.stack(fig_rgb),               # (T, H, W, 3) uint8
            'action':    np.asarray(fig_action),
            'view_cam':  np.asarray(fig_view_cam,  np.int32),
            'push_idx':  np.asarray(fig_push_idx,  np.int32),
            'iter_t':         np.asarray(fig_iter_t, np.int32),
            'iter_swept_d':   np.stack(fig_iter_swept_d) if fig_iter_swept_d else np.zeros((0,), bool),
            'iter_logit_ub':  np.stack(fig_iter_logit_ub) if fig_iter_logit_ub else np.zeros((0,), np.float32),
            'iter_action':    np.asarray(fig_iter_action),
            'iter_chosen':    np.asarray(fig_iter_chosen, np.int32),
        } if (save_figure_dump and fig_ua_full) else None),
    }


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    forced_actions_global = None
    if args.force_action_json:
        with open(args.force_action_json) as f:
            spec = json.load(f)
        forced_actions_global = {int(e['step']): (e['kind'], int(e['idx']))
                                 for e in spec}
        print(f"[force_action] loaded {len(forced_actions_global)} forced steps "
              f"from {args.force_action_json}: {forced_actions_global}",
              flush=True)

    # Both branches are identical: the absolute package import works whether
    # this module is run as `python -m inference.run_eval` from the repo root or
    # imported from elsewhere, since _PKG_ROOT was put on sys.path above.
    try:
        from inference.view_subsets import describe as _describe_subset
        from inference.view_subsets import parse_view_subset
    except ImportError:
        from inference.view_subsets import describe as _describe_subset
        from inference.view_subsets import parse_view_subset
    allowed_cams = parse_view_subset(args.view_subset)
    if len(allowed_cams) < 98:
        print(f'[view_subset] {_describe_subset(args.view_subset)}', flush=True)
        allowed_cams = np.asarray(allowed_cams, dtype=int)
    else:
        allowed_cams = None

    push_kw = {} if args.angle_max_deg is None else \
        dict(angle_max_deg=args.angle_max_deg)
    push = FrontierPushSpecProvider(
        scene_mode=args.scene_mode,
        push_dist_m=args.push_dist,
        n_candidates=args.n_push_cands,
        occ_thresh=args.occ_thresh,
        seed=args.seed,
        **push_kw,
    )
    sw   = SimWrapper(num_envs=1, scene_mode=args.scene_mode,
                      push_beyond_m=args.push_beyond_m)
    print(f'[action_space] |C|={98 if allowed_cams is None else len(allowed_cams)}  '
          f'|P|={args.n_push_cands}  push_beyond={sw.push_beyond_m:.3f}m  '
          f'angle_max={push.angle_max_deg:.1f}deg', flush=True)

    # σ_o sidecar (v2: DiVE plan + CNABU belief). Load once, reset per episode.
    cnabu_sidecar = None
    if args.sigma_o and args.sigma_m:
        try:
            from run_mem_pipeline import (  # noqa: E402  (not included in this package)
                load_sigma_o as _load_sigma_o, load_sigma_m as _load_sigma_m,
                CnabuPaperSidecar,
            )
        except ImportError as e:
            raise SystemExit('--sigma_o/--sigma_m sidecar needs run_mem_pipeline.py from the '
                             'original research tree (not included in this release): ' + str(e))
        print(f'[INFO] σ_o sidecar enabled: {args.sigma_o}', flush=True)
        sigma_o_net = _load_sigma_o(args.sigma_o, n_sem=args.n_sem)
        push_unet, push_dp = _load_sigma_m(args.sigma_m, sigma_o_net, n_sem=args.n_sem)
        cnabu_sidecar = CnabuPaperSidecar(sigma_o_net, push_unet, push_dp,
                                           n_sem=args.n_sem)

    summary = {
        "runner":         "dive_central_recursive_pipeline" + ("__view_only" if args.force_view else ""),
        "scene_mode":     args.scene_mode,
        "num_episodes":   args.num_episodes,
        "action_budget":  args.action_budget,
        "base_seed":      args.seed,
        "uab_belief":     str(args.uab_belief),
        "ua_reward":      str(args.ua_reward),
        "ub_reward":      str(args.ub_reward),
        "central_ckpt":   str(args.central_ckpt),
        "force_view":     bool(args.force_view),
        # action-space sensitivity knobs (defaults reproduce the paper runs)
        "mode_selector":  args.mode_selector,
        "push_selector":  args.push_selector,
        "view_subset":    args.view_subset,
        "n_view_cands":   int(len(allowed_cams)) if allowed_cams is not None else 98,
        "n_push_cands":   args.n_push_cands,
        "push_beyond_m":  float(sw.push_beyond_m),
        "angle_max_deg":  float(push.angle_max_deg),
        "ub_reward_alpha": args.ub_reward_alpha,
        "ub_reward_beta":  args.ub_reward_beta,
        "ub_reward_use_attn": not args.ub_reward_no_attn,
        "ub_reward_attn_heads": args.ub_reward_attn_heads,
        "ua_use_seen_mask": args.ua_use_seen_mask,
        "n_sem":          args.n_sem,
        "n_sem_model":    args.n_sem_model,
        "episodes":       [],
    }

    # ── episode list: test-set respawn or sequential seeds ──────────────────
    if args.test_set_dir:
        from inference.sim.test_set_loader import iter_test_set, restore_episode_into_env  # noqa: E402
        ep_list = list(iter_test_set(args.test_set_dir))
        end = args.episode_end if args.episode_end >= 0 else len(ep_list)
        ep_list = ep_list[args.episode_start:end]
        if args.num_episodes > 0 and args.num_episodes < len(ep_list):
            ep_list = ep_list[:args.num_episodes]
        print(f'[INFO] test_set_dir={args.test_set_dir}  '
              f'shard=[{args.episode_start}:{end}]  episodes={len(ep_list)}', flush=True)
        summary.update({
            "test_set_dir":  args.test_set_dir,
            "episode_start": args.episode_start,
            "episode_end":   end,
            "num_episodes":  len(ep_list),
        })
    else:
        ep_list = [(ep, None) for ep in range(args.num_episodes)]

    # ── per-episode loop ────────────────────────────────────────────────────
    for ep, ep_dir in ep_list:
        ep_seed = args.seed + ep
        print(f'\n[dive_central] === episode {ep} (seed={ep_seed}'
              + (f'  test_set={Path(ep_dir).name}' if ep_dir else '') + ') ===',
              flush=True)

        sw.reset(seed=ep_seed)
        if ep_dir is not None:
            restore_episode_into_env(sw.scene, env_idx=0, ep_dir=ep_dir,
                                     cd_module=sw._cd)
            robot = sw.scene["robot"]
            for _ in range(50):
                robot.set_joint_position_target(
                    sw.capture_jpos, joint_ids=sw.arm_joint_ids)
                sw.scene.write_data_to_sim()
                sw.sim.step()
                sw.scene.update(sw.dt)

        # Fresh orchestrator per episode → UA/UB beliefs reset to 0, central α=β=1.
        orch = Orchestrator(
            uab_belief_ckpt=str(args.uab_belief),
            ua_reward_ckpt=str(args.ua_reward),
            ub_reward_ckpt=str(args.ub_reward),
            central_ckpt  =str(args.central_ckpt),
            with_sweep_central=True, with_semantic_central=True,
            n_sem_central=args.n_sem_model,
            n_steps=args.action_budget,
            device=DEVICE,
            ua_use_seen_mask=args.ua_use_seen_mask,
            ub_reward_alpha=args.ub_reward_alpha,
            ub_reward_beta =args.ub_reward_beta,
            ub_reward_use_attn=not args.ub_reward_no_attn,
            ub_reward_attn_heads=args.ub_reward_attn_heads,
        )
        orch.set_allowed_cams(allowed_cams)

        if args.warmup_sleep > 0:
            print(f'[warmup] sleeping {args.warmup_sleep:.1f}s before inference '
                  f'(rendering sim so window updates)...', flush=True)
            t_warm_end = time.time() + args.warmup_sleep
            while time.time() < t_warm_end:
                sw.sim.render()

        t0 = time.time()
        log, log_v2, belief_dump = run_episode(
            sw, push, orch, ep_seed, args.action_budget,
            args.n_sem, args.force_view,
            n_initial_random_views=args.n_initial_random_views,
            action_selector=args.action_selector,
            cnabu_sidecar=cnabu_sidecar,
            dump_sem=args.dump_sem,
            save_figure_dump=args.save_figure_dump,
            forced_actions=forced_actions_global,
            allowed_cams=allowed_cams,
            mode_selector=args.mode_selector,
            push_selector=args.push_selector,
        )
        wall_time = time.time() - t0

        if belief_dump.get('ua_slices') is not None:
            np.savez(out_dir / f'belief_z{belief_dump["belief_slice_z"]:02d}_ep{ep:03d}.npz',
                     ua=belief_dump['ua_slices'].astype(np.float16),
                     ub=belief_dump['ub_slices'].astype(np.float16))

        if belief_dump.get('sem_dump') is not None:
            sd = belief_dump['sem_dump']
            np.savez_compressed(
                out_dir / f'sem_dump_ep{ep:03d}.npz',
                pred_occ=sd['pred_occ'],
                sem_3d  =sd['sem_3d'],
                gt_id   =sd['gt_id'],
                steps   =sd['steps'],
            )

        if belief_dump.get('figure_dump') is not None:
            fd = belief_dump['figure_dump']
            np.savez_compressed(
                out_dir / f'figure_dump_ep{ep:03d}.npz',
                ua=fd['ua'], ub=fd['ub'],
                alpha=fd['alpha'], beta=fd['beta'],
                rgb=fd['rgb'],
                action=fd['action'],
                view_cam=fd['view_cam'],
                push_idx=fd['push_idx'],
                iter_t=fd['iter_t'],
                iter_swept_d=fd['iter_swept_d'],
                iter_logit_ub=fd['iter_logit_ub'],
                iter_action=fd['iter_action'],
                iter_chosen=fd['iter_chosen'],
            )

        ep_summary = {
            "episode":            ep,
            "seed":               ep_seed,
            "steps":              len(log),
            "push_count":         sum(1 for e in log if e["action"] == "push"),
            "final_miou":         log[-1]["miou"]        if log else 0.0,
            "final_iou_occ":      log[-1]["iou_occ"]     if log else 0.0,
            "final_sem_miou":     log[-1]["sem_miou"]    if log else 0.0,
            "final_sem_acc":      log[-1]["sem_acc"]     if log else 0.0,
            "final_mapped_frac":  log[-1]["mapped_frac"] if log else 0.0,
            "wall_time":          wall_time,
        }
        summary["episodes"].append(ep_summary)

        with open(out_dir / f'log_ep{ep:03d}.json', 'w') as f:
            json.dump(log, f, indent=2)
        if log_v2:
            with open(out_dir / f'log_v2_ep{ep:03d}.json', 'w') as f:
                json.dump(log_v2, f, indent=2)
        with open(out_dir / f'summary{args.summary_suffix}.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print(f'[EP {ep:3d}] steps={len(log):2d}  push={ep_summary["push_count"]:2d}  '
              f'mIoU={ep_summary["final_miou"]:.3f}  '
              f'sem_mIoU={ep_summary["final_sem_miou"]:.3f}  '
              f'sem_acc={ep_summary["final_sem_acc"]:.3f}  '
              f't={wall_time:.1f}s', flush=True)

    if summary["episodes"]:
        eps = summary["episodes"]
        for key in ("final_miou", "final_iou_occ", "final_sem_miou",
                    "final_sem_acc", "final_mapped_frac",
                    "push_count", "wall_time"):
            vals = [e[key] for e in eps]
            summary[f"mean_{key}"] = float(np.mean(vals))
            if len(vals) > 1:
                summary[f"std_{key}"] = float(np.std(vals))
        with open(out_dir / f'summary{args.summary_suffix}.json', 'w') as f:
            json.dump(summary, f, indent=2)
        print(f'\n[DONE] mean mIoU={summary["mean_final_miou"]:.4f}  '
              f'sem_mIoU={summary["mean_final_sem_miou"]:.4f}  '
              f'sem_acc={summary["mean_final_sem_acc"]:.4f}  '
              f'avg_push={summary["mean_push_count"]:.1f}', flush=True)

    if args.keep_open:
        try:
            from inference.sim_wrapper import _simulation_app
            while _simulation_app is not None and _simulation_app.is_running():
                sw.sim.render()
        except KeyboardInterrupt:
            pass

    # NOTE: do NOT call sw.close() — Isaac Lab cleanup deadlocks here.
    # Process-level _exit(0) skips Python finalizers; the OS reclaims the sim
    # resources cleanly.
    import os, sys
    print('[dive_central_recursive] done. _exit(0).', flush=True)
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
