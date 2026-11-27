"""FrontierPushSpecProvider — freeform push candidate generation.

Mirrors inference/sim/scene_low_occlusion.py:compute_frontier_pushes, but at
inference time it works from a *belief* map instead of the GT. Frontier =
shelf-entrance-adjacent occupied voxels in the BEV slab (front
FRONTIER_BASE_CM cm). Each candidate is freeform: (x_voxel, y_voxel,
angle_deg ∈ ±ANGLE_MAX_DEG).

A FreeformPushSpec also carries the swept_map (W, H, D) bool used as σ_m /
DiVE-UB input — generated geometrically from the push column at y=y_voxel.

Usage:
    provider = FrontierPushSpecProvider(
        scene_mode='high', push_dist_m=0.25, n_candidates=20, ...,
    )
    specs       = provider.specs(prob_whD)                # list[FreeformPushSpec]
    swept_maps  = provider.swept_maps_central(specs)      # (N, W, H, D) bool
    swept_dive  = provider.swept_maps_dive(specs)         # (N, D, H, W) bool

prob_whD: cupy or numpy (W=120, H=60, D=80) float — σ_o occupancy prob map
            (e.g. from InfoGainEvalBelief.occupancy_prob_from_belief).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from inference.sim.voxel_map import (
    VOXEL_W, VOXEL_H, VOXEL_D, VOXEL_SIZE,
)


# ── Frontier sampling hyperparams (mirror the data-collection scene) ────────
SUBSCENE_LEN       = 20             # default n_candidates
FRONTIER_BASE_CM   = 15.0
FRONTIER_EXPAND_CM = 5.0
FRONTIER_MAX_CM    = 35.0
ANGLE_MAX_DEG      = 30.0
DEFAULT_PUSH_DIST_M = 0.25          # rod travel length (m)


@dataclass
class FreeformPushSpec:
    """One freeform push: column (x_voxel) + frontier depth (y_voxel) + angle."""
    cand_idx:    int                # index in the candidate list (0..n-1)
    x_voxel:     int
    y_voxel:     int                # frontier y in BEV (= voxel-z in central frame)
    angle_deg:   float
    push_dist_m: float = DEFAULT_PUSH_DIST_M

    # Convenience for the log-entry dict (matches the per-step log schema)
    def to_dict(self) -> dict:
        depth_voxels = int(round(self.push_dist_m / VOXEL_SIZE))
        end_y_vox = min(VOXEL_D - 1, self.y_voxel + depth_voxels - 1)
        return {
            "cand_idx":     int(self.cand_idx),
            "x_voxel":      int(self.x_voxel),
            "y_voxel":      int(self.y_voxel),
            "angle_deg":    float(self.angle_deg),
            "push_dist_m":  float(self.push_dist_m),
            "depth_voxels": int(depth_voxels),
            "start_voxel":  [int(self.x_voxel), 0, int(self.y_voxel)],
            "end_voxel":    [int(self.x_voxel), 0, int(end_y_vox)],
        }


# ── Frontier sampling ────────────────────────────────────────────────────────

def compute_frontier_pushes_from_prob(prob_whD: np.ndarray,
                                       n: int = SUBSCENE_LEN,
                                       occ_thresh: float = 0.5,
                                       rng: Optional[random.Random] = None,
                                       angle_max_deg: float = ANGLE_MAX_DEG,
                                       ) -> list[tuple[int, int, float]]:
    """Belief-driven frontier sampling.

    prob_whD : (W, H, D) float in [0, 1] — σ_o occupancy probability map.
    Returns list of (x_voxel, y_voxel, angle_deg), length n.

    Algorithm (identical to the data-collection scene's compute_frontier_pushes
    except that it thresholds a probability map instead of reading the GT):
      1. binarize prob_whD > occ_thresh → bool (W, H, D)
      2. frontier slab: z (depth) ∈ [0, depth_vox), where
         depth_vox = depth_cm * 0.01 / VOXEL_SIZE, starts at FRONTIER_BASE_CM
         and expands by FRONTIER_EXPAND_CM until enough columns found
      3. candidate x = x voxel that has any occupied voxel in the slab
      4. for each sampled x: pick the first-occupied y voxel (closest to entrance)
      5. angle ∈ ±θ_max  uniform; θ_max = min(d_cm, ANGLE_MAX_DEG)
    """
    r = rng or random
    occ = (prob_whD > occ_thresh).astype(np.bool_)            # (W, H, D)

    depth_cm = FRONTIER_BASE_CM
    candidate_xs: list = []
    while True:
        depth_vox = min(int(round(depth_cm * 0.01 / VOXEL_SIZE)), VOXEL_D)
        slab = occ[:, :, :depth_vox]                          # (W, H, depth_vox)
        x_has_obj = slab.any(axis=2).any(axis=1)              # (W,)
        candidate_xs = np.flatnonzero(x_has_obj).tolist()
        if len(candidate_xs) >= n or depth_cm >= FRONTIER_MAX_CM:
            break
        depth_cm += FRONTIER_EXPAND_CM

    # sample
    if len(candidate_xs) >= n:
        sampled_xs = r.sample(candidate_xs, n)
    elif candidate_xs:
        sampled_xs = list(candidate_xs)
        while len(sampled_xs) < n:
            sampled_xs.append(r.choice(candidate_xs))
    else:
        sampled_xs = [r.randint(0, VOXEL_W - 1) for _ in range(n)]

    pushes: list[tuple[int, int, float]] = []
    for x in sampled_xs:
        col = occ[x, :, :]                                    # (H, D)
        y_has_obj = col.any(axis=0)                           # (D,)
        ys = np.flatnonzero(y_has_obj)
        y_voxel = int(ys[0]) if ys.size > 0 else int(VOXEL_D * 0.4)

        x_cm = x * VOXEL_SIZE * 100.0
        d_cm = min(x_cm, (VOXEL_W - 1 - x) * VOXEL_SIZE * 100.0)
        theta_max = min(d_cm, angle_max_deg)
        angle_deg = r.uniform(-theta_max, theta_max)
        pushes.append((int(x), y_voxel, float(angle_deg)))

    return pushes


# ── Swept-map generation (geometric approximation) ──────────────────────────

def _swept_map_for(spec: FreeformPushSpec) -> np.ndarray:
    """Geometric swept volume for one freeform push:
      column at x_voxel, all H, y ∈ [y_voxel, y_voxel + depth_voxels), rotated by angle_deg.
    Returns (W, H, D) bool. Angle approximation: rotate the y-column around
    the column's center axis in the BEV plane. Simple piecewise-linear sample.
    """
    depth_voxels = int(round(spec.push_dist_m / VOXEL_SIZE))
    swept = np.zeros((VOXEL_W, VOXEL_H, VOXEL_D), dtype=bool)

    # angle around the push start: rotate (dx, dy) in BEV
    theta = np.deg2rad(spec.angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    # sample along the push direction (y forward in voxel-z) every voxel
    for step in range(depth_voxels):
        dy = float(step)
        # rotation about (x_voxel, y_voxel)
        rx = -sin_t * dy           # negative because rotating push direction
        ry =  cos_t * dy
        nx = int(round(spec.x_voxel + rx))
        ny = int(round(spec.y_voxel + ry))
        if 0 <= nx < VOXEL_W and 0 <= ny < VOXEL_D:
            swept[nx, :, ny] = True
    return swept


# ── Provider class ───────────────────────────────────────────────────────────

class FrontierPushSpecProvider:
    """Frontier-sampled freeform push provider for inference."""

    def __init__(self,
                 scene_mode: str = 'high',
                 push_dist_m: float = DEFAULT_PUSH_DIST_M,
                 n_candidates: int = SUBSCENE_LEN,
                 occ_thresh: float = 0.5,
                 seed: Optional[int] = None,
                 angle_max_deg: float = ANGLE_MAX_DEG):
        if scene_mode not in ('high', 'low', 'super_high'):
            raise ValueError(scene_mode)
        # super_high uses same physical scene as high (just stricter placement)
        self.scene_mode  = 'high' if scene_mode == 'super_high' else scene_mode
        self.push_dist_m = push_dist_m
        self.n_candidates = n_candidates
        self.occ_thresh  = occ_thresh
        self.angle_max_deg = angle_max_deg
        self.rng = random.Random(seed) if seed is not None else random

    def n(self) -> int:
        return self.n_candidates

    def specs(self, prob_whD) -> list[FreeformPushSpec]:
        """Return n_candidates freeform push specs sampled from the belief map."""
        if isinstance(prob_whD, torch.Tensor):
            prob_whD = prob_whD.detach().cpu().numpy()
        try:
            import cupy as cp
            if isinstance(prob_whD, cp.ndarray):
                prob_whD = cp.asnumpy(prob_whD)
        except ImportError:
            pass

        triples = compute_frontier_pushes_from_prob(
            prob_whD, n=self.n_candidates,
            occ_thresh=self.occ_thresh, rng=self.rng,
            angle_max_deg=self.angle_max_deg,
        )
        return [FreeformPushSpec(cand_idx=i, x_voxel=x, y_voxel=y,
                                 angle_deg=a, push_dist_m=self.push_dist_m)
                for i, (x, y, a) in enumerate(triples)]

    def swept_maps_central(self, specs: list[FreeformPushSpec]) -> np.ndarray:
        """(N, W=120, H=60, D=80) bool."""
        out = np.zeros((len(specs), VOXEL_W, VOXEL_H, VOXEL_D), dtype=bool)
        for i, s in enumerate(specs):
            out[i] = _swept_map_for(s)
        return out

    def swept_maps_dive(self, specs: list[FreeformPushSpec]) -> np.ndarray:
        """(N, D=60, H=120, W=80) bool — DiVE axis convention.
        central (N, W=120, H=60, D=80)  →  dive (N, 60, 120, 80)
        (swap the W and H axes; depth stays last). In the DiVE convention that
        is D=60 (height), H=120 (shelf x), W=80 (shelf depth), matching
        inference.orchestrator_base.DIVE_SHAPE.
        """
        central = self.swept_maps_central(specs)              # (N, W, H, D)
        return central.transpose(0, 2, 1, 3)                  # → (N, H, W, D)
