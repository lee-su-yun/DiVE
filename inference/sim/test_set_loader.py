"""
Respawn a saved test-set episode into an InteractiveScene env.

A test-set episode is a directory produced by
`python -m inference.sim.scene_low_occlusion --test_set_mode`.

Per-episode `obj_poses.npz` stores active slots' root_state_w in
env-origin-relative coordinates, so they transfer to any env_idx:

  state_w[env_idx, :3] = scene.env_origins[env_idx, :3] + rel[:3]
  state_w[env_idx, 3:] = rel[3:]            # quat + lin/ang vel

Typical usage in an inference script::

    from inference.sim.test_set_loader import load_episode_meta, restore_episode_into_env

    meta = load_episode_meta("data/test_sets/low_occlusion_v1/000000000")
    print("hidden slots:", meta["hidden_slots"])

    # after `scene = InteractiveScene(...)`, `sim.reset()`, gripper attach etc.
    active_slot_names = restore_episode_into_env(
        scene, env_idx=0,
        ep_dir="data/test_sets/low_occlusion_v1/000000000",
    )
    # then stabilize(50) to settle, run inference normally.

The function PARKS all non-active slots at their default off-shelf
`slot_init_pos`, matching the convention the scene module uses.
"""
from __future__ import annotations

import os
import numpy as np
import torch

from inference.sim.ycb_types import SLOT_NAMES, SLOT_TO_TYPE_ID, slot_init_pos


def load_episode_meta(ep_dir: str) -> dict:
    """Light read: return dict with active_slot_names, type_ids, hidden_slots."""
    d = np.load(os.path.join(ep_dir, "obj_poses.npz"), allow_pickle=True)
    return {
        "slot_idxs":       d["slot_idxs"].tolist(),
        "type_ids":        d["type_ids"].tolist(),
        "slot_names":      [str(n) for n in d["slot_names"]],
        "hidden_slots":    d["hidden_slots"].tolist() if "hidden_slots" in d.files else [],
        "root_states_rel": d["root_states_rel"],
    }


def restore_episode_into_env(scene, env_idx: int, ep_dir: str,
                             cd_module=None) -> list[str]:
    """
    Re-spawn the test-set episode into `env_idx` of an active InteractiveScene.
    Returns the list of active slot names for this episode (in the same order
    they appeared in obj_poses.npz).

    If `cd_module` (the imported inference.sim.scene_* module) is provided, also
    overwrites its `_active_slots[env_idx]` so `get_gt_id_map(scene, env_idx)`
    and `get_active_slots(env_idx)` iterate the **test-set's** active slots —
    NOT whatever `randomize_object_positions` left behind during `sw.reset()`.
    Without this patch, GT voxelization mixes random-active slots (now parked
    off-shelf by pass 1 below) with test-set-active slots that aren't in the
    list, producing the symptom: GT shows fewer objects than σ_o detects.

    Caller is responsible for sim.reset(), gripper attachment, friction setup,
    and post-spawn `stabilize(...)` settling.
    """
    meta = load_episode_meta(ep_dir)
    active_names = meta["slot_names"]
    active_set   = set(active_names)
    states_rel   = meta["root_states_rel"]   # (N_active, 13) float32

    env_orig = scene.env_origins[env_idx].detach().cpu().numpy().astype(np.float32)

    # 1) Park ALL slots at off-shelf default positions (consistent with
    #    randomize_object_positions' handling of inactive slots).
    for name in SLOT_NAMES:
        if name not in scene._rigid_objects:
            continue
        obj = scene[name]
        s = obj.data.root_state_w.clone()
        if name in active_set:
            continue   # will be overwritten in pass 2
        ix, iy, iz = slot_init_pos(SLOT_NAMES.index(name))
        s[env_idx, 0] = env_orig[0] + ix
        s[env_idx, 1] = env_orig[1] + iy
        s[env_idx, 2] = env_orig[2] + iz
        sw, sx, sy, sz = obj.cfg.init_state.rot
        s[env_idx, 3] = sw
        s[env_idx, 4] = sx
        s[env_idx, 5] = sy
        s[env_idx, 6] = sz
        s[env_idx, 7:] = 0.0
        obj.write_root_state_to_sim(s)

    # 2) Place active slots from saved relative state.
    for i, name in enumerate(active_names):
        if name not in scene._rigid_objects:
            print(f"[test_set_loader] WARN: active slot {name} not in scene._rigid_objects, skipping")
            continue
        rel = states_rel[i]
        obj = scene[name]
        s = obj.data.root_state_w.clone()
        s[env_idx, 0] = env_orig[0] + float(rel[0])
        s[env_idx, 1] = env_orig[1] + float(rel[1])
        s[env_idx, 2] = env_orig[2] + float(rel[2])
        s[env_idx, 3] = float(rel[3])
        s[env_idx, 4] = float(rel[4])
        s[env_idx, 5] = float(rel[5])
        s[env_idx, 6] = float(rel[6])
        s[env_idx, 7:] = 0.0   # zero velocities on respawn
        obj.write_root_state_to_sim(s)

    # 3) Sync the scene module's `_active_slots` global so downstream
    #    helpers (get_gt_id_map, get_active_slots) iterate the test-set's
    #    active slots, not the random ones from sw.reset.
    if cd_module is not None and hasattr(cd_module, '_active_slots'):
        while len(cd_module._active_slots) <= env_idx:
            cd_module._active_slots.append(None)
        cd_module._active_slots[env_idx] = list(active_names)

    return active_names


def iter_test_set(test_set_dir: str):
    """Yield (episode_id, ep_dir) tuples for every episode in manifest.json order."""
    import json
    manifest = json.load(open(os.path.join(test_set_dir, "manifest.json")))
    for meta in manifest["episodes"]:
        ep_dir = os.path.join(test_set_dir, meta["folder"])
        yield meta["episode_id"], ep_dir
