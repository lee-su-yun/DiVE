"""Pre-generate (current_map, gains[29]) dataset for U_AB_beliefmap UB Reward Model.

Uses the UB model trained by train_UAB.py --task UB --init_belief 0.0.

For each (scene, push_N) with valid pre_occ/post_XX directories, samples M random
configurations:

  1) k ~ U[1, max_steps]
  2) k prior cams (no replacement) from 98 cams of push_N/pre_occ
  3) Roll out pretrained UB model with swept=0 over those k cams
     (belief init = init_belief, default 0.0 to match UB training)
     -> belief_prev  (current_map)
  4) For each candidate push XX in [0, 28]:
        post_view = sample from _group_view_pool(_col_idx_to_group(XX))
        voxel    = push_N/post_{XX:02d}/post_occ/{post_view:03d}.npz
        swept    = push_N/post_{XX:02d}/action.npz['swept_map'] (transpose(1,0,2))
        belief_post_XX = sigmoid(UB(belief_prev, voxel, swept))
        gain_XX = Σ |belief_post_XX - belief_prev|        (L1)

Filtering (fallen_log_revised.txt × episode_log.txt):
  Same as UA generator's _scene_bad_pushes — push_2 / push_3 are excluded if
  upstream of a fallen event.

Saves one .npz per sample at:
    {out_root}/{scene_root_key}/{scene_id}/{push_name}/{sample_idx:03d}.npz
keys:
    current_map : (1, 60, 120, 80) float16
    gains       : (29,)            float32
    k           : int32
    prior_cams  : (k,)             int32
    post_views  : (29,)            int32

명령어 :
    cd /home/sylee/codes/DiVE/3_reward_dataset
    python UB_reward_generate.py \
        --data_roots /data/APOBU/beliefmap_low_occlusion_0423 \
                     /data/APOBU/beliefmap_high_occlusion_0423 \
        --ub_checkpoint /result/APOBU/U_AB_beliefmap/UB_v2_all/best.pth \
        --out_root /data/APOBU/U_AB_beliefmap/ub_reward_dataset \
        --init_belief 0.0 \
        --device 0,1,2,3 --samples_per_scene 64
"""
import os
import sys
import argparse
from collections import defaultdict
import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "support_code"))
from model_UAB import UNet3DPush  # noqa: E402
from dataset_UAB import (  # noqa: E402
    parse_fallen_log, parse_episode_log,
    _col_idx_to_group, _group_view_pool,
)

NUM_CAMERAS = 98
NUM_PUSHES = 29
D, H, W = 60, 120, 80


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_roots', type=str, nargs='+', default=[
        '/data/APOBU/beliefmap_low_occlusion_0423',
        '/data/APOBU/beliefmap_high_occlusion_0423',
    ])
    p.add_argument('--out_root', type=str,
                   default='/data/APOBU/U_AB_beliefmap/ub_reward_dataset')
    p.add_argument('--ub_checkpoint', type=str,
                   default='/result/APOBU/U_AB_beliefmap/UB_v2_all/best.pth')
    p.add_argument('--init_belief', type=float, default=0.0,
                   help='Initial belief value used during rollout. '
                        'Match train_UB._init_belief used for UB training (0.0).')
    p.add_argument('--samples_per_scene', type=int, default=64,
                   help='M samples per (scene, push_N).')
    p.add_argument('--max_steps', type=int, default=8,
                   help='k ~ U[1, max_steps].')
    p.add_argument('--device', type=str, default='0',
                   help='GPU id, or comma-separated list for multi-GPU sharding.')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--amp', action='store_true', default=True,
                   help='Use fp16 autocast (default on).')
    p.add_argument('--no_amp', dest='amp', action='store_false')
    p.add_argument('--scene_start', type=int, default=0)
    p.add_argument('--scene_end', type=int, default=-1)
    p.add_argument('--overwrite', action='store_true')
    return p.parse_args()


def load_ub_model(ckpt_path, device):
    model = UNet3DPush().to(device)
    model.eval()
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f'[UB] loaded {ckpt_path}  '
          f'epoch={ckpt.get("epoch", "?")}  '
          f'val_loss={ckpt.get("val_loss", float("nan")):.4f}')
    return model


def _scene_bad_pushes(scene_dir, scene_name, fallen):
    """Same as UA_reward_generate._scene_bad_pushes."""
    bad = set()
    log_path = os.path.join(scene_dir, 'episode_log.txt')
    if not os.path.isfile(log_path):
        return bad
    transitions = parse_episode_log(log_path)
    by_post = {t['post_push']: t for t in transitions}
    for post_push in [2, 3]:
        t = by_post.get(post_push)
        if t is None:
            continue
        if t['pre_push'] in bad:
            bad.add(post_push)
            continue
        if (scene_name, t['pre_push'], t['col_idx']) in fallen:
            bad.add(post_push)
    return bad


def list_push_dirs(data_roots):
    """Return list of (scene_root_key, scene_name, push_name, push_dir).

    Requires:
      - push_dir/pre_occ/{000..097}.npz all exist
      - push_dir/post_{XX:02d}/{post_occ/, action.npz} for all XX in [0, 29)
    Filters via _scene_bad_pushes (fallen propagation).
    """
    results = []
    for data_root in data_roots:
        root_key = os.path.basename(data_root.rstrip('/'))
        fallen = parse_fallen_log(os.path.join(data_root, 'fallen_log_revised.txt'))
        scene_names = sorted(d for d in os.listdir(data_root)
                             if os.path.isdir(os.path.join(data_root, d)))
        for scene in scene_names:
            scene_dir = os.path.join(data_root, scene)
            bad_pushes = _scene_bad_pushes(scene_dir, scene, fallen)
            for push_name in sorted(os.listdir(scene_dir)):
                if not push_name.startswith('push_'):
                    continue
                try:
                    push_id = int(push_name.split('_')[1])
                except (ValueError, IndexError):
                    continue
                if push_id in bad_pushes:
                    continue
                push_dir = os.path.join(scene_dir, push_name)
                pre_occ_dir = os.path.join(push_dir, 'pre_occ')
                if not os.path.isdir(pre_occ_dir):
                    continue
                if not all(os.path.isfile(os.path.join(pre_occ_dir, f'{v:03d}.npz'))
                           for v in range(NUM_CAMERAS)):
                    continue
                ok = True
                for XX in range(NUM_PUSHES):
                    post_dir = os.path.join(push_dir, f'post_{XX:02d}')
                    if not (os.path.isdir(os.path.join(post_dir, 'post_occ'))
                            and os.path.isfile(os.path.join(post_dir, 'action.npz'))):
                        ok = False
                        break
                if not ok:
                    continue
                results.append((root_key, scene, push_name, push_dir))
    return results


def load_pre_voxels(push_dir):
    """(98, D, H, W) float32 from push_dir/pre_occ/."""
    pre_occ_dir = os.path.join(push_dir, 'pre_occ')
    arrs = [np.load(os.path.join(pre_occ_dir, f'{v:03d}.npz'))['occ']
            for v in range(NUM_CAMERAS)]
    return np.stack(arrs).astype(np.float32)


def load_swept_maps(push_dir):
    """(29, D, H, W) float32 from push_dir/post_XX/action.npz."""
    arrs = []
    for XX in range(NUM_PUSHES):
        action_path = os.path.join(push_dir, f'post_{XX:02d}', 'action.npz')
        s = np.load(action_path)['swept_map'].transpose(1, 0, 2).astype(np.float32)
        arrs.append(s)
    return np.stack(arrs)


def load_post_voxel(push_dir, XX, post_view):
    path = os.path.join(push_dir, f'post_{XX:02d}', 'post_occ', f'{post_view:03d}.npz')
    return np.load(path)['occ'].astype(np.float32)


def _maybe_autocast(use_amp):
    if use_amp:
        return torch.amp.autocast('cuda', dtype=torch.float16)
    import contextlib
    return contextlib.nullcontext()


@torch.no_grad()
def rollout_prior_batched(ub_model, voxels_pre_t, priors_mat, device,
                          init_belief, use_amp):
    """Roll out k pre-cams (swept=0) for B samples sharing same k.

    priors_mat: (B, k) long
    returns:    (B, 1, D, H, W) belief_prev
    """
    B, k = priors_mat.shape
    belief = torch.full((B, 1, D, H, W), init_belief, device=device)
    swept = torch.zeros((B, 1, D, H, W), device=device)
    for t in range(k):
        idx = priors_mat[:, t]
        voxel = voxels_pre_t.index_select(0, idx).unsqueeze(1)
        x = torch.cat([belief, voxel, swept], dim=1)
        with _maybe_autocast(use_amp):
            out = ub_model(x)
        belief = torch.sigmoid(out.float())
    return belief


@torch.no_grad()
def compute_gains_batched(ub_model, belief_prev, post_voxels_t, swept_t,
                          device, use_amp):
    """Per-sample 29 push step → L1 gain.

    belief_prev    : (1, 1, D, H, W)
    post_voxels_t  : (29, D, H, W)   chosen post_view voxels for each XX
    swept_t        : (29, D, H, W)
    returns        : (29,) gains
    """
    bel_exp = belief_prev.expand(NUM_PUSHES, 1, D, H, W)
    voxel = post_voxels_t.unsqueeze(1)
    swept = swept_t.unsqueeze(1)
    x = torch.cat([bel_exp, voxel, swept], dim=1)
    with _maybe_autocast(use_amp):
        out = ub_model(x)
    belief_post = torch.sigmoid(out.float())   # (29, 1, D, H, W)
    delta = belief_post - bel_exp
    gain = delta.abs().sum(dim=(1, 2, 3, 4))   # L1
    return gain


@torch.no_grad()
def process_entry(ub_model, voxels_pre_t, swept_t, push_dir, rng, args, device):
    """Generate M samples for one (scene, push_N).

    Uses post_view cache: per (XX, post_view) load voxel once.
    """
    M = args.samples_per_scene

    # k & prior cams
    ks = rng.randint(1, args.max_steps + 1, size=M).astype(np.int32)
    priors_list = [rng.choice(NUM_CAMERAS, int(k), replace=False) for k in ks]

    # post_view sampling: per sample, per XX
    pools = [_group_view_pool(_col_idx_to_group(XX)) for XX in range(NUM_PUSHES)]
    post_views_all = np.empty((M, NUM_PUSHES), dtype=np.int32)
    for m in range(M):
        for XX in range(NUM_PUSHES):
            post_views_all[m, XX] = int(rng.choice(pools[XX]))

    # Cache (XX, post_view) -> voxel
    voxel_cache = {}
    for XX in range(NUM_PUSHES):
        unique_views = np.unique(post_views_all[:, XX])
        for v in unique_views:
            voxel_cache[(XX, int(v))] = load_post_voxel(push_dir, XX, int(v))

    # Roll out belief_prev grouped by k
    beliefs = torch.empty((M, 1, D, H, W), device=device)
    by_k = defaultdict(list)
    for m, k in enumerate(ks):
        by_k[int(k)].append(m)
    for k, members in by_k.items():
        priors_mat = torch.tensor(
            np.stack([priors_list[m] for m in members]),
            dtype=torch.long, device=device)
        belief_group = rollout_prior_batched(
            ub_model, voxels_pre_t, priors_mat, device,
            args.init_belief, args.amp)
        for i, m in enumerate(members):
            beliefs[m] = belief_group[i]

    # Compute gains per sample (29 candidates each)
    gains_all = np.empty((M, NUM_PUSHES), dtype=np.float32)
    for m in range(M):
        post_voxels_np = np.stack([
            voxel_cache[(XX, int(post_views_all[m, XX]))]
            for XX in range(NUM_PUSHES)
        ])
        post_voxels_t = torch.from_numpy(post_voxels_np).to(device)
        gain = compute_gains_batched(
            ub_model, beliefs[m:m+1], post_voxels_t, swept_t,
            device, args.amp)
        gains_all[m] = gain.cpu().numpy().astype(np.float32)

    beliefs_np = beliefs.cpu().numpy().astype(np.float16)
    return ks, priors_list, post_views_all, beliefs_np, gains_all


def worker(rank, gpu_ids, args, all_entries):
    world_size = len(gpu_ids)
    device = torch.device(f'cuda:{gpu_ids[rank]}')

    push_entries = all_entries[rank::world_size]

    ub_model = load_ub_model(args.ub_checkpoint, device)
    if rank == 0:
        print(f'[rollout] init_belief = {args.init_belief}')
        print(f'[shard] world_size={world_size}  '
              f'this rank ({gpu_ids[rank]}) -> {len(push_entries)} entries')

    rng = np.random.RandomState(args.seed + args.scene_start + rank * 997)

    pbar = tqdm(push_entries, desc=f'gpu{gpu_ids[rank]}',
                position=rank, dynamic_ncols=True)
    for ei, (root_key, scene, push_name, push_dir) in enumerate(pbar):
        out_dir = os.path.join(args.out_root, root_key, scene, push_name)
        os.makedirs(out_dir, exist_ok=True)

        if not args.overwrite and all(
                os.path.isfile(os.path.join(out_dir, f'{m:03d}.npz'))
                for m in range(args.samples_per_scene)):
            pbar.set_postfix_str(f'{root_key}/{scene}/{push_name} (skip)')
            continue

        voxels_pre_np = load_pre_voxels(push_dir)
        voxels_pre_t = torch.from_numpy(voxels_pre_np).to(device)
        swept_np = load_swept_maps(push_dir)
        swept_t = torch.from_numpy(swept_np).to(device)

        ks, priors_list, post_views_all, beliefs_np, gains_np = process_entry(
            ub_model, voxels_pre_t, swept_t, push_dir, rng, args, device)

        for m in range(args.samples_per_scene):
            out_path = os.path.join(out_dir, f'{m:03d}.npz')
            if os.path.isfile(out_path) and not args.overwrite:
                continue
            np.savez(
                out_path,
                current_map=beliefs_np[m],
                gains=gains_np[m],
                k=np.int32(ks[m]),
                prior_cams=priors_list[m].astype(np.int32),
                post_views=post_views_all[m].astype(np.int32),
            )

        pbar.set_postfix_str(
            f'{root_key}/{scene}/{push_name}  '
            f'g_mean={gains_np.mean():.2f}  g_max={gains_np.max():.2f}')

        del voxels_pre_t, swept_t
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    os.makedirs(args.out_root, exist_ok=True)

    gpu_ids = [int(x) for x in str(args.device).split(',') if x.strip() != '']
    if not gpu_ids:
        raise ValueError(f'invalid --device: {args.device!r}')

    all_entries = list_push_dirs(args.data_roots)
    if args.scene_end < 0:
        args.scene_end = len(all_entries)
    all_entries = all_entries[args.scene_start:args.scene_end]
    print(f'[entries] total {len(all_entries)} (scene, push) entries '
          f'[{args.scene_start}:{args.scene_end}]  M={args.samples_per_scene}  '
          f'GPUs={gpu_ids}')

    if len(gpu_ids) == 1:
        worker(0, gpu_ids, args, all_entries)
    else:
        mp.spawn(worker, args=(gpu_ids, args, all_entries),
                 nprocs=len(gpu_ids), join=True)


if __name__ == '__main__':
    main()
