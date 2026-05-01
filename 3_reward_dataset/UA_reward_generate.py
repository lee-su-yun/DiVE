"""Pre-generate (current_map, gains[98]) dataset for U_AB_beliefmap UA Reward Model
training, using the UA model trained by train_UAB.py (--task UA).

For each (scene, push_N) with a valid pre_occ/ directory, samples M random
(k, prior_cams) configurations:

  1) k ~ Uniform[1, max_steps]
  2) k prior cams sampled without replacement from 98 cameras
  3) roll out pretrained UA model over prior cams with swept_map=zeros and
     belief init = init_belief (0.5 to match train_UAB._init_belief)
     -> belief_prev  (current_map)
  4) batch-forward all 98 candidate cams on top of belief_prev
     -> belief_c (1 step each, swept_map=zeros)
  5) gain_c = Σ |belief_c - belief_prev|       (L1)

Data source: push_dir/pre_occ/{000..097}.npz (key 'occ'), matching the
per-view loader used in dataset_UAB._load_views.

Filtering (fallen_log_revised.txt × episode_log.txt):
  - push_1/pre_occ is always included (initial state).
  - push_2/pre_occ is excluded iff the AB transition's actually-selected
    col_idx is in fallen for (scene, 1, col_idx).
  - push_3/pre_occ is excluded iff push_2 is excluded OR the BC transition's
    actually-selected col_idx is in fallen for (scene, 2, col_idx).
  - If fallen_log or episode_log is missing, no filtering is applied.

Saves one .npz per sample at:
    {out_root}/{scene_root_key}/{scene_id}/{push_name}/{sample_idx:03d}.npz
keys:
    current_map : (1, 60, 120, 80) float16
    gains       : (98,) float32
    k           : int32
    prior_cams  : (k,) int32  — camera numbers in [0, 98)

명령어 :
    cd /home/sylee/codes/DiVE/3_reward_dataset
    python UA_reward_generate.py \
    --data_roots /data/APOBU/beliefmap_low_occlusion_0423 \
                 /data/APOBU/beliefmap_high_occlusion_0423 \
    --ua_checkpoint /result/APOBU/U_AB_beliefmap/UA_Kt_v1/best.pth \
    --out_root /data/APOBU/U_AB_beliefmap/ua_Kt_reward_dataset \
    --init_belief 0.0 \
    --device 0,1,2,3 \
    --chunk 128
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
from model_UAB import UNet3DPush
from dataset_UAB import parse_fallen_log, parse_episode_log

NUM_CAMERAS = 98
D, H, W = 60, 120, 80


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_roots', type=str, nargs='+', default=[
        '/data/APOBU/beliefmap_low_occlusion_0422'
    ])
    p.add_argument('--out_root', type=str,
                   default='/data/APOBU/U_AB_beliefmap/ua_reward_dataset')
    p.add_argument('--ua_checkpoint', type=str,
                   default='/result/APOBU/U_AB_beliefmap/UA/best.pth')
    p.add_argument('--init_belief', type=float, default=0.0,
                   help='Initial belief value used during rollout. Must match '
                        'train_UAB._init_belief (0.5 by default).')
    p.add_argument('--samples_per_scene', type=int, default=64,
                   help='M samples per (scene, push_N)')
    p.add_argument('--max_steps', type=int, default=8,
                   help='k ~ U[1, max_steps]')
    p.add_argument('--device', type=str, default='0',
                   help='GPU id, or comma-separated list for multi-GPU sharding.')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--chunk', type=int, default=64,
                   help='forward batch size for gain computation.')
    p.add_argument('--amp', action='store_true', default=True,
                   help='use fp16 autocast for forwards (default on).')
    p.add_argument('--no_amp', dest='amp', action='store_false')
    p.add_argument('--scene_start', type=int, default=0)
    p.add_argument('--scene_end', type=int, default=-1)
    p.add_argument('--overwrite', action='store_true')
    return p.parse_args()


def load_ua_model(ckpt_path, device):
    model = UNet3DPush().to(device)
    model.eval()
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f'[UA] loaded {ckpt_path}  '
          f'epoch={ckpt.get("epoch", "?")}  '
          f'val_loss={ckpt.get("val_loss", float("nan")):.4f}')
    return model


def _scene_bad_pushes(scene_dir, scene_name, fallen):
    """Return set of push_ids whose pre_occ is downstream of a fallen event.

    push_1 is always safe. For push_N in {2, 3}, look up the transition in
    episode_log.txt whose post_push == N; let its pre_push=P and col_idx=C.
    push_N is excluded if (scene, P, C) is in fallen, or if push_P itself is
    already excluded (upstream fallen propagates downstream).
    """
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

    Requires push_dir/pre_occ/{000..097}.npz to all exist. Filters pushes whose
    pre_occ is downstream of a fallen transition (see _scene_bad_pushes).
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
                results.append((root_key, scene, push_name, push_dir))
    return results


def load_all_voxels(push_dir):
    """Load 98 per-view occupancy files -> (98, D, H, W) float32."""
    pre_occ_dir = os.path.join(push_dir, 'pre_occ')
    arrs = [np.load(os.path.join(pre_occ_dir, f'{v:03d}.npz'))['occ']
            for v in range(NUM_CAMERAS)]
    return np.stack(arrs).astype(np.float32)


def _maybe_autocast(use_amp):
    if use_amp:
        return torch.amp.autocast('cuda', dtype=torch.float16)
    import contextlib
    return contextlib.nullcontext()


@torch.no_grad()
def rollout_prior_batched(ua_model, voxels_all_t, priors_mat, device,
                          init_belief, use_amp):
    """Batched rollout for B samples sharing the same k steps.

    priors_mat: (B, k) long tensor of camera indices
    returns:    (B, 1, D, H, W) belief_prev
    """
    B, k = priors_mat.shape
    belief = torch.full((B, 1, D, H, W), init_belief, device=device)
    swept = torch.zeros((B, 1, D, H, W), device=device)
    for t in range(k):
        idx = priors_mat[:, t]
        voxel = voxels_all_t.index_select(0, idx).unsqueeze(1)
        x = torch.cat([belief, voxel, swept], dim=1)
        with _maybe_autocast(use_amp):
            out = ua_model(x)
        belief = torch.sigmoid(out.float())
    return belief


@torch.no_grad()
def compute_gains_batched(ua_model, beliefs_all, voxels_all_t, fwd_batch,
                          device, use_amp):
    """Compute squared gain for every (sample, camera) pair in one pass.

    beliefs_all : (M, 1, D, H, W)
    returns     : (M, 98) gains
    """
    M = beliefs_all.shape[0]
    C = NUM_CAMERAS
    gains = torch.empty((M, C), device=device)

    cn = max(1, fwd_batch // max(1, M))
    for c_start in range(0, C, cn):
        c_end = min(c_start + cn, C)
        cb = c_end - c_start

        bel_exp = beliefs_all.unsqueeze(1).expand(M, cb, 1, D, H, W).reshape(
            M * cb, 1, D, H, W)
        vox = voxels_all_t[c_start:c_end].view(1, cb, 1, D, H, W).expand(
            M, cb, 1, D, H, W).reshape(M * cb, 1, D, H, W)
        swept = torch.zeros((M * cb, 1, D, H, W), device=device)
        x = torch.cat([bel_exp, vox, swept], dim=1)
        with _maybe_autocast(use_amp):
            out = ua_model(x)
        belief_c = torch.sigmoid(out.float())
        delta = belief_c - bel_exp
        g = delta.abs().sum(dim=(1, 2, 3, 4)).view(M, cb)
        gains[:, c_start:c_end] = g

    return gains


@torch.no_grad()
def process_entry(ua_model, voxels_all_t, rng, args, device):
    M = args.samples_per_scene
    ks = rng.randint(1, args.max_steps + 1, size=M).astype(np.int32)
    priors_list = [rng.choice(NUM_CAMERAS, int(k), replace=False) for k in ks]

    beliefs = torch.empty((M, 1, D, H, W), device=device)
    by_k = defaultdict(list)
    for m, k in enumerate(ks):
        by_k[int(k)].append(m)
    for k, members in by_k.items():
        priors_mat = torch.tensor(
            np.stack([priors_list[m] for m in members]),
            dtype=torch.long, device=device)
        belief_group = rollout_prior_batched(
            ua_model, voxels_all_t, priors_mat, device,
            args.init_belief, args.amp)
        for i, m in enumerate(members):
            beliefs[m] = belief_group[i]

    gains = compute_gains_batched(
        ua_model, beliefs, voxels_all_t, args.chunk, device, args.amp)

    beliefs_np = beliefs.cpu().numpy().astype(np.float16)
    gains_np = gains.cpu().numpy().astype(np.float32)
    return ks, priors_list, beliefs_np, gains_np


def worker(rank, gpu_ids, args, all_entries):
    world_size = len(gpu_ids)
    device = torch.device(f'cuda:{gpu_ids[rank]}')

    push_entries = all_entries[rank::world_size]

    ua_model = load_ua_model(args.ua_checkpoint, device)
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

        voxels_np = load_all_voxels(push_dir)
        voxels_all_t = torch.from_numpy(voxels_np).to(device)

        ks, priors_list, beliefs_np, gains_np = process_entry(
            ua_model, voxels_all_t, rng, args, device)

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
            )

        pbar.set_postfix_str(
            f'{root_key}/{scene}/{push_name}  '
            f'g_mean={gains_np.mean():.2f}  g_max={gains_np.max():.2f}')

        del voxels_all_t
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
