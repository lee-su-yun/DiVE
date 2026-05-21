"""GT-marginal UA+UB Reward dataset generator for the unified UAB model.

Uses the unified UAB checkpoint trained by 2_belief_unified/train_UAB_unified.py
(model: support_code/model_UAB_unified.UNet3DPushUnified, 3-ch input
[belief_A, belief_B, voxel], 2-ch output [logit_A, logit_B], no swept_map).

One unified rollout per sample produces BOTH belief_A and belief_B at step k.
Each (k, prior_cams) configuration is shared between the UA and UB sides, so
we write two parallel output trees that match the schemas expected by the
downstream training scripts (4_policy/train_reward_UA.py and
4_policy_test/train_reward_UB.py).

Per (scene, push_N), sample M random configurations:

  1) k ~ U[1, max_steps]
  2) k prior cams (no replacement) from 98 cams of push_N/pre_occ
  3) Unified rollout (init: belief_a=init_belief_a, belief_b=init_belief_b):
       for t in range(k):
           voxel = pre_voxels[priors[t]]
           x = cat([belief_a, belief_b, voxel], dim=1)     # (B, 3, D, H, W)
           logits = uab_model(x)                           # (B, 2, D, H, W)
           belief_a = sigmoid(logits[:, 0:1])
           belief_b = sigmoid(logits[:, 1:2])

  4) UA gains[98] (k-dependent, algebraic shortcut from pre_ray_casting):
       GT_t       = (1/k) Σ_{i∈P} pre_rc[i]
       gain_a[v]  = (1/(k+1)) · Σ_voxel |pre_rc[v] − GT_t|

  5) UB gains[num_posts] (k-independent, from post_NN/post_ray_casting):
       pre_marg   = pre_rc.mean(0)
       post_marg  = post_rc_j.mean(0)
       gain_b[j]  = Σ_voxel |post_marg − pre_marg|
       valid_mask[j] = False if (scene, push_N, j) in fallen∪tipped or post
                        ray_casting missing.

Filtering:
  push_N excluded if its upstream transition was in fallen∪tipped, propagated
  along episode_log.txt (generalized _scene_bad_pushes, same as the original
  GT generators).

Outputs (two parallel trees):
  {out_root_ua}/{root_key}/{scene}/{push_name}/{m:03d}.npz
    current_map (1, 60, 120, 80) fp16    UA belief at step k
    gains       (98,)             fp32
    k           int32
    prior_cams  (k,)              int32

  {out_root_ub}/{root_key}/{scene}/{push_name}/{m:03d}.npz
    current_map (1, 60, 120, 80) fp16    UB belief at step k
    gains       (num_posts,)      fp32
    valid_mask  (num_posts,)      bool
    k           int32
    prior_cams  (k,)              int32

명령어 :
    cd /home/sylee/codes/DiVE/3_reward_dataset_unified
    python UAB_reward_generate_gt.py \\
        --data_roots /data/APOBU/beliefmap_high_occlusion_ycb_v3 \\
                     /result/DiVE_data/beliefmap_low_occlusion_ycb_v3 \\
        --uab_checkpoint /result/APOBU/DiVE/UAB_unified_ycb_v3_TFanneal/best.pth \\
        --out_root_ua /data/APOBU/DiVE/ua_reward_dataset_ycb_v3_unified \\
        --out_root_ub /data/APOBU/DiVE/ub_reward_dataset_ycb_v3_unified \\
        --init_belief_a 0.0 --init_belief_b 0.0 \\
        --num_posts 20 \\
        --device 4,5 --samples_per_scene 64 --io_threads 8
"""
import os
import sys
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "support_code"))
from model_UAB_unified import UNet3DPushUnified  # noqa: E402
from dataset_UAB import parse_fallen_log, parse_episode_log  # noqa: E402

NUM_CAMERAS = 98
D, H, W = 60, 120, 80


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_roots', type=str, nargs='+', required=True)
    p.add_argument('--out_root_ua', type=str, required=True,
                   help='Output tree for UA reward samples '
                        '(current_map=belief_A, gains[98]).')
    p.add_argument('--out_root_ub', type=str, required=True,
                   help='Output tree for UB reward samples '
                        '(current_map=belief_B, gains[num_posts], valid_mask).')
    p.add_argument('--uab_checkpoint', type=str, required=True,
                   help='Path to unified UAB best.pth.')
    p.add_argument('--num_posts', type=int, default=20,
                   help='Number of post_XX directories per push (ycb_v3=20).')
    p.add_argument('--init_belief_a', type=float, default=0.0,
                   help='Match train_UAB_unified.py --init_belief_a (0.0).')
    p.add_argument('--init_belief_b', type=float, default=0.0,
                   help='Match train_UAB_unified.py --init_belief_b (0.0).')
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
    p.add_argument('--io_threads', type=int, default=8,
                   help='Threads for prefetch / parallel I/O per entry.')
    p.add_argument('--prior_a', type=float, default=0.1,
                   help='Match train_UAB_unified.py --prior_a (only used for '
                        'model instantiation; weights overridden by checkpoint).')
    p.add_argument('--prior_b', type=float, default=0.1,
                   help='Match train_UAB_unified.py --prior_b (only used for '
                        'model instantiation; weights overridden by checkpoint).')
    return p.parse_args()


def load_uab_model(ckpt_path, device, prior_a, prior_b):
    model = UNet3DPushUnified(prior_a=prior_a, prior_b=prior_b).to(device)
    model.eval()
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f'[UAB] loaded {ckpt_path}  '
          f'epoch={ckpt.get("epoch", "?")}  '
          f'val_loss_sf={ckpt.get("val_loss_selffeed", float("nan")):.4f}  '
          f'val_loss_tf={ckpt.get("val_loss_tf", float("nan")):.4f}')
    return model


def _scene_bad_pushes(scene_dir, scene_name, excluded):
    """Propagate fallen/tipped downstream through episode_log transitions."""
    bad = set()
    log_path = os.path.join(scene_dir, 'episode_log.txt')
    if not os.path.isfile(log_path):
        return bad
    transitions = parse_episode_log(log_path)
    by_post = {t['post_push']: t for t in transitions}
    for post_push in sorted(by_post.keys()):
        t = by_post[post_push]
        if t['pre_push'] in bad:
            bad.add(post_push)
            continue
        if (scene_name, t['pre_push'], t['col_idx']) in excluded:
            bad.add(post_push)
    return bad


def list_push_dirs(data_roots):
    """Return (entries, excluded_per_root).

    entry = (root_key, scene_name, push_name, push_dir, push_id)
    Requires push_dir/pre_ray_casting.npy AND
            (pre_occ.npz OR pre_occ/{000..097}.npz).
    """
    results = []
    excluded_per_root = {}
    for data_root in data_roots:
        root_key = os.path.basename(data_root.rstrip('/'))
        fallen = parse_fallen_log(os.path.join(data_root, 'fallen_log_revised.txt'))
        tipped = parse_fallen_log(os.path.join(data_root, 'tipped_log_revised.txt'))
        excluded = fallen | tipped
        excluded_per_root[root_key] = excluded
        scene_names = sorted(d for d in os.listdir(data_root)
                             if os.path.isdir(os.path.join(data_root, d)))
        for scene in scene_names:
            scene_dir = os.path.join(data_root, scene)
            bad_pushes = _scene_bad_pushes(scene_dir, scene, excluded)
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
                pre_rc_path = os.path.join(push_dir, 'pre_ray_casting.npy')
                if not os.path.isfile(pre_rc_path):
                    continue
                stacked = os.path.join(push_dir, 'pre_occ.npz')
                per_view_dir = os.path.join(push_dir, 'pre_occ')
                if not (os.path.isfile(stacked)
                        or (os.path.isdir(per_view_dir) and
                            all(os.path.isfile(os.path.join(per_view_dir,
                                                            f'{v:03d}.npz'))
                                for v in range(NUM_CAMERAS)))):
                    continue
                results.append((root_key, scene, push_name, push_dir, push_id))
    return results, excluded_per_root


def load_pre_voxels(push_dir):
    """(98, D, H, W) float32. Prefer stacked pre_occ.npz['occupied']."""
    stacked = os.path.join(push_dir, 'pre_occ.npz')
    if os.path.isfile(stacked):
        with np.load(stacked) as d:
            key = 'occupied' if 'occupied' in d.files else d.files[0]
            return d[key].astype(np.float32, copy=False)
    pre_occ_dir = os.path.join(push_dir, 'pre_occ')
    arrs = [np.load(os.path.join(pre_occ_dir, f'{v:03d}.npz'))['occ']
            for v in range(NUM_CAMERAS)]
    return np.stack(arrs).astype(np.float32)


def load_pre_rc(push_dir):
    """(98, D, H, W) float32 from unpacking pre_ray_casting.npy."""
    packed = np.load(os.path.join(push_dir, 'pre_ray_casting.npy'))
    return np.unpackbits(packed, axis=-1).astype(np.float32, copy=False)


def _unpack_rc_marginal(path):
    """mean over 98 cams of unpacked ray_casting -> (D, H, W) float64."""
    packed = np.load(path)
    return np.unpackbits(packed, axis=-1).mean(axis=0, dtype=np.float64)


def _load_post_marginal(post_rc_path):
    if not os.path.isfile(post_rc_path):
        return None
    return _unpack_rc_marginal(post_rc_path)


def compute_ub_gt_gains(push_dir, scene_name, push_id, excluded, num_posts,
                        io_pool=None):
    """Return (gains[num_posts] float32, valid_mask[num_posts] bool).

    fallen/tipped or missing post_ray_casting.npy → gain=0, mask=False.
    pre_marginal computed in parallel with the per-post marginals via io_pool.
    """
    paths = []  # (j, post_rc_path)
    for j in range(num_posts):
        if (scene_name, push_id, j) in excluded:
            continue
        paths.append(
            (j, os.path.join(push_dir, f'post_{j:02d}', 'post_ray_casting.npy')))

    pre_rc_path = os.path.join(push_dir, 'pre_ray_casting.npy')
    if io_pool is not None:
        pre_future = io_pool.submit(_unpack_rc_marginal, pre_rc_path)
        post_futures = [(j, io_pool.submit(_load_post_marginal, p))
                        for j, p in paths]
        pre_marginal = pre_future.result()
        marginals = [(j, f.result()) for j, f in post_futures]
    else:
        pre_marginal = _unpack_rc_marginal(pre_rc_path)
        marginals = [(j, _load_post_marginal(p)) for j, p in paths]

    gains = np.zeros(num_posts, dtype=np.float32)
    valid_mask = np.zeros(num_posts, dtype=bool)
    for j, post_marginal in marginals:
        if post_marginal is None:
            continue
        gains[j] = float(np.abs(post_marginal - pre_marginal).sum())
        valid_mask[j] = True
    return gains, valid_mask


def _prepare_entry(entry, excluded_per_root, num_posts, io_pool):
    """Run all CPU/I/O for one entry: load voxels + pre_rc + compute UB GT."""
    root_key, scene, push_name, push_dir, push_id = entry
    ub_gains, valid_mask = compute_ub_gt_gains(
        push_dir, scene, push_id, excluded_per_root[root_key],
        num_posts, io_pool=io_pool)
    voxels = load_pre_voxels(push_dir)
    pre_rc = load_pre_rc(push_dir)
    return voxels, pre_rc, ub_gains, valid_mask


def _maybe_autocast(use_amp):
    if use_amp:
        return torch.amp.autocast('cuda', dtype=torch.float16)
    import contextlib
    return contextlib.nullcontext()


@torch.no_grad()
def rollout_prior_unified_batched(uab_model, voxels_t, priors_mat, device,
                                  init_belief_a, init_belief_b, use_amp):
    """Unified rollout for B samples sharing the same k.

    priors_mat: (B, k) long tensor of camera indices
    returns: (belief_a, belief_b)  each (B, 1, D, H, W) at step k
    """
    B, k = priors_mat.shape
    belief_a = torch.full((B, 1, D, H, W), init_belief_a, device=device)
    belief_b = torch.full((B, 1, D, H, W), init_belief_b, device=device)
    for t in range(k):
        idx = priors_mat[:, t]
        voxel = voxels_t.index_select(0, idx).unsqueeze(1)
        x = torch.cat([belief_a, belief_b, voxel], dim=1)   # (B, 3, D, H, W)
        with _maybe_autocast(use_amp):
            logits = uab_model(x)                           # (B, 2, D, H, W)
        belief_a = torch.sigmoid(logits[:, 0:1].float())
        belief_b = torch.sigmoid(logits[:, 1:2].float())
    return belief_a, belief_b


@torch.no_grad()
def rollout_all_samples_unified(uab_model, voxels_t, ks, priors_list,
                                init_belief_a, init_belief_b, use_amp, device):
    """Group M samples by k → batched unified rollout."""
    M = len(ks)
    beliefs_a = torch.empty((M, 1, D, H, W), device=device)
    beliefs_b = torch.empty((M, 1, D, H, W), device=device)
    by_k = defaultdict(list)
    for m, k in enumerate(ks):
        by_k[int(k)].append(m)
    for k, members in by_k.items():
        priors_mat = torch.tensor(
            np.stack([priors_list[m] for m in members]),
            dtype=torch.long, device=device)
        ba, bb = rollout_prior_unified_batched(
            uab_model, voxels_t, priors_mat, device,
            init_belief_a, init_belief_b, use_amp)
        for i, m in enumerate(members):
            beliefs_a[m] = ba[i]
            beliefs_b[m] = bb[i]
    return beliefs_a, beliefs_b


@torch.no_grad()
def compute_ua_gains_for_samples(pre_rc_t, ks, priors_list, device):
    """UA gains per sample: (M, 98) float32 via algebraic shortcut."""
    M = len(ks)
    gains = torch.empty((M, NUM_CAMERAS), device=device)
    for m in range(M):
        k = int(ks[m])
        priors = torch.as_tensor(priors_list[m], device=device, dtype=torch.long)
        GT_t = pre_rc_t.index_select(0, priors).mean(dim=0, keepdim=True)
        gains[m] = (pre_rc_t - GT_t).abs().sum(dim=(1, 2, 3)) / (k + 1)
    return gains


def _all_done(out_dir_ua, out_dir_ub, M, overwrite):
    if overwrite:
        return False
    if not (os.path.isdir(out_dir_ua) and os.path.isdir(out_dir_ub)):
        return False
    for m in range(M):
        if not (os.path.isfile(os.path.join(out_dir_ua, f'{m:03d}.npz'))
                and os.path.isfile(os.path.join(out_dir_ub, f'{m:03d}.npz'))):
            return False
    return True


def worker(rank, gpu_ids, args, all_entries, excluded_per_root):
    world_size = len(gpu_ids)
    device = torch.device(f'cuda:{gpu_ids[rank]}')

    my_entries = all_entries[rank::world_size]

    uab_model = load_uab_model(args.uab_checkpoint, device,
                               args.prior_a, args.prior_b)
    if rank == 0:
        print(f'[rollout] init_belief_a={args.init_belief_a}  '
              f'init_belief_b={args.init_belief_b}  num_posts={args.num_posts}')
        print(f'[shard] world_size={world_size}  '
              f'this rank ({gpu_ids[rank]}) -> {len(my_entries)} entries')

    M = args.samples_per_scene
    work_entries = []
    for entry in my_entries:
        root_key, scene, push_name, _, _ = entry
        out_dir_ua = os.path.join(args.out_root_ua, root_key, scene, push_name)
        out_dir_ub = os.path.join(args.out_root_ub, root_key, scene, push_name)
        if _all_done(out_dir_ua, out_dir_ub, M, args.overwrite):
            continue
        work_entries.append(entry)

    if rank == 0:
        print(f'[shard] rank {gpu_ids[rank]}: {len(work_entries)} to do '
              f'({len(my_entries) - len(work_entries)} already done)')

    rng = np.random.RandomState(args.seed + args.scene_start + rank * 997)

    if not work_entries:
        return

    with ThreadPoolExecutor(max_workers=args.io_threads) as io_pool, \
         ThreadPoolExecutor(max_workers=1) as prefetch_pool:

        next_future = prefetch_pool.submit(
            _prepare_entry, work_entries[0], excluded_per_root,
            args.num_posts, io_pool)

        pbar = tqdm(work_entries, desc=f'gpu{gpu_ids[rank]}',
                    position=rank, dynamic_ncols=True)
        for ei, entry in enumerate(pbar):
            voxels_np, pre_rc_np, ub_gains, valid_mask = next_future.result()
            if ei + 1 < len(work_entries):
                next_future = prefetch_pool.submit(
                    _prepare_entry, work_entries[ei + 1],
                    excluded_per_root, args.num_posts, io_pool)

            root_key, scene, push_name, push_dir, push_id = entry
            out_dir_ua = os.path.join(args.out_root_ua, root_key, scene, push_name)
            out_dir_ub = os.path.join(args.out_root_ub, root_key, scene, push_name)
            os.makedirs(out_dir_ua, exist_ok=True)
            os.makedirs(out_dir_ub, exist_ok=True)

            voxels_t = torch.from_numpy(voxels_np).to(device, non_blocking=True)
            pre_rc_t = torch.from_numpy(pre_rc_np).to(device, non_blocking=True)

            ks = rng.randint(1, args.max_steps + 1, size=M).astype(np.int32)
            priors_list = [rng.choice(NUM_CAMERAS, int(k), replace=False)
                           for k in ks]

            ua_gains = compute_ua_gains_for_samples(
                pre_rc_t, ks, priors_list, device)
            ua_gains_np = ua_gains.cpu().numpy().astype(np.float32)
            del pre_rc_t

            beliefs_a, beliefs_b = rollout_all_samples_unified(
                uab_model, voxels_t, ks, priors_list,
                args.init_belief_a, args.init_belief_b, args.amp, device)
            beliefs_a_np = beliefs_a.cpu().numpy().astype(np.float16)
            beliefs_b_np = beliefs_b.cpu().numpy().astype(np.float16)

            for m in range(M):
                ua_path = os.path.join(out_dir_ua, f'{m:03d}.npz')
                if args.overwrite or not os.path.isfile(ua_path):
                    np.savez(
                        ua_path,
                        current_map=beliefs_a_np[m],
                        gains=ua_gains_np[m],
                        k=np.int32(ks[m]),
                        prior_cams=priors_list[m].astype(np.int32),
                    )
                ub_path = os.path.join(out_dir_ub, f'{m:03d}.npz')
                if args.overwrite or not os.path.isfile(ub_path):
                    np.savez(
                        ub_path,
                        current_map=beliefs_b_np[m],
                        gains=ub_gains,
                        valid_mask=valid_mask,
                        k=np.int32(ks[m]),
                        prior_cams=priors_list[m].astype(np.int32),
                    )

            ub_g_str = (
                f"ub_g_mean={ub_gains[valid_mask].mean():.2f}"
                if valid_mask.any() else "ub_g_mean=N/A")
            pbar.set_postfix_str(
                f'{root_key}/{scene}/{push_name}  '
                f'ua_g_mean={ua_gains_np.mean():.2f}  '
                f'{ub_g_str}  n_valid={int(valid_mask.sum())}/{args.num_posts}')

            del voxels_t, beliefs_a, beliefs_b
            torch.cuda.empty_cache()


def main():
    args = parse_args()
    os.makedirs(args.out_root_ua, exist_ok=True)
    os.makedirs(args.out_root_ub, exist_ok=True)

    gpu_ids = [int(x) for x in str(args.device).split(',') if x.strip() != '']
    if not gpu_ids:
        raise ValueError(f'invalid --device: {args.device!r}')

    all_entries, excluded_per_root = list_push_dirs(args.data_roots)
    if args.scene_end < 0:
        args.scene_end = len(all_entries)
    all_entries = all_entries[args.scene_start:args.scene_end]
    print(f'[entries] total {len(all_entries)} (scene, push) entries '
          f'[{args.scene_start}:{args.scene_end}]  M={args.samples_per_scene}  '
          f'GPUs={gpu_ids}  num_posts={args.num_posts}')

    if len(gpu_ids) == 1:
        worker(0, gpu_ids, args, all_entries, excluded_per_root)
    else:
        mp.spawn(worker, args=(gpu_ids, args, all_entries, excluded_per_root),
                 nprocs=len(gpu_ids), join=True)


if __name__ == '__main__':
    main()
