"""Dataset over pre-generated UB reward samples (ycb_v3, NUM_PUSHES=20).

Self-contained copy for 4_policy_test (siamese / permutation-equivariant
reward model experiments). Identical to support_code/dataset_reward_UB.py;
sample tuple shape is unchanged so the new model can drop in.

reward_root layout (output of 3_reward_dataset/UB_reward_generate_gt.py):
    {reward_root}/{occ_subdir}/{scene}/{push_name}/{idx}.npz
npz keys:
    current_map : (1, 60, 120, 80) fp16     UB rollout belief at step k
    gains       : (20,)            fp32     GT marginal-diff per post
    valid_mask  : (20,)            bool     False = fallen/tipped/missing
    k           : int32                     prior rollout depth
    prior_cams  : (k,)             int32

Returns per-sample tensors:
    current     : (1, 60, 120, 80) float32
    swept_maps  : (20, 60, 120, 80) float32   loaded from data_root on demand
    gains       : (20,)             float32
    valid_mask  : (20,)             bool

`data_roots` are the ORIGINAL beliefmap roots (e.g.
/data/APOBU/beliefmap_high_occlusion_ycb_v3). The occ_subdir under each
reward_root must match a basename of one of the data_roots so we can resolve
the absolute path for swept_maps.

Train/val split: scenes are SHUFFLED with split_seed before picking the last
`num_val_scenes` for validation.

I/O optimization:
  - swept_maps for one (scene, push) is 20 small npz files (~50ms warm) and
    is shared across all `samples_per_scene` samples of that push. We
      (a) yield indices in push-locality order via PushBucketSampler so
          consecutive samples within a worker share the same push, and
      (b) keep an in-process functools.lru_cache on _load_swept_maps so
          each push's swept_maps is loaded from disk at most once per worker.
    With persistent_workers=True the cache survives across epochs.
"""
import os
import functools
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler


NUM_PUSHES = 20
D, H, W = 60, 120, 80


def _load_swept_maps(scene_dir, push_name):
    """(NUM_PUSHES, D, H, W) float32.

    Prefers `{push_dir}/swept_stack.npz` (one file open) created by
    3_reward_dataset/stack_swept_maps.py. Falls back to per-view
    `post_XX/action.npz` (20 file opens) when the stacked file is absent.
    """
    push_dir = os.path.join(scene_dir, push_name)
    stack_path = os.path.join(push_dir, 'swept_stack.npz')
    if os.path.isfile(stack_path):
        with np.load(stack_path) as d:
            return d['swept_stack'].astype(np.float32, copy=False)
    arrs = []
    for XX in range(NUM_PUSHES):
        action_path = os.path.join(push_dir, f'post_{XX:02d}', 'action.npz')
        s = np.load(action_path)['swept_map'].transpose(1, 0, 2).astype(np.float32)
        arrs.append(s)
    return np.stack(arrs)


@functools.lru_cache(maxsize=2)
def _load_swept_maps_cached(scene_dir, push_name):
    return _load_swept_maps(scene_dir, push_name)


class RewardDatasetUB(Dataset):
    """Returns (current, swept_maps, gains, valid_mask) per sample."""

    def __init__(self, paths_with_meta):
        self.items = paths_with_meta

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        sample_path, scene_dir, push_name = self.items[idx]
        with np.load(sample_path) as d:
            current = torch.from_numpy(d['current_map'].astype(np.float32))
            gains = torch.from_numpy(d['gains'].astype(np.float32))
            valid_mask = torch.from_numpy(d['valid_mask'].astype(bool))
        swept = _load_swept_maps_cached(scene_dir, push_name)
        swept_t = torch.from_numpy(swept)
        return current, swept_t, gains, valid_mask


class PushBucketSampler(Sampler):
    """Yields sample indices in push-locality order.

    Groups items by (scene_dir, push_name) into buckets. Each epoch:
      - shuffle bucket ORDER (if shuffle_buckets)
      - shuffle indices WITHIN each bucket (if shuffle_within)
    Consecutive yielded indices belong to the same push, so a tiny per-worker
    LRU cache on swept_maps achieves near-100% hit rate after the first sample
    in each push.

    DDP: when world_size > 1, buckets are partitioned across ranks (each bucket
    assigned to exactly one rank per epoch) so locality is preserved within a
    rank. Each rank also shuffles its own assigned-bucket order independently.
    """

    def __init__(self, items, seed=0, shuffle_buckets=True, shuffle_within=True,
                 rank=0, world_size=1, subsample_frac=1.0):
        groups = defaultdict(list)
        for idx, (_path, scene_dir, push_name) in enumerate(items):
            groups[(scene_dir, push_name)].append(idx)
        self.buckets = list(groups.values())
        self.seed = int(seed)
        self.shuffle_buckets = shuffle_buckets
        self.shuffle_within = shuffle_within
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.subsample_frac = float(subsample_frac)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _my_buckets_order(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        n = len(self.buckets)
        order = np.arange(n)
        if self.shuffle_buckets:
            rng.shuffle(order)
        # Subsample at the bucket level so push-locality is preserved.
        # With shuffle_buckets=True (train) we get a different 20% each epoch;
        # with shuffle_buckets=False (val) the same 20% every epoch.
        if self.subsample_frac < 1.0:
            keep = max(1, int(round(n * self.subsample_frac)))
            order = order[:keep]
        if self.world_size > 1:
            order = order[self.rank::self.world_size]
        return order, rng

    def __len__(self):
        order, _ = self._my_buckets_order()
        return int(sum(len(self.buckets[i]) for i in order))

    def __iter__(self):
        order, rng = self._my_buckets_order()
        for bi in order:
            g = list(self.buckets[bi])
            if self.shuffle_within:
                rng.shuffle(g)
            for idx in g:
                yield idx


def _scan(reward_root, data_root_map, k_min, k_max):
    """Return list of (scene_name, sample_path, scene_dir, push_name)."""
    out = []
    if not os.path.isdir(reward_root):
        return out
    for occ in sorted(os.listdir(reward_root)):
        occ_dir = os.path.join(reward_root, occ)
        if not os.path.isdir(occ_dir):
            continue
        if occ not in data_root_map:
            print(f'[scan] WARN: no matching data_root for occ={occ!r} '
                  f'(known: {list(data_root_map)})')
            continue
        data_root_abs = data_root_map[occ]
        for scene in sorted(os.listdir(occ_dir)):
            scene_dir_reward = os.path.join(occ_dir, scene)
            if not os.path.isdir(scene_dir_reward):
                continue
            scene_dir_data = os.path.join(data_root_abs, scene)
            if not os.path.isdir(scene_dir_data):
                continue
            for push in sorted(os.listdir(scene_dir_reward)):
                push_dir = os.path.join(scene_dir_reward, push)
                if not os.path.isdir(push_dir):
                    continue
                for fn in sorted(os.listdir(push_dir)):
                    if not fn.endswith('.npz'):
                        continue
                    path = os.path.join(push_dir, fn)
                    try:
                        with np.load(path) as d:
                            k = int(d['k'])
                            if 'valid_mask' in d.files and not bool(
                                    d['valid_mask'].any()):
                                continue
                    except Exception:
                        continue
                    if k_min is not None and k < k_min:
                        continue
                    if k_max is not None and k > k_max:
                        continue
                    out.append((scene, path, scene_dir_data, push))
    return out


def get_dataloaders(reward_roots, data_roots,
                    batch_size=16, num_workers=4,
                    k_min=2, k_max=5, num_val_scenes=10,
                    split_seed=0,
                    rank=0, world_size=1,
                    subsample_frac=1.0):
    if isinstance(reward_roots, str):
        reward_roots = [reward_roots]
    if isinstance(data_roots, str):
        data_roots = [data_roots]

    data_root_map = {os.path.basename(r.rstrip('/')): r for r in data_roots}

    all_items = []
    for root in reward_roots:
        all_items += _scan(root, data_root_map, k_min, k_max)

    scenes = sorted({s for s, _, _, _ in all_items})
    if num_val_scenes > 0:
        rng = np.random.RandomState(split_seed)
        shuffled = scenes.copy()
        rng.shuffle(shuffled)
        val_scenes = set(shuffled[-num_val_scenes:])
    else:
        val_scenes = set()

    train_items = [(p, sd, pn) for s, p, sd, pn in all_items if s not in val_scenes]
    val_items = [(p, sd, pn) for s, p, sd, pn in all_items if s in val_scenes]

    train_ds = RewardDatasetUB(train_items)
    val_ds = RewardDatasetUB(val_items)

    train_sampler = PushBucketSampler(
        train_items, seed=split_seed,
        shuffle_buckets=True, shuffle_within=True,
        rank=rank, world_size=world_size,
        subsample_frac=subsample_frac,
    )

    common = dict(
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    if num_workers > 0:
        common['prefetch_factor'] = 4

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        sampler=train_sampler, shuffle=False, drop_last=True,
        **common,
    )

    val_loader = None
    if rank == 0:
        val_sampler = PushBucketSampler(
            val_items, seed=split_seed,
            shuffle_buckets=False, shuffle_within=False,
            rank=0, world_size=1,
            subsample_frac=subsample_frac,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            sampler=val_sampler,
            **common,
        )

    return train_loader, val_loader
