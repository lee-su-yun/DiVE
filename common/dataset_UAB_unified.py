"""Unified UA+UB dataset (view-only sequence, no push transition).

One sample = one push_X directory; the post_XX states are never read.
The sequence is `num_steps` views sampled at random from push_X/pre_occ/.

Returns:
    voxel_maps   : (num_steps, 60, 120, 80) float32 — per-view occupancy
    per_view_pre : (num_steps, 60, 120, 80) uint8   — pre_ray_casting[K] (binary)
    gt_marginal  : (60, 120, 80)            float32 — push_visibility_all_marginal.npy

The training loop builds the UA target from the cumulative mean of per_view_pre,
and uses gt_marginal directly as the UB target (it is time-invariant).
"""
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from data_utils import parse_fallen_log, _worker_init_fn, _load_views


NUM_VIEWS = 98


def _load_ray_casting_views(path, view_indices):
    """mmap, slice the requested views, unpackbits -> (n_views, 60, 120, 80) uint8 binary."""
    packed = np.load(path, mmap_mode='r')                  # (98, 60, 120, 10) uint8
    sliced = np.ascontiguousarray(packed[view_indices])    # (n_views, 60, 120, 10)
    return np.unpackbits(sliced, axis=-1)                  # (n_views, 60, 120, 80)


def _sample_views(num_steps, rng):
    """Sample `num_steps` distinct view indices from [0, NUM_VIEWS)."""
    if hasattr(rng, 'integers'):
        idx = rng.choice(NUM_VIEWS, num_steps, replace=False)
    else:
        idx = rng.choice(NUM_VIEWS, num_steps, replace=False)
    return np.sort(idx).astype(np.int64)


class UABUnifiedDataset(Dataset):
    def __init__(self, samples, num_steps=10, deterministic=False):
        self.samples = samples
        self.num_steps = num_steps
        self.deterministic = deterministic

        if deterministic:
            rng = np.random.RandomState(42)
            self._fixed = [_sample_views(num_steps, rng) for _ in samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        if self.deterministic:
            views = self._fixed[idx]
        else:
            views = _sample_views(self.num_steps, np.random)

        # Per-view occupancy (model input)
        voxels = _load_views(s['pre_occ_dir'], views)  # (num_steps, 60, 120, 80) float32

        # Per-view binary visibility (UA GT raw)
        per_view_pre = _load_ray_casting_views(s['pre_ray_casting_path'], views)

        # UB GT: scalar marginal (time-invariant, per push)
        gt_marginal = np.load(s['gt_marginal_path']).astype(np.float32)

        return (
            torch.from_numpy(voxels),         # (num_steps, 60, 120, 80) float32
            torch.from_numpy(per_view_pre),   # (num_steps, 60, 120, 80) uint8
            torch.from_numpy(gt_marginal),    # (60, 120, 80) float32
        )


def _build_samples(scene_dir, scene_name, excluded, is_val, train_out, val_out):
    """One sample per push_X directory.

    `excluded` is a set of (scene, push_id, post_id) tuples, but this dataset never
    touches individual posts, so the per-post exclusions are ignored rather than
    escalated into "drop the whole push". That is safe because gt_marginal was
    already built with the fallen/tipped posts left out.
    """
    for push_name in sorted(os.listdir(scene_dir)):
        if not push_name.startswith('push_'):
            continue
        try:
            push_id = int(push_name.split('_')[1])
        except (ValueError, IndexError):
            continue

        push_dir = os.path.join(scene_dir, push_name)
        pre_occ_dir = os.path.join(push_dir, 'pre_occ')
        pre_rc_path = os.path.join(push_dir, 'pre_ray_casting.npy')
        gt_marginal_path = os.path.join(push_dir, 'push_visibility_all_marginal.npy')

        if not (os.path.isdir(pre_occ_dir)
                and os.path.isfile(pre_rc_path)
                and os.path.isfile(gt_marginal_path)):
            continue

        sample = {
            'pre_occ_dir': pre_occ_dir,
            'pre_ray_casting_path': pre_rc_path,
            'gt_marginal_path': gt_marginal_path,
            'scene_name': scene_name,
            'push_id': push_id,
        }
        (val_out if is_val else train_out).append(sample)


def get_dataloaders(data_roots, batch_size=8, num_steps=10,
                    num_val_scenes=10, num_workers=4, rank=0, world_size=1):
    all_scenes = set()
    for data_root in data_roots:
        for d in os.listdir(data_root):
            if os.path.isdir(os.path.join(data_root, d)):
                all_scenes.add(d)
    sorted_scenes = sorted(all_scenes)
    val_scene_set = set(sorted_scenes[-num_val_scenes:])

    train_samples, val_samples = [], []
    for data_root in data_roots:
        fallen = parse_fallen_log(os.path.join(data_root, 'fallen_log_revised.txt'))
        tipped = parse_fallen_log(os.path.join(data_root, 'tipped_log_revised.txt'))
        excluded = fallen | tipped
        print(f'[UAB_unified] {data_root}: fallen={len(fallen)} tipped={len(tipped)} '
              f'excluded={len(excluded)}')
        scene_names = sorted(d for d in os.listdir(data_root)
                             if os.path.isdir(os.path.join(data_root, d)))
        for name in scene_names:
            is_val = name in val_scene_set
            _build_samples(os.path.join(data_root, name), name, excluded, is_val,
                           train_samples, val_samples)

    train_dataset = UABUnifiedDataset(train_samples, num_steps=num_steps, deterministic=False)

    loader_kwargs = dict(num_workers=num_workers, pin_memory=True,
                         worker_init_fn=_worker_init_fn)

    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, drop_last=True,
        shuffle=(train_sampler is None), sampler=train_sampler,
        **loader_kwargs)

    val_loader = None
    if rank == 0:
        val_dataset = UABUnifiedDataset(val_samples, num_steps=num_steps, deterministic=True)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            **loader_kwargs)

    return train_loader, val_loader
