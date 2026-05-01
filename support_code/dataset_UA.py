"""UA-only dataset that returns per-view binary visibility for cumulative-mean GT.

각 sample 반환 (학습 step loop에서 GT를 step별로 cumulative mean으로 만들기 위함):
    voxel_maps    : (num_steps, 60, 120, 80) float32 — per-view occupancy (occ)
    swept_maps    : (num_steps, 60, 120, 80) float32 — push step에 한해 swept_map
    per_view_pre  : (num_steps, 60, 120, 80) uint8   — v_pre[K[t]] (binary)
    per_view_post : (num_steps, 60, 120, 80) uint8   — v_post[K[t]] (binary)

K = concat(pre_views, post_views) (길이 num_steps)
v_pre는 push_X/pre_ray_casting.npy, v_post는 push_X/post_NN/post_ray_casting.npy에서 mmap+unpackbits.
"""

import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler

# 같은 support_code/ 폴더 내 dataset_UAB.py에서 import
from dataset_UAB import (
    parse_fallen_log,
    _sample_view_schedule,
    _worker_init_fn,
    _load_views,
)


def _load_ray_casting_views(path, view_indices):
    """mmap 후 K 인덱스 슬라이스 → unpackbits → (n_views, 60, 120, 80) uint8 binary."""
    packed = np.load(path, mmap_mode='r')          # (98, 60, 120, 10) uint8
    sliced = np.ascontiguousarray(packed[view_indices])  # (n_views, 60, 120, 10)
    return np.unpackbits(sliced, axis=-1)          # (n_views, 60, 120, 80)


class UADataset(Dataset):
    def __init__(self, samples, num_steps=10, deterministic=False):
        self.samples = samples
        self.num_steps = num_steps
        self.deterministic = deterministic

        if deterministic:
            rng = np.random.RandomState(42)
            self._fixed = [
                _sample_view_schedule(s['col_idx'], num_steps, rng) for s in samples
            ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        if self.deterministic:
            push_step, pre_views, post_views = self._fixed[idx]
        else:
            push_step, pre_views, post_views = _sample_view_schedule(
                s['col_idx'], self.num_steps, np.random)

        # Per-view occupancy (model 입력) — 기존 UABDataset과 동일
        pre_voxels = _load_views(s['pre_occ_dir'], pre_views)
        post_voxels = _load_views(s['post_occ_dir'], post_views)
        voxels = np.concatenate([pre_voxels, post_voxels], axis=0)  # (num_steps, 60, 120, 80)

        # Swept map (push step에만)
        swept_map = np.load(s['action_path'])['swept_map']
        swept_map = swept_map.transpose(1, 0, 2).astype(np.float32)
        swept_maps = np.zeros((self.num_steps, 60, 120, 80), dtype=np.float32)
        swept_maps[push_step] = swept_map

        # Per-view binary visibility (GT용 raw): K = pre_views ∪ post_views (순서 유지)
        K = np.concatenate([pre_views, post_views]).astype(np.int64)
        per_view_pre = _load_ray_casting_views(s['pre_ray_casting_path'], K)
        per_view_post = _load_ray_casting_views(s['post_ray_casting_path'], K)

        return (
            torch.from_numpy(voxels),
            torch.from_numpy(swept_maps),
            torch.from_numpy(per_view_pre),    # uint8
            torch.from_numpy(per_view_post),   # uint8
        )


def _build_ua_samples(scene_dir, scene_name, excluded, is_val, train_out, val_out):
    """UA: (push_n, post_XX) pair 단위. pre_ray_casting.npy 와 post_ray_casting.npy 모두 필요."""
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
        if not (os.path.isdir(pre_occ_dir) and os.path.isfile(pre_rc_path)):
            continue

        for post_name in sorted(os.listdir(push_dir)):
            if not post_name.startswith('post_'):
                continue
            try:
                post_id = int(post_name.split('_')[1])
            except (ValueError, IndexError):
                continue
            if (scene_name, push_id, post_id) in excluded:
                continue

            post_dir = os.path.join(push_dir, post_name)
            post_occ_dir = os.path.join(post_dir, 'post_occ')
            post_rc_path = os.path.join(post_dir, 'post_ray_casting.npy')
            action_path = os.path.join(post_dir, 'action.npz')
            if not (os.path.isdir(post_occ_dir) and
                    os.path.isfile(post_rc_path) and
                    os.path.isfile(action_path)):
                continue

            sample = {
                'pre_occ_dir': pre_occ_dir,
                'post_occ_dir': post_occ_dir,
                'pre_ray_casting_path': pre_rc_path,
                'post_ray_casting_path': post_rc_path,
                'action_path': action_path,
                'col_idx': post_id,
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
        # GT 생성(making_push_visibility_per_view_all.py)과 일관되게 fallen + tipped 모두 제외
        fallen = parse_fallen_log(os.path.join(data_root, 'fallen_log_revised.txt'))
        tipped = parse_fallen_log(os.path.join(data_root, 'tipped_log_revised.txt'))
        excluded = fallen | tipped
        print(f'[UA] {data_root}: fallen={len(fallen)} tipped={len(tipped)} '
              f'excluded={len(excluded)}')
        scene_names = sorted(d for d in os.listdir(data_root)
                             if os.path.isdir(os.path.join(data_root, d)))
        for name in scene_names:
            is_val = name in val_scene_set
            _build_ua_samples(os.path.join(data_root, name), name, excluded, is_val,
                              train_samples, val_samples)

    train_dataset = UADataset(train_samples, num_steps=num_steps, deterministic=False)

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
        val_dataset = UADataset(val_samples, num_steps=num_steps, deterministic=True)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            **loader_kwargs)

    return train_loader, val_loader
