import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler


PUSH_STEP_MIN = 2
PUSH_STEP_MAX = 8
NUM_VIEWS = 98
GROUP_STRIDE = 14  # 98 = 14 columns * 7 rows
MAX_GROUP_ID = 13  # col_idx 28 is merged into group 13


def _col_idx_to_group(col_idx):
    return min(col_idx // 2, MAX_GROUP_ID)


def _group_view_pool(group_id):
    """Cameras {group_id + 14*k} within [0, 98)."""
    return np.array(
        [group_id + GROUP_STRIDE * k for k in range(7)
         if group_id + GROUP_STRIDE * k < NUM_VIEWS],
        dtype=np.int64,
    )


def parse_fallen_log(path):
    """Parse fallen_log_revised.txt lines of the form:
        episode=000000015  push_1  post_03  fallen=['obj_10']
    Returns set of (scene_name, push_id, post_id).
    """
    bad = set()
    if not os.path.isfile(path):
        return bad
    pat = re.compile(r'episode=(\S+)\s+push_(\d+)\s+post_(\d+)')
    with open(path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                bad.add((m.group(1), int(m.group(2)), int(m.group(3))))
    return bad


def parse_episode_log(log_path):
    """Return list of {'col_idx': int, 'pre_push': int, 'post_push': int} for AB/BC transitions."""
    transitions = []
    with open(log_path) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        m = re.match(r'transition_(\w+)', lines[i].strip())
        if m:
            name = m.group(1)
            i += 1
            if i >= len(lines):
                break
            col_m = re.search(r'col_idx\s*:\s*(\d+)', lines[i].strip())
            if col_m:
                col_idx = int(col_m.group(1))
                if name == 'AB':
                    transitions.append({'col_idx': col_idx, 'pre_push': 1, 'post_push': 2})
                elif name == 'BC':
                    transitions.append({'col_idx': col_idx, 'pre_push': 2, 'post_push': 3})
        i += 1
    return transitions


def _worker_init_fn(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed + worker_id)


def _load_views(occ_dir, view_indices):
    arrs = [np.load(os.path.join(occ_dir, f'{int(v):03d}.npz'))['occ']
            for v in view_indices]
    return np.stack(arrs).astype(np.float32)


def _sample_view_schedule(col_idx, num_steps, rng):
    """Draw (push_step, pre_views, post_views) for one sample.

    pre_views : push_step distinct views from the full 98-view pool.
    post_views: first view drawn from the group pool for this col_idx,
                remaining (num_post - 1) from the full pool excluding the first.
    """
    push_step = int(rng.integers(PUSH_STEP_MIN, PUSH_STEP_MAX + 1)) \
        if hasattr(rng, 'integers') else int(rng.randint(PUSH_STEP_MIN, PUSH_STEP_MAX + 1))
    num_post = num_steps - push_step

    pre_views = np.sort(rng.choice(NUM_VIEWS, push_step, replace=False)).astype(np.int64)

    pool = _group_view_pool(_col_idx_to_group(col_idx))
    first_post = int(rng.choice(pool))

    if num_post - 1 > 0:
        remaining = np.setdiff1d(np.arange(NUM_VIEWS, dtype=np.int64), [first_post])
        other = rng.choice(remaining, num_post - 1, replace=False)
        post_views = np.concatenate([[first_post], np.sort(other)]).astype(np.int64)
    else:
        post_views = np.array([first_post], dtype=np.int64)

    return push_step, pre_views, post_views


class UABDataset(Dataset):
    """Unified dataset: each sample is a dict with
        pre_occ_dir, post_occ_dir, gt_pre_path, gt_post_path, action_path, col_idx.
    Returns (voxel_maps, gt_pre, gt_post, swept_maps) of shapes
        (num_steps,60,120,80), (60,120,80), (60,120,80), (num_steps,60,120,80).
    """

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

        pre_voxels = _load_views(s['pre_occ_dir'], pre_views)
        post_voxels = _load_views(s['post_occ_dir'], post_views)
        voxels = np.concatenate([pre_voxels, post_voxels], axis=0)

        gt_pre = np.load(s['gt_pre_path']).astype(np.float32)
        gt_post = np.load(s['gt_post_path']).astype(np.float32)

        swept_map = np.load(s['action_path'])['swept_map']
        swept_map = swept_map.transpose(1, 0, 2).astype(np.float32)

        swept_maps = np.zeros((self.num_steps, 60, 120, 80), dtype=np.float32)
        swept_maps[push_step] = swept_map

        return (
            torch.from_numpy(voxels),
            torch.from_numpy(gt_pre),
            torch.from_numpy(gt_post),
            torch.from_numpy(swept_maps),
        )


def _build_ua_samples(scene_dir, scene_name, fallen, is_val, train_out, val_out):
    """UA: each (push_n, post_XX) pair is a sample.
       col_idx = post_XX index. GT is per-post visibility.
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
        gt_pre_path = os.path.join(push_dir, 'pre_visibility.npy')
        if not (os.path.isdir(pre_occ_dir) and os.path.isfile(gt_pre_path)):
            continue

        for post_name in sorted(os.listdir(push_dir)):
            if not post_name.startswith('post_'):
                continue
            try:
                post_id = int(post_name.split('_')[1])
            except (ValueError, IndexError):
                continue
            if (scene_name, push_id, post_id) in fallen:
                continue

            post_dir = os.path.join(push_dir, post_name)
            post_occ_dir = os.path.join(post_dir, 'post_occ')
            gt_post_path = os.path.join(post_dir, 'post_visibility.npy')
            action_path = os.path.join(post_dir, 'action.npz')
            if not (os.path.isdir(post_occ_dir) and
                    os.path.isfile(gt_post_path) and
                    os.path.isfile(action_path)):
                continue

            sample = {
                'pre_occ_dir': pre_occ_dir,
                'post_occ_dir': post_occ_dir,
                'gt_pre_path': gt_pre_path,
                'gt_post_path': gt_post_path,
                'action_path': action_path,
                'col_idx': post_id,
            }
            (val_out if is_val else train_out).append(sample)


def _build_ub_samples(scene_dir, scene_name, fallen, is_val, train_out, val_out):
    """UB: each AB/BC transition is a sample.
       Pre/post occ both come from push_{pre or post}/pre_occ/;
       swept_map comes from push_{pre}/post_{col_idx}/action.npz;
       GT is push_{pre or post}/push_visibility.npy.
    """
    log_path = os.path.join(scene_dir, 'episode_log.txt')
    if not os.path.isfile(log_path):
        return

    for t in parse_episode_log(log_path):
        col_idx = t['col_idx']
        pre_push = t['pre_push']
        post_push = t['post_push']

        if (scene_name, pre_push, col_idx) in fallen:
            continue

        pre_push_dir = os.path.join(scene_dir, f'push_{pre_push}')
        post_push_dir = os.path.join(scene_dir, f'push_{post_push}')

        pre_occ_dir = os.path.join(pre_push_dir, 'pre_occ')
        post_occ_dir = os.path.join(post_push_dir, 'pre_occ')
        gt_pre_path = os.path.join(pre_push_dir, 'push_visibility_all.npy')
        gt_post_path = os.path.join(post_push_dir, 'push_visibility_all.npy')
        action_path = os.path.join(pre_push_dir, f'post_{col_idx:02d}', 'action.npz')

        if not (os.path.isdir(pre_occ_dir) and os.path.isdir(post_occ_dir) and
                os.path.isfile(gt_pre_path) and os.path.isfile(gt_post_path) and
                os.path.isfile(action_path)):
            continue

        sample = {
            'pre_occ_dir': pre_occ_dir,
            'post_occ_dir': post_occ_dir,
            'gt_pre_path': gt_pre_path,
            'gt_post_path': gt_post_path,
            'action_path': action_path,
            'col_idx': col_idx,
        }
        (val_out if is_val else train_out).append(sample)


def get_dataloaders(data_roots, task, batch_size=8, num_steps=10,
                    num_val_scenes=10, num_workers=4, rank=0, world_size=1):
    task = task.upper()
    assert task in ('UA', 'UB'), f'Unknown task: {task}'
    build_fn = _build_ua_samples if task == 'UA' else _build_ub_samples

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
        scene_names = sorted(d for d in os.listdir(data_root)
                             if os.path.isdir(os.path.join(data_root, d)))
        for name in scene_names:
            is_val = name in val_scene_set
            build_fn(os.path.join(data_root, name), name, fallen, is_val,
                     train_samples, val_samples)

    train_dataset = UABDataset(train_samples, num_steps=num_steps, deterministic=False)

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
        val_dataset = UABDataset(val_samples, num_steps=num_steps, deterministic=True)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            **loader_kwargs)

    return train_loader, val_loader
