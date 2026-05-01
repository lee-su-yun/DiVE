"""Dataset over pre-generated UB reward samples.

reward_root layout (mirrors UA generator output):
    {reward_root}/{occ_subdir}/{scene}/{push_name}/{idx}.npz
npz keys: current_map (1,60,120,80) fp16, gains (29,) fp32, k int32,
          prior_cams (k,) int32, post_views (29,) int32

Returns per-sample tensors:
    current     : (1, 60, 120, 80) float32
    swept_maps  : (29, 60, 120, 80) float32   loaded from data_root on demand
    gains       : (29,)             float32

`data_roots` are the ORIGINAL beliefmap roots (e.g.
/data/APOBU/beliefmap_low_occlusion_0423). The occ_subdir under each
reward_root must match a basename of one of the data_roots so we can resolve
the absolute path for swept_maps.

Train/val split: scenes are SHUFFLED with split_seed before picking the last
`num_val_scenes` for validation (matches UA).
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler


NUM_PUSHES = 29
D, H, W = 60, 120, 80


def _load_swept_maps(scene_dir, push_name):
    """(29, D, H, W) float32 from {scene_dir}/{push_name}/post_XX/action.npz."""
    push_dir = os.path.join(scene_dir, push_name)
    arrs = []
    for XX in range(NUM_PUSHES):
        action_path = os.path.join(push_dir, f'post_{XX:02d}', 'action.npz')
        s = np.load(action_path)['swept_map'].transpose(1, 0, 2).astype(np.float32)
        arrs.append(s)
    return np.stack(arrs)


class RewardDatasetUB(Dataset):
    """Returns (current, swept_maps, gains) per sample.

    paths_with_meta: list of (sample_path, scene_dir, push_name).
    """

    def __init__(self, paths_with_meta):
        self.items = paths_with_meta

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        sample_path, scene_dir, push_name = self.items[idx]
        d = np.load(sample_path)
        current = torch.from_numpy(d['current_map'].astype(np.float32))
        gains = torch.from_numpy(d['gains'].astype(np.float32))
        swept = _load_swept_maps(scene_dir, push_name)
        swept_t = torch.from_numpy(swept)
        return current, swept_t, gains


def _scan(reward_root, data_root_map, k_min, k_max):
    """Return list of (scene_name, sample_path, scene_dir, push_name).

    data_root_map: {basename(data_root): data_root_abs_path}
    """
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
                        k = int(np.load(path)['k'])
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
                    rank=0, world_size=1):
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

    if world_size > 1:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        sampler=train_sampler, shuffle=(train_sampler is None),
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )

    val_loader = None
    if rank == 0:
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )

    return train_loader, val_loader
