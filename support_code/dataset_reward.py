"""Dataset over pre-generated UA reward samples.

Expects reward_root to hold:
    {reward_root}/{occ_subdir}/{scene}/{push_name}/{idx}.npz
with npz keys: current_map (1,60,120,80) fp16, gains (98,) fp32, k int32,
               prior_cams (k,) int32.

Returns per-sample tensors:
    current : (1, 60, 120, 80) float32
    gains   : (98,)             float32  (raw squared gains)

Filtering:
    --k_min / --k_max select which samples to keep based on the stored k.

Train/val split:
    last `num_val_scenes` scenes (sorted across roots) go to validation.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler


NUM_CAMERAS = 98
GRID_ROWS, GRID_COLS = 7, 14


class RewardDataset(Dataset):
    """Returns (current, gains, seen_mask) per sample.

    seen_mask: (7, 14) float32 binary mask of cameras used as prior_cams.
               Same row-major layout as the model output (cam = row*14 + col).
    """

    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        d = np.load(self.paths[idx])
        current = torch.from_numpy(d['current_map'].astype(np.float32))
        gains = torch.from_numpy(d['gains'].astype(np.float32))
        prior_cams = d['prior_cams']
        mask = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float32)
        for c in prior_cams:
            mask[int(c) // GRID_COLS, int(c) % GRID_COLS] = 1.0
        return current, gains, torch.from_numpy(mask)


def _scan(reward_root, k_min, k_max):
    """Return list of (scene_name, path) filtered by k range."""
    out = []
    if not os.path.isdir(reward_root):
        return out
    for occ in sorted(os.listdir(reward_root)):
        occ_dir = os.path.join(reward_root, occ)
        if not os.path.isdir(occ_dir):
            continue
        for scene in sorted(os.listdir(occ_dir)):
            scene_dir = os.path.join(occ_dir, scene)
            if not os.path.isdir(scene_dir):
                continue
            for push in sorted(os.listdir(scene_dir)):
                push_dir = os.path.join(scene_dir, push)
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
                    out.append((scene, path))
    return out


def get_dataloaders(reward_roots, batch_size=16, num_workers=4,
                    k_min=2, k_max=5, num_val_scenes=10,
                    split_seed=0,
                    rank=0, world_size=1):
    """Per-sample reward dataset.

    train/val split: scenes are SHUFFLED with split_seed before picking the
    last `num_val_scenes` for validation. This is critical to avoid the
    reward train/val split coinciding with the UA train/val split, which
    would otherwise put low-gain (UA-train) scenes entirely in reward-train
    and high-gain (UA-val) scenes entirely in reward-val.
    """
    if isinstance(reward_roots, str):
        reward_roots = [reward_roots]

    all_items = []
    for root in reward_roots:
        all_items += _scan(root, k_min, k_max)

    scenes = sorted({s for s, _ in all_items})
    if num_val_scenes > 0:
        rng = np.random.RandomState(split_seed)
        shuffled = scenes.copy()
        rng.shuffle(shuffled)
        val_scenes = set(shuffled[-num_val_scenes:])
    else:
        val_scenes = set()

    train_paths = [p for s, p in all_items if s not in val_scenes]
    val_paths = [p for s, p in all_items if s in val_scenes]

    train_ds = RewardDataset(train_paths)
    val_ds = RewardDataset(val_paths)

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
