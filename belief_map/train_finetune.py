"""
belief_map/train_finetune.py
============================
**FINETUNE** script for BeliefUpdateNetRecursive, the recursive semantic belief
estimator (`f_scene`). It starts from a pre-trained recursive checkpoint
(n_sem=15, with slots 12-14 already reserved) and finetunes it on a dataset that
contains new object classes.

How it differs from the from-scratch trainer (`belief_map/train.py`):
  - `--init_from <ckpt>` is required — the pretrained state_dict is loaded
    partially, before the model is wrapped for DDP. When the head shape matches
    (n_sem=15 unchanged) every tensor is copied directly.
  - `--max_scenes` defaults to 15, i.e. at most 15 episodes per data_dir.
  - `--lr` defaults to 3e-5, which keeps finetuning stable.
  - `--epochs` defaults to 20.
  - `--data_dirs` has no default; pass the dataset roots to finetune on.

Everything else — network, dataset layout, sampling conventions, loss, masking,
and the DDP setup — is the same as in the from-scratch trainer:
  - Per-view sem_obs is a n_sem-channel one-hot rather than a 1-channel scalar
    lookup.
  - Inside the rollout, sem_belief (the accumulated softmax of the previous
    step's sem_logits) is fed back as the next step's input.

n_sem = 15 (default):
  The current data only activates 0=free + 11 objects = 12 classes. Three slots
  (12, 13, 14) are reserved up front so that finetuning on additional classes
  needs no change to the architecture. Classes 12-14 simply never receive a
  positive sample; softmax and CE still behave normally.

Channel layout (n_sem=15):
  Ch 0,1     : alpha/50, beta/50   (occupancy belief, recursive)
  Ch 2       : occ                 (per-view occupancy obs)
  Ch 3       : swept_map           (push action region)
  Ch 4~18    : sem_obs one-hot     (per-view semantic obs, 15ch)
  Ch 19~33   : sem_belief          (recursive semantic belief softmax, 15ch)
  → c_in = 34, c_out = 2 + 15 = 17

The best checkpoint (by validation occupancy IoU) is written to
`{--out_dir}/{--out_name}` as a plain state_dict.

Usage (from the repo root):
    export PYTHONPATH=$PWD

    # ─── single GPU ───
    CUDA_VISIBLE_DEVICES=0 python -u -m belief_map.train_finetune \
      --init_from checkpoints/belief_map_estimator/estimator.pt \
      --data_dirs /path/to/data/beliefmap_low_occlusion_unseen \
                  /path/to/data/beliefmap_high_occlusion_unseen \
      --out_dir runs --out_name estimator_finetuned.pt \
      --batch 16 --num_workers 8 \
      --wandb --wandb_project BeliefMap-Push \
      --wandb_run finetune_recursive_unseen

    # ─── DDP (4 GPUs) ───
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --standalone \
      -m belief_map.train_finetune \
      --init_from checkpoints/belief_map_estimator/estimator.pt \
      --data_dirs /path/to/data/beliefmap_low_occlusion_unseen \
                  /path/to/data/beliefmap_high_occlusion_unseen \
      --out_dir runs --out_name estimator_finetuned.pt \
      --batch 32 --num_workers 8 \
      --n_sem 15 --w_sem 0.1
"""

import argparse
import os
import re
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

import sys
sys.path.insert(0, str(Path(__file__).parents[1]))

from belief_map.net_recursive import BeliefUpdateNetRecursive
# NOTE: `depth_semantic_to_voxel` / `recover_env_origin` are imported lazily, and
#       only on the `--no_precomputed_sem` fallback path (see __getitem__ below).

N_VIEWS   = 98
N_STEPS   = 20
VOL_SHAPE = (120, 60, 80)
K_RANGE   = 20
# Number of sem_logits / sem_belief channels in the model. The current data only
# uses 0=free + 11 objects = 12 classes; the remaining 3 slots (12, 13, 14) are
# reserved for finetuning on additional classes later on.
N_SEM_CLASSES = 15
N_CHAINS  = 5

PUSH_MIN_PRE = 3
PUSH_MIN_GAP = 4

VALID_VIEWS = list(range(N_VIEWS))

_TRANS_PAT = re.compile(r'transition_([A-E])([A-E]):\s*post_(\d+)')


# ── DDP setup ──────────────────────────────────────────────────────────────────

def _setup_ddp():
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    if local_rank == -1:
        return -1, 0, 1, False
    rank       = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size, True


def _cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def _is_ddp(net) -> bool:
    return isinstance(net, nn.parallel.DistributedDataParallel)


def _unwrap_net(net):
    if _is_ddp(net) or isinstance(net, nn.DataParallel):
        return net.module
    return net


# ── Episode meta / parsing ─────────────────────────────────────────────────────

def _parse_episode_transitions(log_path: Path):
    if not log_path.exists():
        return None
    text = log_path.read_text()
    chain_idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    out = [None] * 4
    for a, b, kk in _TRANS_PAT.findall(text):
        if chain_idx[b] - chain_idx[a] != 1:
            return None
        local_k = int(kk) - chain_idx[a] * K_RANGE
        if not (0 <= local_k < K_RANGE):
            return None
        out[chain_idx[a]] = local_k
    if any(x is None for x in out):
        return None
    return out


def _load_episode_meta(ep_dir: Path, check_semantic_occ: bool = True):
    transitions = _parse_episode_transitions(ep_dir / 'episode_log.txt')
    if transitions is None:
        return None
    sentinels = [
        ep_dir / 'push_1' / 'pre_gt.npz',
        ep_dir / 'push_1' / 'semantic_map.npz',
        ep_dir / 'push_5' / f'post_{K_RANGE - 1:02d}' / 'gt.npz',
    ]
    if check_semantic_occ:
        sentinels += [
            ep_dir / 'push_1' / 'pre_semantic_occ.npz',
            ep_dir / 'push_5' / f'post_{K_RANGE - 1:02d}' / 'post_semantic_occ.npz',
        ]
    if not all(p.exists() for p in sentinels):
        return None
    return {'ep_dir': ep_dir, 'transitions': transitions}


def _sample_push_timesteps(n_push: int,
                           n_steps: int = N_STEPS,
                           min_pre: int = PUSH_MIN_PRE,
                           min_gap: int = PUSH_MIN_GAP):
    ts = []
    lo = min_pre
    for i in range(n_push):
        hi = (n_steps - 1) - (n_push - 1 - i) * min_gap
        t = int(np.random.randint(lo, hi + 1))
        ts.append(t)
        lo = t + min_gap
    return ts


# ── Dataset ────────────────────────────────────────────────────────────────────

class MultiPushDataset(Dataset):
    def __init__(self, ep_dirs, samples_per_episode: int = 100,
                 load_workers: int = 16,
                 min_n_push: int = 2,
                 use_precomputed_sem: bool = True):
        from functools import partial
        self.min_n_push          = int(min_n_push)
        self.use_precomputed_sem = bool(use_precomputed_sem)
        print(f'Preloading episode meta for {len(ep_dirs)} episodes  '
              f'(min_n_push={self.min_n_push}, use_precomputed_sem={self.use_precomputed_sem}) ...')
        t0 = time.time()
        load_fn = partial(_load_episode_meta, check_semantic_occ=self.use_precomputed_sem)
        with ThreadPoolExecutor(max_workers=load_workers) as ex:
            metas = list(ex.map(load_fn, ep_dirs))
        self.episodes = [m for m in metas if m is not None]
        self.samples_per_episode = samples_per_episode
        print(f'  valid: {len(self.episodes)} / {len(ep_dirs)}  '
              f'({time.time()-t0:.1f}s)  →  samples/epoch={len(self.episodes) * samples_per_episode}')

    def __len__(self):
        return len(self.episodes) * self.samples_per_episode

    @staticmethod
    def _load_gt(path):
        g = np.load(path)['gt']
        return (g.transpose(1, 0, 2) > 0.5).astype(np.float32)

    @staticmethod
    def _load_semmap(path):
        return np.load(path)['semantic_map'].astype(np.int64)

    def __getitem__(self, idx):
        ep          = self.episodes[idx // self.samples_per_episode]
        ep_dir      = ep['ep_dir']
        transitions = ep['transitions']

        n_push      = int(np.random.randint(self.min_n_push, 5))
        start_chain = int(np.random.randint(0, N_CHAINS - n_push + 1))
        chain_idxs  = [start_chain + i for i in range(n_push)]
        Ks = ([transitions[c] for c in chain_idxs[:-1]]
              + [int(np.random.randint(0, K_RANGE))])
        ts = _sample_push_timesteps(n_push)

        def push_dir(c):  return ep_dir / f'push_{c + 1}'
        def post_dir(c, K): return push_dir(c) / f'post_{K:02d}'

        cache = {}

        def get_pre_3d(c):
            k = ('pre3d', c)
            if k not in cache:
                pd = push_dir(c)
                cache[k] = (self._load_gt(pd / 'pre_gt.npz'),
                            self._load_semmap(pd / 'semantic_map.npz'))
            return cache[k]

        def get_post_3d(c, K):
            k = ('post3d', c, K)
            if k not in cache:
                pd = post_dir(c, K)
                cache[k] = (self._load_gt(pd / 'gt.npz'),
                            self._load_semmap(pd / 'semantic_map.npz'),
                            np.load(pd / 'action.npz')['swept_map'].astype(np.float32))
            return cache[k]

        def get_pre_views(c):
            k = ('preV', c)
            if k not in cache:
                pd = push_dir(c)
                if self.use_precomputed_sem:
                    cache[k] = {
                        'occupied': np.load(pd / 'pre_occ.npz')['occupied'],
                        'sem_occ':  np.load(pd / 'pre_semantic_occ.npz')['labels'],
                    }
                else:
                    cache[k] = {
                        'occupied': np.load(pd / 'pre_occ.npz')['occupied'],
                        'depth':    np.load(pd / 'pre_rgbd.npz', allow_pickle=True)['depth_uint16'],
                        'sem_2d':   np.load(pd / 'pre_semantic.npz')['semantic'],
                        'cam':      np.load(pd / 'camera_poses.npz'),
                    }
            return cache[k]

        def get_post_views(c, K):
            k = ('postV', c, K)
            if k not in cache:
                pd = post_dir(c, K)
                if self.use_precomputed_sem:
                    cache[k] = {
                        'occupied': np.load(pd / 'occ.npz')['occupied'],
                        'sem_occ':  np.load(pd / 'post_semantic_occ.npz')['labels'],
                    }
                else:
                    cache[k] = {
                        'occupied': np.load(pd / 'occ.npz')['occupied'],
                        'depth':    np.load(pd / 'rgbd.npz', allow_pickle=True)['depth_uint16'],
                        'sem_2d':   np.load(pd / 'semantic.npz')['semantic'],
                        'cam':      np.load(push_dir(c) / 'camera_poses.npz'),
                    }
            return cache[k]

        views   = np.zeros((N_STEPS, *VOL_SHAPE), dtype=np.float32)
        gts     = np.zeros((N_STEPS, *VOL_SHAPE), dtype=np.float32)
        sweeps  = np.zeros((N_STEPS, *VOL_SHAPE), dtype=np.float32)
        sems    = np.zeros((N_STEPS, *VOL_SHAPE), dtype=np.float32)
        gt_sems = np.zeros((N_STEPS, *VOL_SHAPE), dtype=np.int64)

        for t in range(N_STEPS):
            n_done = sum(1 for tj in ts if tj <= t)
            if n_done == 0:
                c_local, mode, is_push = 0, 'pre', False
            elif ts[n_done - 1] == t:
                c_local, mode, is_push = n_done - 1, 'post', True
            elif n_done < n_push:
                c_local, mode, is_push = n_done, 'pre', False
            else:
                c_local, mode, is_push = n_push - 1, 'post', False

            c, K = chain_idxs[c_local], Ks[c_local]

            if mode == 'pre':
                gt_3d, sem_3d_gt = get_pre_3d(c)
                vmeta = get_pre_views(c)
            else:
                gt_3d, sem_3d_gt, swept = get_post_3d(c, K)
                vmeta = get_post_views(c, K)
                if is_push:
                    sweeps[t] = swept

            gts[t]     = gt_3d
            gt_sems[t] = sem_3d_gt

            v_idx   = VALID_VIEWS[np.random.randint(0, len(VALID_VIEWS))]
            occ_hwd = vmeta['occupied'][v_idx].astype(np.float32)
            views[t] = occ_hwd.transpose(1, 0, 2)

            if self.use_precomputed_sem:
                sem_hwd  = vmeta['sem_occ'][v_idx].astype(np.float32)
                sems [t] = sem_hwd.transpose(1, 0, 2)
            else:
                # On-the-fly back-projection (--no_precomputed_sem). Needs the per-view
                # depth and 2D semantics, which the Hub archives leave out, so this
                # path only works on data you collected yourself.
                from inference.utils.voxel_projection import depth_semantic_to_voxel
                from belief_map.env_origin import recover_env_origin
                depth_v = vmeta['depth'][v_idx].astype(np.float32) / 1000.0
                depth_v[depth_v == 0.0] = np.nan
                sem_2d  = vmeta['sem_2d'][v_idx]
                cam     = vmeta['cam']
                pos     = cam['positions'].astype(np.float32)
                quats   = cam['orientations'].astype(np.float32)
                env_o   = recover_env_origin(pos[0])
                sem_voxel = depth_semantic_to_voxel(
                    np.nan_to_num(depth_v, nan=0.0), sem_2d,
                    pos[v_idx], quats[v_idx], env_o,
                )
                sems[t] = sem_voxel.astype(np.float32)

        return (
            torch.from_numpy(views),
            torch.from_numpy(sweeps),
            torch.from_numpy(gts),
            torch.from_numpy(sems),
            torch.from_numpy(gt_sems),
            int(ts[0]),
        )


# ── Metrics ────────────────────────────────────────────────────────────────────

def iou_from_prob(prob: torch.Tensor, gt: torch.Tensor, thr: float = 0.5) -> float:
    pred  = (prob > thr)
    gt_b  = gt.bool()
    inter = (pred & gt_b).sum().item()
    union = (pred | gt_b).sum().item()
    return inter / union if union > 0 else 0.0


def sem_acc_on_occupied(sem_logits: torch.Tensor, gt_sem: torch.Tensor) -> float:
    pred = sem_logits.argmax(dim=1)
    mask = gt_sem > 0
    if not mask.any():
        return 0.0
    return (pred[mask] == gt_sem[mask]).float().mean().item()


def sem_miou(sem_logits: torch.Tensor, gt_sem: torch.Tensor,
             n_sem: int, exclude_bg: bool = True) -> float:
    """
    Mean IoU over semantic classes.
      With exclude_bg=True, class 0 (free) is dropped and the mean is taken over
      the object classes only. A class that never appears in the GT (e.g. the
      reserved slots 12-14) has union=0 and is skipped.
    """
    pred = sem_logits.argmax(dim=1)            # (B, W, H, D)
    start = 1 if exclude_bg else 0
    ious  = []
    for c in range(start, n_sem):
        gt_c  = (gt_sem == c)
        pd_c  = (pred   == c)
        inter = (gt_c & pd_c).sum().item()
        union = gt_c.sum().item() + pd_c.sum().item() - inter
        if union > 0:
            ious.append(inter / union)
    return sum(ious) / len(ious) if ious else 0.0


def _update_seen_and_mask(seen_classes: torch.Tensor,
                          sem_t: torch.Tensor,
                          gt_sem_t: torch.Tensor,
                          n_sem_classes: int) -> torch.Tensor:
    B = sem_t.shape[0]
    cur_flat = sem_t.long().reshape(B, -1).clamp(0, n_sem_classes - 1)
    seen_classes.scatter_(1, cur_flat, torch.ones_like(cur_flat, dtype=torch.bool))
    seen_classes[:, 0] = False
    gt_flat = gt_sem_t.reshape(B, -1).clamp(0, n_sem_classes - 1)
    mask_obj = seen_classes.gather(1, gt_flat).view_as(gt_sem_t)
    return ((gt_sem_t == 0) | mask_obj).float()


# ── Pretrained partial load ─────────────────────────────────────────────────────

def _load_pretrained_partial(net: nn.Module, ckpt_path: Path,
                             device: str, is_main: bool = True):
    """
    Partially load a pretrained state_dict.
      - Same shape: copied as is.
      - Only dim 0 differs (the head grew in out_ch): copy the leading channels
        and leave the new ones at their random init.
      - Shape mismatch / missing key: skipped.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt and not any(
            k.startswith(('unet.', 'class_scalar')) for k in ckpt.keys()):
        ckpt = ckpt['state_dict']
    own_sd = net.state_dict()

    direct, partial, skipped = [], [], []
    with torch.no_grad():
        for k, v in ckpt.items():
            if k not in own_sd:
                skipped.append((k, tuple(v.shape), 'not in new model'))
                continue
            t = own_sd[k]
            if t.shape == v.shape:
                t.copy_(v)
                direct.append(k)
            elif t.dim() == v.dim() and t.shape[1:] == v.shape[1:] and t.shape[0] >= v.shape[0]:
                n_copy = v.shape[0]
                t[:n_copy].copy_(v)
                partial.append((k, tuple(v.shape), tuple(t.shape), n_copy))
            else:
                skipped.append((k, tuple(v.shape), f'incompatible with {tuple(t.shape)}'))

    if is_main:
        print(f'[finetune] loaded pretrained from {ckpt_path}')
        print(f'  direct copy : {len(direct)} tensors')
        for k, oshape, nshape, n in partial:
            print(f'  partial copy: {k}  {oshape} → {nshape}  (copied first {n} of dim 0)')
        for k, shape, reason in skipped:
            print(f'  SKIPPED     : {k}  shape={shape}  ({reason})')
        own_keys = set(own_sd.keys())
        ckpt_keys = set(ckpt.keys())
        missing = own_keys - ckpt_keys
        if missing:
            print(f'  new (random init): {sorted(missing)}')
    return direct, partial, skipped


# ── Train / Val epoch ────────────────────────────────────────────────────────────

def _gpu_mem(device):
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        used = torch.cuda.memory_allocated(idx) / 1024**2
        resv = torch.cuda.memory_reserved(idx)  / 1024**2
        return f'GPU{idx} {used:.0f}/{resv:.0f}MB'
    return ''

def run_epoch(net, loader, optimizer, device, train: bool,
              epoch: int = 0, log_interval: int = 5,
              use_wandb: bool = False, global_step: int = 0,
              w_sem: float = 1.0,
              rank: int = 0, world_size: int = 1):
    net.train(train)
    total_loss     = 0.0
    total_occ_loss = 0.0
    total_sem_loss = 0.0
    total_occ_iou  = 0.0
    total_sem_iou  = 0.0
    total_sem_acc  = 0.0
    n              = 0
    n_batches      = len(loader)
    phase          = 'train' if train else 'val'
    n_sem          = _unwrap_net(net).n_sem
    is_main        = (rank == 0)

    for batch_idx, (views, sweeps, gts, sems, gt_sems, push_steps) in enumerate(loader):
        B = views.shape[0]
        views   = views  .to(device)
        sweeps  = sweeps .to(device)
        gts     = gts    .to(device)
        sems    = sems   .to(device)
        gt_sems = gt_sems.to(device)

        alpha = torch.ones(B, *VOL_SHAPE, device=device)
        beta  = torch.ones(B, *VOL_SHAPE, device=device)
        # Recursive semantic belief — initialized with a uniform prior (1/n_sem)
        sem_belief = torch.full(
            (B, n_sem, *VOL_SHAPE),
            1.0 / n_sem,
            device=device,
        )
        seen_classes = torch.zeros(B, N_SEM_CLASSES, dtype=torch.bool, device=device)
        step_occ_loss = 0.0
        step_sem_loss = 0.0

        for t in range(N_STEPS):
            occ_t    = views  [:, t]
            sweep_t  = sweeps [:, t]
            gt_t     = gts    [:, t]
            sem_t    = sems   [:, t]
            gt_sem_t = gt_sems[:, t]

            mask_f = _update_seen_and_mask(seen_classes, sem_t, gt_sem_t, N_SEM_CLASSES)
            denom  = mask_f.sum().clamp(min=1.0)

            if train:
                alpha, beta, sem_logits = net(alpha, beta, occ_t, sweep_t, sem_t, sem_belief)
                prob  = alpha / (alpha + beta)
                l_occ = ((prob - gt_t) ** 2 * mask_f).sum() / denom
                ce    = F.cross_entropy(sem_logits, gt_sem_t, reduction='none')
                l_sem = (ce * mask_f).sum() / denom
                loss  = l_occ + w_sem * l_sem
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()
                alpha = alpha.detach()
                beta  = beta .detach()
                sem_belief = F.softmax(sem_logits, dim=1).detach()
                step_occ_loss += l_occ.item()
                step_sem_loss += l_sem.item()
            else:
                with torch.no_grad():
                    alpha, beta, sem_logits = net(alpha, beta, occ_t, sweep_t, sem_t, sem_belief)
                    prob  = alpha / (alpha + beta)
                    l_occ = ((prob - gt_t) ** 2 * mask_f).sum() / denom
                    ce    = F.cross_entropy(sem_logits, gt_sem_t, reduction='none')
                    l_sem = (ce * mask_f).sum() / denom
                    sem_belief = F.softmax(sem_logits, dim=1)
                    step_occ_loss += l_occ.item()
                    step_sem_loss += l_sem.item()

        with torch.no_grad():
            prob = alpha / (alpha + beta)
            occ_iou = iou_from_prob(prob, gt_t)
            sem_acc = sem_acc_on_occupied(sem_logits, gt_sem_t)
            sem_iou = sem_miou(sem_logits, gt_sem_t, n_sem=n_sem, exclude_bg=True)

        batch_occ = step_occ_loss / N_STEPS
        batch_sem = step_sem_loss / N_STEPS
        batch_loss = batch_occ + w_sem * batch_sem

        total_loss     += batch_loss * B
        total_occ_loss += batch_occ  * B
        total_sem_loss += batch_sem  * B
        total_occ_iou  += occ_iou * B
        total_sem_iou  += sem_iou * B
        total_sem_acc  += sem_acc * B
        n              += B

        if is_main and (batch_idx % 10 == 0 or batch_idx == n_batches - 1):
            print(f'  [{phase}] e{epoch} b{batch_idx}/{n_batches}  '
                  f'loss={batch_loss:.5f} (occ={batch_occ:.5f} sem={batch_sem:.5f})  '
                  f'occ_iou={occ_iou:.4f}  sem_iou={sem_iou:.4f}  sem_acc={sem_acc:.4f}  '
                  f'{_gpu_mem(device)}', flush=True)

        if is_main and use_wandb and train:
            wandb.log({
                'train/batch_loss':     batch_loss,
                'train/batch_occ_loss': batch_occ,
                'train/batch_sem_loss': batch_sem,
                'train/batch_occ_iou':  occ_iou,
                'train/batch_sem_iou':  sem_iou,
                'train/batch_sem_acc':  sem_acc,
            }, step=global_step + batch_idx)

    if world_size > 1 and dist.is_initialized():
        agg = torch.tensor([total_loss, total_occ_loss, total_sem_loss,
                            total_occ_iou, total_sem_iou, total_sem_acc, float(n)],
                           device=device)
        dist.all_reduce(agg, op=dist.ReduceOp.SUM)
        total_loss, total_occ_loss, total_sem_loss, \
            total_occ_iou, total_sem_iou, total_sem_acc, n_f = agg.tolist()
        n = max(int(n_f), 1)

    return (total_loss / n, total_occ_loss / n, total_sem_loss / n,
            total_occ_iou / n, total_sem_iou / n, total_sem_acc / n)


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dirs', nargs='+',
                        default=[],
                        help='Dataset roots to finetune on (one or more '
                             'directories of episodes). No default.')
    parser.add_argument('--init_from',     required=True, type=str,
                        help='Pretrained ckpt (recursive_n15_onehot). '
                             'The state_dict is loaded partially; if the head shape '
                             'matches, everything is copied.')
    parser.add_argument('--out_dir',       default='checkpoints')
    parser.add_argument('--out_name',      default='belief_update_finetune_unseen_v1_recursive.pt')
    parser.add_argument('--epochs',        type=int,   default=20)
    parser.add_argument('--lr',            type=float, default=3e-5)
    parser.add_argument('--batch',         type=int,   default=16)
    parser.add_argument('--num_workers',   type=int,   default=8)
    parser.add_argument('--device',        default='cuda')
    parser.add_argument('--base_ch',       type=int,   default=32)
    parser.add_argument('--val_ratio',     type=float, default=0.1)
    parser.add_argument('--max_scenes',    type=int,   default=15,
                        help='per data_dir episode cap (finetune default 15)')
    parser.add_argument('--n_sem',         type=int,   default=N_SEM_CLASSES,
                        help=f'Number of semantic output classes (default {N_SEM_CLASSES}; '
                             'the recursive model requires > 0). The current data uses only '
                             '12 classes, but the default is 15 to leave room for classes '
                             'added during finetuning.')
    parser.add_argument('--samples_per_episode', type=int, default=100)
    parser.add_argument('--w_sem',         type=float, default=1.0)
    parser.add_argument('--min_n_push',    type=int, default=2)
    parser.add_argument('--use_precomputed_sem', dest='use_precomputed_sem',
                        action='store_true', default=True)
    parser.add_argument('--no_precomputed_sem', dest='use_precomputed_sem',
                        action='store_false')
    parser.add_argument('--wandb',         action='store_true')
    parser.add_argument('--wandb_project', default='BeliefMap-Push')
    parser.add_argument('--wandb_run',     default=None)
    args = parser.parse_args()

    assert args.n_sem > 0, 'the recursive model requires --n_sem > 0'

    local_rank, rank, world_size, is_ddp = _setup_ddp()
    is_main = (rank == 0)

    use_wandb = args.wandb and WANDB_AVAILABLE
    if args.wandb and not WANDB_AVAILABLE and is_main:
        print('[WARNING] wandb not installed. Run: pip install wandb')
    if is_main and use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run,
            config={
                'data_dirs':     args.data_dirs,
                'epochs':        args.epochs,
                'lr':            args.lr,
                'batch':         args.batch,
                'effective_batch': args.batch * world_size,
                'base_ch':       args.base_ch,
                'val_ratio':     args.val_ratio,
                'num_workers':   args.num_workers,
                'n_steps':       N_STEPS,
                'k_range':       K_RANGE,
                'with_semantic': True,
                'semantic_recursive': True,
                'sem_belief_init':    'uniform (1/n_sem)',
                'sem_belief_feedback': 'softmax(sem_logits).detach()',
                'sem_encoding':  f'one_hot (n_sem={args.n_sem})',
                'sem_source':    'precomputed semantic_occ' if args.use_precomputed_sem
                                 else 'on-the-fly depth_semantic_to_voxel',
                'min_n_push':    args.min_n_push,
                'use_precomputed_sem': args.use_precomputed_sem,
                'parallel':      'DDP' if is_ddp else
                                 ('DataParallel' if torch.cuda.device_count() > 1 else 'single'),
                'world_size':    world_size,
                'n_sem':         args.n_sem,
                'w_sem':         args.w_sem,
            },
        )

    if is_ddp:
        device = f'cuda:{local_rank}'
    else:
        device = args.device
    out_dir = Path(args.out_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)

    all_ep_dirs = []
    for data_dir in args.data_dirs:
        ep_dirs = sorted([p for p in Path(data_dir).iterdir() if p.is_dir()])
        if args.max_scenes is not None:
            ep_dirs = ep_dirs[:args.max_scenes]
        all_ep_dirs.extend(ep_dirs)

    rng     = np.random.default_rng(42)
    idx     = rng.permutation(len(all_ep_dirs))
    n_val   = max(1, round(len(all_ep_dirs) * args.val_ratio))
    val_set = set(idx[:n_val].tolist())
    all_train_dirs = [all_ep_dirs[i] for i in range(len(all_ep_dirs)) if i not in val_set]
    all_val_dirs   = [all_ep_dirs[i] for i in range(len(all_ep_dirs)) if i     in val_set]
    if is_main:
        print(f'Total episodes: {len(all_ep_dirs)}  →  train={len(all_train_dirs)}, val={len(all_val_dirs)}')
        print(f'Parallel mode: {"DDP" if is_ddp else ("DP" if torch.cuda.device_count() > 1 else "single")}, '
              f'world_size={world_size}, rank={rank}, local_rank={local_rank}, device={device}')

    train_ds = MultiPushDataset(
        all_train_dirs, samples_per_episode=args.samples_per_episode,
        min_n_push=args.min_n_push, use_precomputed_sem=args.use_precomputed_sem,
    )
    val_ds   = MultiPushDataset(
        all_val_dirs, samples_per_episode=args.samples_per_episode,
        min_n_push=args.min_n_push, use_precomputed_sem=args.use_precomputed_sem,
    )
    if is_main:
        print(f'Train samples: {len(train_ds)}  Val samples: {len(val_ds)}')

    if is_ddp:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size,
                                           rank=rank, shuffle=True, drop_last=False)
        val_sampler   = DistributedSampler(val_ds,   num_replicas=world_size,
                                           rank=rank, shuffle=False, drop_last=False)
    else:
        train_sampler = None
        val_sampler   = None

    train_ld = DataLoader(train_ds, batch_size=args.batch,
                          sampler=train_sampler,
                          shuffle=(train_sampler is None),
                          num_workers=args.num_workers, pin_memory=True,
                          persistent_workers=(args.num_workers > 0))
    val_ld   = DataLoader(val_ds,   batch_size=args.batch,
                          sampler=val_sampler, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True,
                          persistent_workers=(args.num_workers > 0))

    net = BeliefUpdateNetRecursive(base_ch=args.base_ch, with_sweep=True,
                                   with_semantic=True, n_sem=args.n_sem).to(device)
    if is_main:
        c_in_total = 4 + args.n_sem + args.n_sem   # a,b,occ,sweep + sem_obs one-hot + sem_belief
        print(f'Finetuning (BeliefUpdateNetRecursive, c_in={c_in_total} '
              f'[4 + sem_obs one-hot {args.n_sem} + sem_belief {args.n_sem}], '
              f'n_sem={args.n_sem})')

    # ── Pretrained partial load (before the DDP wrap) ────────────────────────
    init_path = Path(args.init_from)
    if not init_path.exists():
        raise FileNotFoundError(f'--init_from not found: {init_path}')
    _load_pretrained_partial(net, init_path, device, is_main=is_main)

    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    if is_main:
        print(f'Network params: {n_params/1e6:.2f}M')

    if is_ddp:
        if is_main:
            print(f'Using DDP across {world_size} ranks (local_rank={local_rank}).')
        net = nn.parallel.DistributedDataParallel(net, device_ids=[local_rank],
                                                  output_device=local_rank,
                                                  find_unused_parameters=False)
    elif device.startswith('cuda') and torch.cuda.device_count() > 1:
        if is_main:
            print(f'Using DataParallel across {torch.cuda.device_count()} GPUs (fallback).')
        net = nn.DataParallel(net)

    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_occ_iou = 0.0
    best_path        = out_dir / args.out_name
    n_train_batches  = len(train_ld)

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        global_step = (epoch - 1) * n_train_batches
        tr_loss, tr_occ, tr_sem, tr_occ_iou, tr_sem_iou, tr_sem_acc = run_epoch(
            net, train_ld, optimizer, device, train=True,
            epoch=epoch, use_wandb=use_wandb, global_step=global_step,
            w_sem=args.w_sem, rank=rank, world_size=world_size)
        vl_loss, vl_occ, vl_sem, vl_occ_iou, vl_sem_iou, vl_sem_acc = run_epoch(
            net, val_ld, optimizer, device, train=False,
            epoch=epoch, use_wandb=use_wandb, w_sem=args.w_sem,
            rank=rank, world_size=world_size)
        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        flag = ''
        if is_main and vl_occ_iou > best_val_occ_iou:
            best_val_occ_iou = vl_occ_iou
            state_dict = _unwrap_net(net).state_dict()
            torch.save(state_dict, best_path)
            flag = '  ← best'

        if is_main and use_wandb:
            wandb.log({
                'train/epoch_loss':     tr_loss,
                'train/epoch_occ_loss': tr_occ,
                'train/epoch_sem_loss': tr_sem,
                'train/epoch_occ_iou':  tr_occ_iou,
                'train/epoch_sem_iou':  tr_sem_iou,
                'train/epoch_sem_acc':  tr_sem_acc,
                'val/epoch_loss':       vl_loss,
                'val/epoch_occ_loss':   vl_occ,
                'val/epoch_sem_loss':   vl_sem,
                'val/epoch_occ_iou':    vl_occ_iou,
                'val/epoch_sem_iou':    vl_sem_iou,
                'val/epoch_sem_acc':    vl_sem_acc,
                'lr':                   lr,
                'best_val_occ_iou':     best_val_occ_iou,
                'epoch':                epoch,
            }, step=global_step + n_train_batches)

        if is_main:
            print(f'Ep {epoch:4d}/{args.epochs} | '
                  f'train loss={tr_loss:.5f} (occ={tr_occ:.5f} sem={tr_sem:.5f}) '
                  f'occ_iou={tr_occ_iou:.4f} sem_iou={tr_sem_iou:.4f} sem_acc={tr_sem_acc:.4f} | '
                  f'val loss={vl_loss:.5f} (occ={vl_occ:.5f} sem={vl_sem:.5f}) '
                  f'occ_iou={vl_occ_iou:.4f} sem_iou={vl_sem_iou:.4f} sem_acc={vl_sem_acc:.4f} | '
                  f'lr={lr:.2e}{flag}', flush=True)

    if is_main:
        print(f'\nBest val occ IoU: {best_val_occ_iou:.4f}  →  {best_path}')
        if use_wandb:
            wandb.summary['best_val_occ_iou'] = best_val_occ_iou
            wandb.finish()

    _cleanup_ddp()


if __name__ == '__main__':
    main()
