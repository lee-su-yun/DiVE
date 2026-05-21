"""Train UA reward model on (current_map -> gains[98]) samples.

Two training objectives supported via --loss_type:

1) loss_type=kl (rank-aware soft distribution)
   target = softmax( normalize(gains) / T )
   loss   = KL( target || softmax(logit) )

2) loss_type=bce_topk (HARD top-K vs bottom — recommended for "good vs bad
   group" separation, NOT exact within-group ordering)
   target = 1 for top-K cams (per sample), 0 otherwise
   loss   = BCEWithLogits(logit, target)

3) loss_type=bce_sigmoid_z (SOFT top vs bottom)
   target = sigmoid((g - g.mean) / g.std / T)
   loss   = BCEWithLogits(logit, target)
   per-sample mean is the implicit threshold; T controls steepness.

Per-sample normalization (--target_norm) is applied only to the kl variant.

Metrics (logged each epoch):
    val_top1, val_top3      : argmax(gains) hit rate under logits
    val_p_at_10             : |topk(logit,10) ∩ topk(gains,10)| / 10
    val_p_at_20             : same with K=20
    val_pred_entropy        : softmax-based; uniform = ln(98) ≈ 4.585.
                              Near 0 with constant argmax => model collapse.

명령어 :
    cd /home/sylee/codes/DiVE/4_policy
    python train_reward_UA.py \
        --reward_roots /data/APOBU/U_AB_beliefmap/ua_Kt_reward_dataset \
        --save_dir /result/APOBU/U_AB_beliefmap/reward_Kt_v1 \
        --wandb_run_name reward_Kt_v1_sigz \
        --loss_type bce_sigmoid_z --temperature 1.0 \
        --device 0,1,2,3
"""
import os
import sys
import argparse
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import wandb
from tqdm import tqdm

# Self-contained: dataset / model live alongside this script in 4_policy_unified/.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_reward import get_dataloaders  # noqa: E402
from model_reward import RewardNet  # noqa: E402

os.environ['PYTHONUNBUFFERED'] = '1'

# Avoid fd-strategy leak that exhausted shm after many epochs (errno=0 on shm_open).
mp.set_sharing_strategy('file_system')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--reward_roots', type=str, nargs='+', required=True)
    p.add_argument('--save_dir', type=str, required=True)
    p.add_argument('--wandb_run_name', type=str, required=True)
    p.add_argument('--wandb_project', type=str, default='U_AB_reward')

    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--patience', type=int, default=8)
    p.add_argument('--num_val_scenes', type=int, default=10)
    p.add_argument('--num_workers', type=int, default=8)

    p.add_argument('--k_min', type=int, default=2)
    p.add_argument('--k_max', type=int, default=5)
    p.add_argument('--split_seed', type=int, default=0,
                   help='Seed for the random scene shuffle that separates '
                        'reward train/val. Different seed = different val.')

    p.add_argument('--loss_type', type=str, default='bce_sigmoid_z',
                   choices=['kl', 'bce_topk', 'bce_sigmoid_z'],
                   help='Training objective. Default bce_sigmoid_z gives a '
                        'soft top/bottom grouping with relaxed within-group '
                        'ordering — best when you want "top group recognized" '
                        'without strict ranking inside it.')
    p.add_argument('--top_k', type=int, default=20,
                   help='K for bce_topk loss and for Precision@K metric.')
    p.add_argument('--subtract_cam_mean', action='store_true', default=False,
                   help='Subtract dataset-wide per-cam mean from gains BEFORE '
                        'forming the target. Removes the global cam prior so '
                        'the model must use input to score above-baseline cams '
                        'for THIS sample.')

    p.add_argument('--temperature', type=float, default=1.0,
                   help='Used by kl (softmax temperature) and bce_sigmoid_z '
                        '(controls sigmoid steepness).')
    p.add_argument('--target_norm', type=str, default='zscore',
                   choices=['none', 'zscore', 'minmax', 'max'],
                   help='Per-sample normalization for loss_type=kl only.')
    p.add_argument('--log1p_target', action='store_true', default=False,
                   help='Apply log1p to gains BEFORE normalization (kl only).')
    p.add_argument('--no_log1p_target', dest='log1p_target', action='store_false')

    p.add_argument('--base_ch', type=int, default=16)
    p.add_argument('--use_seen_mask', action='store_true', default=False,
                   help='Concatenate the (7,14) binary mask of prior_cams as '
                        'an extra channel before the 2D head. Tells the model '
                        'explicitly which cameras have already been observed.')
    p.add_argument('--device', type=str, default='0')
    p.add_argument('--ddp_port', type=int, default=29300)
    return p.parse_args()


def _normalize(g, kind, log1p):
    if log1p:
        g = torch.log1p(g)
    if kind == 'zscore':
        m = g.mean(dim=-1, keepdim=True)
        s = g.std(dim=-1, keepdim=True) + 1e-6
        return (g - m) / s
    if kind == 'minmax':
        mn = g.min(dim=-1, keepdim=True).values
        mx = g.max(dim=-1, keepdim=True).values
        return (g - mn) / (mx - mn + 1e-6)
    if kind == 'max':
        return g / (g.max(dim=-1, keepdim=True).values + 1e-6)
    return g  # 'none'


def compute_loss(logit, gains, args, cam_mean=None):
    """Returns (loss, target_for_inspection).

    If cam_mean is provided (and --subtract_cam_mean), the per-cam dataset mean
    is subtracted from gains FIRST. This removes the global per-cam prior and
    forces the model to use the input to score above-baseline cams.
    """
    if cam_mean is not None:
        gains = gains - cam_mean
    if args.loss_type == 'kl':
        g = _normalize(gains, args.target_norm, args.log1p_target)
        tgt = F.softmax(g / args.temperature, dim=-1)
        log_pred = F.log_softmax(logit, dim=-1)
        return F.kl_div(log_pred, tgt, reduction='batchmean'), tgt
    if args.loss_type == 'bce_topk':
        idx = gains.topk(args.top_k, dim=-1).indices
        tgt = torch.zeros_like(gains)
        tgt.scatter_(1, idx, 1.0)
        return F.binary_cross_entropy_with_logits(logit, tgt), tgt
    if args.loss_type == 'bce_sigmoid_z':
        g = _normalize(gains, 'zscore', False)
        tgt = torch.sigmoid(g / args.temperature)
        return F.binary_cross_entropy_with_logits(logit, tgt), tgt
    raise ValueError(f'unknown loss_type: {args.loss_type}')


@torch.no_grad()
def estimate_cam_mean(loader, device):
    """One-pass mean over a dataloader -> (98,) tensor."""
    total = torch.zeros(98, device=device)
    n = 0
    for batch in tqdm(loader, desc='cam_mean'):
        gains = batch[1].to(device, non_blocking=True)
        total += gains.sum(dim=0)
        n += gains.size(0)
    return total / max(n, 1)


def topk_hit(pred_logit, gains_raw, k):
    idx = pred_logit.topk(k, dim=-1).indices
    best = gains_raw.argmax(dim=-1, keepdim=True)
    return (idx == best).any(dim=-1).float().mean().item()


def precision_at_k(pred_logit, gains_raw, k):
    """|topk(pred) ∩ topk(gains)| / k, averaged over batch."""
    pred_top = pred_logit.topk(k, dim=-1).indices  # (B, k)
    gt_top = gains_raw.topk(k, dim=-1).indices     # (B, k)
    # build set membership via one-hot
    B = pred_top.size(0)
    gt_mask = torch.zeros_like(gains_raw)
    gt_mask.scatter_(1, gt_top, 1.0)
    hits = gt_mask.gather(1, pred_top).sum(-1) / k
    return hits.mean().item()


def pred_entropy(pred_logit):
    """Mean per-sample entropy of softmax(logit).
    Uniform => ln(98) ≈ 4.585; collapsed delta => 0.
    """
    p = F.softmax(pred_logit, dim=-1)
    ent = -(p * torch.log(p.clamp_min(1e-12))).sum(dim=-1)
    return ent.mean().item()


def _accumulate_metrics(logit, gains, B, sums, args):
    """In-place update sums dict with batch-level metric contributions."""
    lf = logit.float()
    sums['top1'] += topk_hit(lf, gains, 1) * B
    sums['top3'] += topk_hit(lf, gains, 3) * B
    sums['p_at_k'] += precision_at_k(lf, gains, args.top_k) * B
    sums['p_at_10'] += precision_at_k(lf, gains, 10) * B
    sums['ent'] += pred_entropy(lf) * B


def _make_sums():
    return {'top1': 0.0, 'top3': 0.0, 'p_at_k': 0.0, 'p_at_10': 0.0,
            'ent': 0.0, 'loss': 0.0, 'n': 0.0}


def train_one_epoch(model, loader, optimizer, device, scaler, args,
                    rank, world_size, cam_mean=None):
    model.train()
    sums = _make_sums()

    pbar = tqdm(loader, desc='Train') if rank == 0 else loader
    for current, gains, mask in pbar:
        current = current.to(device, non_blocking=True)
        gains = gains.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        B = current.shape[0]

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            logit = model(current, mask) if args.use_seen_mask else model(current)
            loss, _ = compute_loss(logit, gains, args, cam_mean=cam_mean)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            # metrics use raw gains as ground truth (user wants absolute-best cam)
            _accumulate_metrics(logit, gains, B, sums, args)
        sums['loss'] += loss.item() * B
        sums['n'] += B

        if rank == 0 and hasattr(pbar, 'set_postfix'):
            pbar.set_postfix(loss=f'{loss.item():.4f}')

    if world_size > 1:
        keys = ['loss', 'n', 'top1', 'top3', 'p_at_k', 'p_at_10', 'ent']
        stats = torch.tensor([sums[k] for k in keys],
                              device=device, dtype=torch.float64)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        for k, v in zip(keys, stats.tolist()):
            sums[k] = v

    n = sums['n']
    return {k: sums[k] / n for k in
            ['loss', 'top1', 'top3', 'p_at_k', 'p_at_10', 'ent']}


@torch.no_grad()
def validate(model, loader, device, args, cam_mean=None):
    model.eval()
    sums = _make_sums()
    for current, gains, mask in tqdm(loader, desc='Val'):
        current = current.to(device, non_blocking=True)
        gains = gains.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        B = current.shape[0]
        with torch.amp.autocast('cuda'):
            logit = model(current, mask) if args.use_seen_mask else model(current)
            loss, _ = compute_loss(logit, gains, args, cam_mean=cam_mean)
        _accumulate_metrics(logit, gains, B, sums, args)
        sums['loss'] += loss.item() * B
        sums['n'] += B
    n = sums['n']
    return {k: sums[k] / n for k in
            ['loss', 'top1', 'top3', 'p_at_k', 'p_at_10', 'ent']}


def main(rank, world_size, args):
    device = torch.device(f'cuda:{rank}')
    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        wandb.init(project=args.wandb_project, name=args.wandb_run_name,
                   config=vars(args))

    train_loader, val_loader = get_dataloaders(
        args.reward_roots,
        batch_size=args.batch_size, num_workers=args.num_workers,
        k_min=args.k_min, k_max=args.k_max,
        num_val_scenes=args.num_val_scenes,
        split_seed=args.split_seed,
        rank=rank, world_size=world_size,
    )
    if rank == 0 and val_loader is not None:
        print(f'[reward] train={len(train_loader.dataset)}  '
              f'val={len(val_loader.dataset)}  '
              f'k∈[{args.k_min},{args.k_max}]')

    cam_mean = None
    if args.subtract_cam_mean:
        if rank == 0:
            print('[debias] computing per-cam mean over training set...')
        cam_mean = estimate_cam_mean(train_loader, device)
        if rank == 0:
            print(f'[debias] cam_mean range: '
                  f'[{cam_mean.min().item():.2f}, {cam_mean.max().item():.2f}]  '
                  f'(ratio {cam_mean.max().item()/cam_mean.min().item():.2f}x)')

    model = RewardNet(base_ch=args.base_ch,
                       use_seen_mask=args.use_seen_mask).to(device)
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler('cuda')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val = float('inf')
    patience = 0

    for epoch in range(args.epochs):
        if world_size > 1:
            train_loader.sampler.set_epoch(epoch)

        tr = train_one_epoch(
            model, train_loader, optimizer, device, scaler, args,
            rank, world_size, cam_mean=cam_mean)

        if rank == 0:
            raw = model.module if hasattr(model, 'module') else model
            va = validate(raw, val_loader, device, args, cam_mean=cam_mean)
            wandb.log({
                'epoch': epoch,
                'train_loss': tr['loss'], 'val_loss': va['loss'],
                'train_top1': tr['top1'], 'val_top1': va['top1'],
                'train_top3': tr['top3'], 'val_top3': va['top3'],
                f'train_p_at_{args.top_k}': tr['p_at_k'],
                f'val_p_at_{args.top_k}': va['p_at_k'],
                'train_p_at_10': tr['p_at_10'],
                'val_p_at_10': va['p_at_10'],
                'train_pred_entropy': tr['ent'],
                'val_pred_entropy': va['ent'],
                'lr': optimizer.param_groups[0]['lr'],
            })
            print(f"Epoch {epoch:03d} | tr_loss={tr['loss']:.4f} "
                  f"val_loss={va['loss']:.4f}  "
                  f"P@{args.top_k}={va['p_at_k']:.3f} P@10={va['p_at_10']:.3f}  "
                  f"top1={va['top1']:.3f} top3={va['top3']:.3f}  "
                  f"ent={va['ent']:.2f}")
            if va['loss'] < best_val:
                best_val = va['loss']
                patience = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': raw.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'val_loss': va['loss'],
                    'val_top1': va['top1'],
                    'val_top3': va['top3'],
                    'val_p_at_k': va['p_at_k'],
                    'val_p_at_10': va['p_at_10'],
                    'val_pred_entropy': va['ent'],
                    'args': vars(args),
                }, os.path.join(args.save_dir, 'best.pth'))
                print(f"  -> saved best (val_loss={va['loss']:.4f})")
            else:
                patience += 1

        scheduler.step()

        stop = torch.tensor(0, device=device)
        if rank == 0 and patience >= args.patience:
            stop = torch.tensor(1, device=device)
        if world_size > 1:
            dist.broadcast(stop, src=0)
        if stop.item() == 1:
            if rank == 0:
                print(f'Early stop at epoch {epoch} (patience={args.patience})')
            break

    if rank == 0:
        wandb.finish()


def ddp_worker(rank, world_size, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(args.ddp_port)
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    try:
        main(rank, world_size, args)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    gpu_ids = [int(x) for x in args.device.split(',')]
    world_size = len(gpu_ids)
    if world_size > 1:
        print(f'[DDP] launching {world_size} on GPUs: {args.device}')
        mp.spawn(ddp_worker, args=(world_size, args),
                 nprocs=world_size, join=True)
    else:
        main(rank=0, world_size=1, args=args)
