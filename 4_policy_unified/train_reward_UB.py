"""Train UB reward model (siamese / permutation-equivariant variant).

Self-contained training script for 4_policy_test. Imports the local
dataset_reward_UB and model_reward_UB modules (no support_code dependency).

Loss objectives (only these two; for parallel ablation):
    1) loss_type=bce_sigmoid_z   BCEWithLogits with sigmoid(masked_zscore(g)/T)
                                 target. Score read as logit. Ranking-oriented.
    2) loss_type=mse             Direct regression on gain. Score read as
                                 prediction; predicted_gain = score * gain_scale.

Metrics (per epoch, masked by valid_mask):
    val_top1, val_top3      argmax(gains) hit-rate under predicted scores
    val_p_at_K, val_p_at_10 |topk(score) ∩ topk(gains)| / K  (K from --top_k)
    val_pred_entropy        softmax-based; uniform-over-20 ≈ ln(20) ≈ 2.996

명령어 (bce_sigmoid_z, 빠른 세팅 탐색 — --test = subsample 20%) :
    cd /home/sylee/codes/DiVE/4_policy_test
    python train_reward_UB.py \\
        --reward_roots /data/APOBU/DiVE/ub_reward_dataset_ycb_v3_marginal \\
        --data_roots /data/APOBU/beliefmap_high_occlusion_ycb_v3 \\
                     /result/DiVE_data/beliefmap_low_occlusion_ycb_v3 \\
        --save_dir /result/APOBU/DiVE/reward_UB_ycb_v3_marginal_siamese_sigz \\
        --wandb_run_name reward_UB_siamese_sigz \\
        --loss_type bce_sigmoid_z --temperature 1.0 \\
        --test \\
        --device 0,1

명령어 (mse, 다른 GPU에 병렬로) :
    python train_reward_UB.py \\
        --reward_roots /data/APOBU/DiVE/ub_reward_dataset_ycb_v3_marginal \\
        --data_roots /data/APOBU/beliefmap_high_occlusion_ycb_v3 \\
                     /result/DiVE_data/beliefmap_low_occlusion_ycb_v3 \\
        --save_dir /result/APOBU/DiVE/reward_UB_ycb_v3_marginal_siamese_mse \\
        --wandb_run_name reward_UB_siamese_mse \\
        --loss_type mse --gain_scale 1e4 \\
        --test \\
        --device 2,3 --ddp_port 29302

세팅이 정해지면 --test 빼고 full dataset 으로 본 학습 돌리면 됨.
명시적 비율은 --subsample_frac 0.3 같은 식으로 지정 가능.
"""
import os
import math
import time
import argparse

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import wandb
from tqdm import tqdm

from dataset_reward_UB import get_dataloaders
from model_reward_UB import RewardNetUB

os.environ['PYTHONUNBUFFERED'] = '1'

# Avoid fd-strategy leak that exhausted shm after ~4 epochs (errno=0 on shm_open).
mp.set_sharing_strategy('file_system')

NUM_PUSHES = 20


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--reward_roots', type=str, nargs='+', required=True)
    p.add_argument('--data_roots', type=str, nargs='+', required=True,
                   help='Original beliefmap roots (for swept_map loading).')
    p.add_argument('--save_dir', type=str, required=True)
    p.add_argument('--wandb_run_name', type=str, required=True)
    p.add_argument('--wandb_project', type=str, default='U_AB_reward')
    p.add_argument('--no_wandb', action='store_true',
                   help='Disable wandb entirely (no init, no log).')

    # NOTE: effective compute batch = batch_size * NUM_PUSHES. Default 16
    # gives 320 per step — lower if OOM. Old channel-stacked model used 64.
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--patience', type=int, default=8)
    p.add_argument('--num_val_scenes', type=int, default=10)
    p.add_argument('--num_workers', type=int, default=8)

    p.add_argument('--k_min', type=int, default=2)
    p.add_argument('--k_max', type=int, default=7)
    p.add_argument('--split_seed', type=int, default=0)

    p.add_argument('--loss_type', type=str, default='bce_sigmoid_z',
                   choices=['bce_sigmoid_z', 'mse'])
    p.add_argument('--best_metric', type=str, default='val_loss',
                   choices=['val_loss', 'val_top1', 'val_p_at_k',
                            'val_p_at_10', 'val_gain_ratio'],
                   help='Metric used for best-ckpt selection and early stop. '
                        'val_loss = min loss. Others = max metric value.')
    p.add_argument('--top_k', type=int, default=5,
                   help='K for Precision@K metric only.')
    p.add_argument('--temperature', type=float, default=1.0,
                   help='Used by bce_sigmoid_z target sharpness.')
    p.add_argument('--gain_scale', type=float, default=1e4,
                   help='Divisor applied to gains for mse target. '
                        'Predicted gain at inference = output * gain_scale.')

    p.add_argument('--base_ch', type=int, default=16)
    p.add_argument('--alpha', type=float, default=1.0,
                   help='Initial weight on overlap_norm '
                        '(final = alpha*overlap_norm + beta*learned).')
    p.add_argument('--beta', type=float, default=0.1,
                   help='Initial weight on learned residual. Small (e.g. 0.1) '
                        'anchors output to overlap baseline.')
    p.add_argument('--learnable_ab', action='store_true',
                   help='Make alpha, beta nn.Parameter and let optimizer tune.')
    p.add_argument('--use_attn', action='store_true',
                   help='Insert 1-layer self-attention across N action tokens '
                        'between encoder GAP and MLP head. Lets each action '
                        'compare against others (top1 discrimination).')
    p.add_argument('--attn_heads', type=int, default=4,
                   help='Number of attention heads (used with --use_attn).')
    p.add_argument('--subsample_frac', type=float, default=1.0,
                   help='Fraction of (scene,push) buckets used per epoch. '
                        'Train: different random subset each epoch; val: same '
                        'subset across epochs (deterministic).')
    p.add_argument('--max_steps_per_epoch', type=int, default=0,
                   help='Cap train steps per epoch (0 = no cap). Different '
                        'samples each epoch (shuffle is on).')
    p.add_argument('--max_val_steps', type=int, default=0,
                   help='Cap val steps per epoch (0 = no cap). Same samples '
                        'each epoch (val sampler is deterministic).')
    p.add_argument('--test', action='store_true',
                   help='Quick-iteration mode: max_steps_per_epoch=100, '
                        'max_val_steps=50 (≈ 1.5 min/epoch at bs=4). Either '
                        'cap can still be overridden explicitly.')
    p.add_argument('--device', type=str, default='0')
    p.add_argument('--ddp_port', type=int, default=29301)
    p.add_argument('--log_every_steps', type=int, default=50,
                   help='Log per-batch train metrics to wandb + terminal every '
                        'N steps. Keys prefixed with "step_". 0 disables.')
    args = p.parse_args()
    if args.test:
        if args.max_steps_per_epoch == 0:
            args.max_steps_per_epoch = 100
        if args.max_val_steps == 0:
            args.max_val_steps = 50
    return args


def _masked_zscore(g, mask):
    """Per-sample zscore of g over valid (mask=True) positions.

    Invalid positions are kept in `g` for shape but excluded from stats.
    Callers mask invalid positions in downstream losses.
    """
    mf = mask.float()
    cnt = mf.sum(-1, keepdim=True).clamp_min(1.0)
    m = (g * mf).sum(-1, keepdim=True) / cnt
    var = (((g - m) ** 2) * mf).sum(-1, keepdim=True) / cnt
    s = var.sqrt() + 1e-6
    return (g - m) / s


def compute_loss(score, gains, valid_mask, args):
    mask = valid_mask
    mf = mask.float()
    denom = mf.sum().clamp_min(1.0)
    if args.loss_type == 'bce_sigmoid_z':
        g = _masked_zscore(gains, mask)
        tgt = torch.sigmoid(g / args.temperature)
        bce = F.binary_cross_entropy_with_logits(score, tgt, reduction='none')
        bce = bce * mf
        return bce.sum() / denom, tgt
    if args.loss_type == 'mse':
        tgt = gains / args.gain_scale
        elt = (score - tgt) ** 2
        elt = elt * mf
        return elt.sum() / denom, tgt
    raise ValueError(f'unknown loss_type: {args.loss_type}')


def topk_hit(pred, gains_raw, valid_mask, k):
    pred = pred.masked_fill(~valid_mask, float('-inf'))
    gains = gains_raw.masked_fill(~valid_mask, float('-inf'))
    idx = pred.topk(k, dim=-1).indices
    best = gains.argmax(dim=-1, keepdim=True)
    return (idx == best).any(dim=-1).float().mean().item()


def precision_at_k(pred, gains_raw, valid_mask, k):
    pred = pred.masked_fill(~valid_mask, float('-inf'))
    gains = gains_raw.masked_fill(~valid_mask, float('-inf'))
    pred_top = pred.topk(k, dim=-1).indices
    gt_top = gains.topk(k, dim=-1).indices
    gt_mask = torch.zeros_like(gains)
    gt_mask.scatter_(1, gt_top, 1.0)
    hits = gt_mask.gather(1, pred_top).sum(-1) / k
    return hits.mean().item()


def pred_entropy(pred, valid_mask):
    masked = pred.masked_fill(~valid_mask, float('-inf'))
    p = F.softmax(masked, dim=-1)
    ent = -(p * torch.log(p.clamp_min(1e-12))).sum(dim=-1)
    return ent.mean().item()


def gain_ratio_metric(score, gains_raw, valid_mask):
    """gain[argmax(score)] / max(gain), averaged over batch.

    Continuous version of top1: 0 = picked worst valid action,
    1 = picked the actual best. Robust to near-ties (#1 vs #2 close).
    """
    s = score.masked_fill(~valid_mask, float('-inf'))
    g = gains_raw.masked_fill(~valid_mask, float('-inf'))
    pick = s.argmax(dim=-1)                                       # (B,)
    picked = gains_raw.gather(1, pick.unsqueeze(-1)).squeeze(-1)  # (B,)
    best = g.max(dim=-1).values                                   # (B,)
    return (picked / (best.abs() + 1e-9)).mean().item()


def _accumulate_metrics(score, gains, valid_mask, B, sums, args):
    sf = score.float()
    sums['top1'] += topk_hit(sf, gains, valid_mask, 1) * B
    sums['top3'] += topk_hit(sf, gains, valid_mask, 3) * B
    sums['p_at_k'] += precision_at_k(sf, gains, valid_mask, args.top_k) * B
    sums['p_at_10'] += precision_at_k(sf, gains, valid_mask, 10) * B
    sums['ent'] += pred_entropy(sf, valid_mask) * B
    sums['gain_ratio'] += gain_ratio_metric(sf, gains, valid_mask) * B
    if args.loss_type == 'mse':
        mf = valid_mask.float()
        ae = (sf * args.gain_scale - gains).abs() * mf
        denom = mf.sum().clamp_min(1.0)
        sums['mae'] += (ae.sum() / denom).item() * B


def _make_sums():
    return {'top1': 0.0, 'top3': 0.0, 'p_at_k': 0.0, 'p_at_10': 0.0,
            'ent': 0.0, 'mae': 0.0, 'gain_ratio': 0.0,
            'loss': 0.0, 'n': 0.0}


# Best-ckpt / early-stop metric direction.
HIGHER_IS_BETTER = {
    'val_loss': False,
    'val_top1': True,
    'val_p_at_k': True,
    'val_p_at_10': True,
    'val_gain_ratio': True,
}

# Map flag → key in the train/val summary dict.
METRIC_KEY = {
    'val_loss': 'loss',
    'val_top1': 'top1',
    'val_p_at_k': 'p_at_k',
    'val_p_at_10': 'p_at_10',
    'val_gain_ratio': 'gain_ratio',
}


def train_one_epoch(model, loader, optimizer, device, scaler, args,
                    rank, world_size, epoch=0):
    model.train()
    sums = _make_sums()
    cap = int(args.max_steps_per_epoch) if args.max_steps_per_epoch > 0 else 0
    steps_per_epoch = min(len(loader), cap) if cap > 0 else len(loader)
    log_every = max(0, int(args.log_every_steps))

    if rank == 0:
        pbar = tqdm(loader, total=steps_per_epoch, desc='Train', leave=False)
    else:
        pbar = loader
    for i, (current, swept, gains, valid_mask) in enumerate(pbar):
        if cap > 0 and i >= cap:
            break
        current = current.to(device, non_blocking=True)
        swept = swept.to(device, non_blocking=True)
        gains = gains.to(device, non_blocking=True)
        valid_mask = valid_mask.to(device, non_blocking=True)
        B = current.shape[0]

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            score = model(current, swept, valid_mask)
            loss, _ = compute_loss(score, gains, valid_mask, args)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            _accumulate_metrics(score, gains, valid_mask, B, sums, args)
        sums['loss'] += loss.item() * B
        sums['n'] += B

        if rank == 0:
            if hasattr(pbar, 'set_postfix'):
                pbar.set_postfix(loss=f'{loss.item():.4f}',
                                 scale=f'{scaler.get_scale():.0f}')
            if log_every > 0 and (i % log_every == 0):
                with torch.no_grad():
                    sf = score.float()
                    s_top1 = topk_hit(sf, gains, valid_mask, 1)
                    s_top3 = topk_hit(sf, gains, valid_mask, 3)
                    s_pk = precision_at_k(sf, gains, valid_mask, args.top_k)
                    s_p10 = precision_at_k(sf, gains, valid_mask, 10)
                    s_ent = pred_entropy(sf, valid_mask)
                    step_log = {
                        'step_loss': loss.item(),
                        'step_top1': s_top1,
                        'step_top3': s_top3,
                        f'step_p_at_{args.top_k}': s_pk,
                        'step_p_at_10': s_p10,
                        'step_pred_entropy': s_ent,
                        'global_step': epoch * steps_per_epoch + i,
                    }
                    mae_part = ''
                    if args.loss_type == 'mse':
                        mf = valid_mask.float()
                        ae = (sf * args.gain_scale - gains).abs() * mf
                        s_mae = (ae.sum() / mf.sum().clamp_min(1.0)).item()
                        step_log['step_mae'] = s_mae
                        mae_part = f"  mae={s_mae:.0f}"
                wandb.log(step_log)
                tqdm.write(
                    f"  step {epoch:03d}.{i:04d} | loss={loss.item():.4f}  "
                    f"top1={s_top1:.3f}  top3={s_top3:.3f}  "
                    f"p@{args.top_k}={s_pk:.3f}  p@10={s_p10:.3f}  "
                    f"ent={s_ent:.2f}{mae_part}"
                )

    if world_size > 1:
        keys = ['loss', 'n', 'top1', 'top3', 'p_at_k', 'p_at_10',
                'ent', 'mae', 'gain_ratio']
        stats = torch.tensor([sums[k] for k in keys],
                             device=device, dtype=torch.float64)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        for k, v in zip(keys, stats.tolist()):
            sums[k] = v

    if rank == 0:
        print(f'  [amp] scale={scaler.get_scale():.1f}  '
              f'(<1 means inf-grad skips dominating optimizer steps)')

    n = sums['n']
    return {k: sums[k] / n for k in
            ['loss', 'top1', 'top3', 'p_at_k', 'p_at_10',
             'ent', 'mae', 'gain_ratio']}


@torch.no_grad()
def validate(model, loader, device, args):
    model.eval()
    sums = _make_sums()
    cap = int(args.max_val_steps) if args.max_val_steps > 0 else 0
    total = min(len(loader), cap) if cap > 0 else len(loader)
    pbar = tqdm(loader, total=total, desc='Val', leave=False)
    for i, (current, swept, gains, valid_mask) in enumerate(pbar):
        if cap > 0 and i >= cap:
            break
        current = current.to(device, non_blocking=True)
        swept = swept.to(device, non_blocking=True)
        gains = gains.to(device, non_blocking=True)
        valid_mask = valid_mask.to(device, non_blocking=True)
        B = current.shape[0]
        with torch.amp.autocast('cuda'):
            score = model(current, swept, valid_mask)
            loss, _ = compute_loss(score, gains, valid_mask, args)
        _accumulate_metrics(score, gains, valid_mask, B, sums, args)
        sums['loss'] += loss.item() * B
        sums['n'] += B
    n = sums['n']
    return {k: sums[k] / n for k in
            ['loss', 'top1', 'top3', 'p_at_k', 'p_at_10',
             'ent', 'mae', 'gain_ratio']}


def main(rank, world_size, args):
    device = torch.device(f'cuda:{rank}')
    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        wandb_mode = 'disabled' if args.no_wandb else 'online'
        wandb.init(project=args.wandb_project, name=args.wandb_run_name,
                   config=vars(args), mode=wandb_mode)

    train_loader, val_loader = get_dataloaders(
        args.reward_roots, args.data_roots,
        batch_size=args.batch_size, num_workers=args.num_workers,
        k_min=args.k_min, k_max=args.k_max,
        num_val_scenes=args.num_val_scenes,
        split_seed=args.split_seed,
        rank=rank, world_size=world_size,
        subsample_frac=args.subsample_frac,
    )
    if rank == 0 and val_loader is not None:
        frac_str = (f'  subsample_frac={args.subsample_frac:.2f}'
                    if args.subsample_frac < 1.0 else '')
        cap_tr = args.max_steps_per_epoch
        cap_va = args.max_val_steps
        cap_str = ''
        if cap_tr > 0 or cap_va > 0:
            cap_str = f'  max_steps_tr={cap_tr or "∞"}/va={cap_va or "∞"}'
        print(f'[reward_UB siamese] train={len(train_loader.dataset)}  '
              f'val={len(val_loader.dataset)}  '
              f'k∈[{args.k_min},{args.k_max}]{frac_str}{cap_str}')
        # Random baselines for paste-and-diagnose.
        print(f'[random baselines @ N={NUM_PUSHES}]  '
              f'top1={1/NUM_PUSHES:.3f}  top3={3/NUM_PUSHES:.3f}  '
              f'p@{args.top_k}={args.top_k/NUM_PUSHES:.3f}  '
              f'p@10={10/NUM_PUSHES:.3f}  ent=ln{NUM_PUSHES}={math.log(NUM_PUSHES):.3f}')
        if args.loss_type == 'bce_sigmoid_z':
            print(f'[loss=bce_sigmoid_z, T={args.temperature}]  '
                  f'random BCE ≈ 0.693; learning = strictly lower')
        elif args.loss_type == 'mse':
            print(f'[loss=mse, gain_scale={args.gain_scale:.1e}]  '
                  f'compare val_mae to gain magnitude (printed each epoch)')
        ab_mode = 'learnable' if args.learnable_ab else 'fixed'
        attn_str = f'  attn={args.attn_heads}h' if args.use_attn else '  attn=off'
        print(f'[residual mixing, {ab_mode}]  final = alpha*overlap_norm + '
              f'beta*learned  (init alpha={args.alpha}, beta={args.beta}){attn_str}')

    model = RewardNetUB(base_ch=args.base_ch,
                        alpha=args.alpha, beta=args.beta,
                        learnable_ab=args.learnable_ab,
                        use_attn=args.use_attn,
                        attn_heads=args.attn_heads).to(device)
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler('cuda')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val = (-float('inf') if HIGHER_IS_BETTER[args.best_metric]
                else float('inf'))
    patience = 0

    for epoch in range(args.epochs):
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        t0 = time.time()
        tr = train_one_epoch(
            model, train_loader, optimizer, device, scaler, args,
            rank, world_size, epoch=epoch)
        dt_tr = time.time() - t0

        if rank == 0:
            raw = model.module if hasattr(model, 'module') else model
            t1 = time.time()
            va = validate(raw, val_loader, device, args)
            dt_va = time.time() - t1
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
                'train_mae': tr['mae'], 'val_mae': va['mae'],
                'train_gain_ratio': tr['gain_ratio'],
                'val_gain_ratio': va['gain_ratio'],
                'lr': optimizer.param_groups[0]['lr'],
            })
            def _row(name, s):
                mae_part = (f"  mae={s['mae']:.0f}"
                            if args.loss_type == 'mse' else '')
                return (f"  {name} | loss={s['loss']:.4f}  "
                        f"top1={s['top1']:.3f}  top3={s['top3']:.3f}  "
                        f"p@{args.top_k}={s['p_at_k']:.3f}  "
                        f"p@10={s['p_at_10']:.3f}  "
                        f"gain_ratio={s['gain_ratio']:.3f}  "
                        f"ent={s['ent']:.2f}{mae_part}")

            # Best-ckpt selection by chosen metric.
            metric_key = METRIC_KEY[args.best_metric]
            cur = va[metric_key]
            higher_better = HIGHER_IS_BETTER[args.best_metric]
            improved = (cur > best_val) if higher_better else (cur < best_val)

            star = '*' if improved else ' '
            a_cur, b_cur = raw.current_ab()
            ab_str = f"  α={a_cur:.3f} β={b_cur:.3f}" if args.learnable_ab else ''
            print(f"Epoch {epoch:03d} | tr {dt_tr:.0f}s  va {dt_va:.0f}s  "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}{ab_str}")
            print(_row('train', tr))
            print(_row('val  ', va))
            shown_best = max(best_val, cur) if higher_better else min(best_val, cur)
            best_str = (f"best {args.best_metric}={shown_best:.4f}  "
                        f"patience={(0 if improved else patience+1)}/{args.patience}")
            print(f"  status{star}| {best_str}")
            if improved:
                best_val = cur
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
                    'val_gain_ratio': va['gain_ratio'],
                    'val_pred_entropy': va['ent'],
                    'best_metric': args.best_metric,
                    'best_value': cur,
                    'args': vars(args),
                }, os.path.join(args.save_dir, 'best.pth'))
                print(f"  -> saved best ({args.best_metric}={cur:.4f})")
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
