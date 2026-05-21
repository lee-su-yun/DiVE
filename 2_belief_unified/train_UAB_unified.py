"""Unified UA+UB 3D UNet trainer (Option A: shared backbone, 2-channel output head).

- Input  : (B, 3, D, H, W) = [belief_A_t, belief_B_t, voxel_t]   (no swept_map)
- Output : (B, 2, D, H, W) = [logit_UA, logit_UB]
- Sequence: view-only (push 없음), 따라서 transition mask / pre-post switch 전부 없음.

GT:
    UA: GT_A[t] = (1/(t+1)) Σ_{k∈K_t} pre_ray_casting[k]       (cumulative mean of pre-view visibility)
    UB: GT_B    = push_visibility_all_marginal.npy             (time-invariant)

Loss: loss = loss_fn(logit_UA, GT_A) + lambda_b * loss_fn(logit_UB, GT_B)

명령어:
    cd /home/sylee/codes/DiVE/2_belief_unified
    sudo -E /home/sylee/miniconda3/envs/APOBU/bin/python train_UAB_unified.py \
        --data_roots /data/APOBU/beliefmap_low_occlusion_0423 \
                     /data/APOBU/beliefmap_high_occlusion_0423 \
        --save_dir /result/APOBU/U_AB_beliefmap/UAB_unified_v1 \
        --wandb_run_name UAB_unified_v1 \
        --device 2,3 \
        --teacher_forcing
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "support_code"))
from model_UAB_unified import UNet3DPushUnified  # noqa: E402
from loss_UAB import soft_bce_loss as loss_fn  # noqa: E402
from dataset_UAB_unified import get_dataloaders  # noqa: E402

os.environ['PYTHONUNBUFFERED'] = '1'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Unified UA+UB trainer (shared backbone, 2-channel head).')
    parser.add_argument('--data_roots', type=str, nargs='+', required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--wandb_run_name', type=str, required=True)
    parser.add_argument('--wandb_project', type=str, default='U_AB')

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_steps', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--num_val_scenes', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=16)

    parser.add_argument('--device', type=str, default='0',
                        help='CUDA_VISIBLE_DEVICES value; comma-separated for DDP.')
    parser.add_argument('--prior_a', type=float, default=0.1,
                        help='UA head bias init: initial sigmoid(logit_A) == prior_a.')
    parser.add_argument('--prior_b', type=float, default=0.1,
                        help='UB head bias init: initial sigmoid(logit_B) == prior_b.')
    parser.add_argument('--lambda_b', type=float, default=1.0,
                        help='Weight for UB loss in the combined objective.')
    parser.add_argument('--init_belief_a', type=float, default=0.0)
    parser.add_argument('--init_belief_b', type=float, default=0.0)

    parser.add_argument('--teacher_forcing', action='store_true',
                        help='Shorthand for --teacher_forcing_prob 1.0 on UA belief.')
    parser.add_argument('--teacher_forcing_prob', type=float, default=None,
                        help='UA belief TF prob. 0.0=self-feed, 1.0=pure TF, in (0,1)=scheduled sampling. '
                             'UB belief은 항상 self-feed (GT가 time-invariant라 TF 의미 약함).')
    parser.add_argument('--tf_anneal_to', type=float, default=None,
                        help='If set, linearly anneal UA teacher_forcing_prob to this value across epochs.')
    parser.add_argument('--best_metric', type=str, default='selffeed',
                        choices=['selffeed', 'tf'])
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to checkpoint to load model weights from.')
    parser.add_argument('--ddp_port', type=int, default=29299)
    args = parser.parse_args()
    if args.teacher_forcing_prob is None:
        args.teacher_forcing_prob = 1.0 if args.teacher_forcing else 0.0
    return args


def get_epoch_tf_prob(epoch, total_epochs, prob_start, prob_end):
    if prob_end is None:
        return prob_start
    frac = epoch / max(total_epochs - 1, 1)
    return prob_start + frac * (prob_end - prob_start)


def _init_belief(B, device, value=0.0):
    return torch.full((B, 1, 60, 120, 80), value, device=device)


def train_one_epoch(model, loader, optimizer, device, num_steps, scaler,
                    rank, world_size, tf_prob=0.0,
                    init_belief_a=0.0, init_belief_b=0.0, lambda_b=1.0):
    model.train()
    total_loss = 0.0
    total_loss_a = 0.0
    total_loss_b = 0.0
    total_samples = 0

    pbar = tqdm(loader, desc='Train') if rank == 0 else loader
    for voxel_maps, per_view_pre, gt_marginal in pbar:
        voxel_maps = voxel_maps.to(device, non_blocking=True)
        per_view_pre = per_view_pre.to(device, non_blocking=True)
        gt_marginal = gt_marginal.to(device, non_blocking=True)
        B = voxel_maps.shape[0]

        belief_a = _init_belief(B, device, init_belief_a)
        belief_b = _init_belief(B, device, init_belief_b)

        acc_pre = torch.zeros(B, 60, 120, 80, device=device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            step_loss = 0.0
            step_loss_a = 0.0
            step_loss_b = 0.0

            for t in range(num_steps):
                voxel = voxel_maps[:, t:t+1]
                x = torch.cat([belief_a, belief_b, voxel], dim=1)  # (B, 3, D, H, W)

                logits = model(x)                  # (B, 2, D, H, W)
                logit_a = logits[:, 0:1]
                logit_b = logits[:, 1:2]

                # UA GT: cumulative mean of per-view pre-visibility
                acc_pre = acc_pre + per_view_pre[:, t].float()
                gt_a = acc_pre / float(t + 1)      # (B, D, H, W)

                # UB GT: time-invariant marginal
                gt_b = gt_marginal                 # (B, D, H, W)

                loss_a_t = loss_fn(logit_a, gt_a)
                loss_b_t = loss_fn(logit_b, gt_b)
                loss_t = loss_a_t + lambda_b * loss_b_t

                step_loss = step_loss + loss_t
                step_loss_a = step_loss_a + loss_a_t
                step_loss_b = step_loss_b + loss_b_t

                # Belief feedback
                if tf_prob >= 1.0:
                    belief_a = gt_a.unsqueeze(1)   # pure TF: BPTT 차단
                elif tf_prob <= 0.0:
                    belief_a = torch.sigmoid(logit_a)
                else:
                    sf_a = torch.sigmoid(logit_a)
                    use_tf = torch.rand(B, 1, 1, 1, 1, device=device) < tf_prob
                    belief_a = torch.where(use_tf, gt_a.unsqueeze(1), sf_a)

                # UB belief은 항상 self-feed (GT가 time-invariant라 TF가 큰 의미 없음)
                belief_b = torch.sigmoid(logit_b)

        scaler.scale(step_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += step_loss.item() * B
        total_loss_a += step_loss_a.item() * B
        total_loss_b += step_loss_b.item() * B
        total_samples += B
        if rank == 0:
            pbar.set_postfix(
                loss=f'{step_loss.item():.4f}',
                la=f'{step_loss_a.item():.4f}',
                lb=f'{step_loss_b.item():.4f}',
            )
            if (total_samples // B) % 10 == 0:
                wandb.log({
                    'train_step_loss': step_loss.item(),
                    'train_step_loss_a': step_loss_a.item(),
                    'train_step_loss_b': step_loss_b.item(),
                    'lr': optimizer.param_groups[0]['lr'],
                })

    if world_size > 1:
        stats = torch.tensor(
            [total_loss, total_loss_a, total_loss_b, total_samples],
            device=device, dtype=torch.float64)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_loss, total_loss_a, total_loss_b, total_samples = (
            stats[0].item(), stats[1].item(), stats[2].item(), stats[3].item())

    return (total_loss / total_samples,
            total_loss_a / total_samples,
            total_loss_b / total_samples)


@torch.no_grad()
def validate(model, loader, device, num_steps,
             init_belief_a=0.0, init_belief_b=0.0, lambda_b=1.0):
    """두 모드 동시 평가: self-feed + teacher-forced (UA belief에 한해)."""
    model.eval()
    total_loss_sf = 0.0
    total_loss_tf = 0.0
    total_loss_a_sf = 0.0
    total_loss_b_sf = 0.0
    total_loss_a_tf = 0.0
    total_loss_b_tf = 0.0
    total_samples = 0

    for voxel_maps, per_view_pre, gt_marginal in tqdm(loader, desc='Val'):
        voxel_maps = voxel_maps.to(device, non_blocking=True)
        per_view_pre = per_view_pre.to(device, non_blocking=True)
        gt_marginal = gt_marginal.to(device, non_blocking=True)
        B = voxel_maps.shape[0]

        belief_a_sf = _init_belief(B, device, init_belief_a)
        belief_b_sf = _init_belief(B, device, init_belief_b)
        belief_a_tf = _init_belief(B, device, init_belief_a)
        belief_b_tf = _init_belief(B, device, init_belief_b)

        acc_pre = torch.zeros(B, 60, 120, 80, device=device)

        with torch.amp.autocast('cuda'):
            sl_sf = 0.0
            sl_tf = 0.0
            sl_a_sf = 0.0; sl_b_sf = 0.0
            sl_a_tf = 0.0; sl_b_tf = 0.0

            for t in range(num_steps):
                voxel = voxel_maps[:, t:t+1]

                x_sf = torch.cat([belief_a_sf, belief_b_sf, voxel], dim=1)
                logits_sf = model(x_sf)
                logit_a_sf, logit_b_sf = logits_sf[:, 0:1], logits_sf[:, 1:2]

                x_tf = torch.cat([belief_a_tf, belief_b_tf, voxel], dim=1)
                logits_tf = model(x_tf)
                logit_a_tf, logit_b_tf = logits_tf[:, 0:1], logits_tf[:, 1:2]

                acc_pre = acc_pre + per_view_pre[:, t].float()
                gt_a = acc_pre / float(t + 1)
                gt_b = gt_marginal

                la_sf = loss_fn(logit_a_sf, gt_a); lb_sf = loss_fn(logit_b_sf, gt_b)
                la_tf = loss_fn(logit_a_tf, gt_a); lb_tf = loss_fn(logit_b_tf, gt_b)

                sl_a_sf += la_sf; sl_b_sf += lb_sf
                sl_a_tf += la_tf; sl_b_tf += lb_tf
                sl_sf = sl_sf + la_sf + lambda_b * lb_sf
                sl_tf = sl_tf + la_tf + lambda_b * lb_tf

                # next belief
                belief_a_sf = torch.sigmoid(logit_a_sf)
                belief_b_sf = torch.sigmoid(logit_b_sf)
                belief_a_tf = gt_a.unsqueeze(1)         # UA: TF
                belief_b_tf = torch.sigmoid(logit_b_tf) # UB: self-feed even in TF mode

        total_loss_sf += sl_sf.item() * B
        total_loss_tf += sl_tf.item() * B
        total_loss_a_sf += sl_a_sf.item() * B
        total_loss_b_sf += sl_b_sf.item() * B
        total_loss_a_tf += sl_a_tf.item() * B
        total_loss_b_tf += sl_b_tf.item() * B
        total_samples += B

    n = total_samples
    return {
        'sf': total_loss_sf / n,
        'tf': total_loss_tf / n,
        'a_sf': total_loss_a_sf / n,
        'b_sf': total_loss_b_sf / n,
        'a_tf': total_loss_a_tf / n,
        'b_tf': total_loss_b_tf / n,
    }


def main(rank, world_size, args):
    device = torch.device(f'cuda:{rank}')

    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    train_loader, val_loader = get_dataloaders(
        args.data_roots,
        batch_size=args.batch_size, num_steps=args.num_steps,
        num_val_scenes=args.num_val_scenes, num_workers=args.num_workers,
        rank=rank, world_size=world_size,
    )
    if rank == 0 and val_loader is not None:
        print(f'[UAB_unified] Train: {len(train_loader.dataset)} samples, '
              f'Val: {len(val_loader.dataset)} samples')
        print(f'[UAB_unified] tf_prob_start={args.teacher_forcing_prob} '
              f'tf_anneal_to={args.tf_anneal_to} lambda_b={args.lambda_b} '
              f'best_metric={args.best_metric}')

    model = UNet3DPushUnified(prior_a=args.prior_a, prior_b=args.prior_b).to(device)

    if args.resume_from is not None:
        if rank == 0:
            print(f'[UAB_unified] Loading model weights from {args.resume_from}')
        ckpt = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])

    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler('cuda')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        if world_size > 1:
            train_loader.sampler.set_epoch(epoch)

        epoch_tf_prob = get_epoch_tf_prob(
            epoch, args.epochs, args.teacher_forcing_prob, args.tf_anneal_to)

        train_loss, train_loss_a, train_loss_b = train_one_epoch(
            model, train_loader, optimizer, device,
            args.num_steps, scaler, rank, world_size,
            tf_prob=epoch_tf_prob,
            init_belief_a=args.init_belief_a,
            init_belief_b=args.init_belief_b,
            lambda_b=args.lambda_b,
        )

        if rank == 0:
            raw_model = model.module if hasattr(model, 'module') else model
            v = validate(raw_model, val_loader, device, args.num_steps,
                         init_belief_a=args.init_belief_a,
                         init_belief_b=args.init_belief_b,
                         lambda_b=args.lambda_b)

            val_loss = v['tf'] if args.best_metric == 'tf' else v['sf']

            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_loss_a': train_loss_a,
                'train_loss_b': train_loss_b,
                'val_loss_selffeed': v['sf'],
                'val_loss_tf': v['tf'],
                'val_loss_a_selffeed': v['a_sf'],
                'val_loss_b_selffeed': v['b_sf'],
                'val_loss_a_tf': v['a_tf'],
                'val_loss_b_tf': v['b_tf'],
                'tf_prob': epoch_tf_prob,
                'lr': optimizer.param_groups[0]['lr'],
            })

            print(f'Epoch {epoch:03d} | tf_p={epoch_tf_prob:.2f} | '
                  f'Train: {train_loss:.4f} (A={train_loss_a:.4f} B={train_loss_b:.4f}) | '
                  f'Val SF: {v["sf"]:.4f} (A={v["a_sf"]:.4f} B={v["b_sf"]:.4f}) | '
                  f'Val TF: {v["tf"]:.4f} (A={v["a_tf"]:.4f} B={v["b_tf"]:.4f})')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                ckpt_path = os.path.join(args.save_dir, 'best.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': raw_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'val_loss_selffeed': v['sf'],
                    'val_loss_tf': v['tf'],
                    'val_loss_a_selffeed': v['a_sf'],
                    'val_loss_b_selffeed': v['b_sf'],
                }, ckpt_path)
                print(f'  -> Saved best checkpoint '
                      f'(by {args.best_metric}, val_loss={val_loss:.4f})')
            else:
                patience_counter += 1

        scheduler.step()

        should_stop = torch.tensor(0, device=device)
        if rank == 0 and patience_counter >= args.patience:
            should_stop = torch.tensor(1, device=device)
        if world_size > 1:
            dist.broadcast(should_stop, src=0)
        if should_stop.item() == 1:
            if rank == 0:
                print(f'Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)')
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
        print(f'[DDP] Launching {world_size} processes on GPUs: {args.device}')
        mp.spawn(ddp_worker, args=(world_size, args), nprocs=world_size, join=True)
    else:
        main(rank=0, world_size=1, args=args)
