"""UB (DiVE-P backbone) 3D UNet trainer — direct soft BCE on push_visibility_all GT.

UA 학습은 2_belief/train_UA.py (cumulative-mean GT) 별도 파일.

명령어 :
    cd /home/sylee/codes/DiVE/2_belief
    sudo -E /home/sylee/miniconda3/envs/APOBU/bin/python train_UB.py \
        --data_roots /data/APOBU/beliefmap_low_occlusion_0423 \
                     /data/APOBU/beliefmap_high_occlusion_0423 \
        --save_dir /result/APOBU/U_AB_beliefmap/UB_v1 \
        --wandb_run_name UB_v1 \
        --init_belief 0.0 \
        --device 0,1 --epochs 50 --patience 5
"""
import os
import sys
import argparse
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb
from tqdm import tqdm

# support_code/ 의 model / loss / dataset 사용
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "support_code"))
from dataset_UAB import get_dataloaders  # noqa: E402
from model_UAB import UNet3DPush  # noqa: E402
from loss_UAB import soft_bce_loss as loss_fn  # noqa: E402

os.environ['PYTHONUNBUFFERED'] = '1'


def parse_args():
    parser = argparse.ArgumentParser(
        description='UB (DiVE-P backbone) 3D UNet trainer (direct soft BCE on push_visibility_all GT).')
    parser.add_argument('--data_roots', type=str, nargs='+', required=True,
                        help='One or more dataset root directories.')
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
    parser.add_argument('--prior', type=float, default=0.1,
                        help='Last-layer bias init: initial sigmoid(logit) == prior.')
    parser.add_argument('--init_belief', type=float, default=0.0,
                        help='Initial belief value at t=0. DiVE 컨벤션상 0.0으로 명시 권장.')
    parser.add_argument('--ddp_port', type=int, default=29299)
    return parser.parse_args()


def _init_belief(B, device, value=0.5):
    return torch.full((B, 1, 60, 120, 80), value, device=device)


def train_one_epoch(model, loader, optimizer, device, num_steps, scaler, rank, world_size,
                    init_belief=0.5):
    model.train()
    total_loss = 0.0
    total_samples = 0

    pbar = tqdm(loader, desc='Train') if rank == 0 else loader
    for voxel_maps, gt_pre, gt_post, swept_maps in pbar:
        voxel_maps = voxel_maps.to(device, non_blocking=True)
        gt_pre = gt_pre.to(device, non_blocking=True)
        gt_post = gt_post.to(device, non_blocking=True)
        swept_maps = swept_maps.to(device, non_blocking=True)
        B = voxel_maps.shape[0]

        belief = _init_belief(B, device, init_belief)
        has_push = (swept_maps.view(B, num_steps, -1).sum(dim=2) > 0).float()
        cum_push = has_push.cumsum(dim=1).clamp(max=1.0)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            step_loss = 0.0
            pre_loss = 0.0
            post_loss = 0.0
            pre_count = 0
            post_count = 0

            for t in range(num_steps):
                voxel = voxel_maps[:, t:t+1]
                swept = swept_maps[:, t:t+1]
                x = torch.cat([belief, voxel, swept], dim=1)

                logit = model(x)
                new_belief = torch.sigmoid(logit)

                mask = cum_push[:, t].view(B, 1, 1, 1)
                gt = gt_pre * (1 - mask) + gt_post * mask

                loss_t = loss_fn(logit, gt)
                step_loss = step_loss + loss_t

                # Full BPTT through belief chain.
                belief = new_belief

                n_post = (cum_push[:, t] > 0).sum().item()
                n_pre = B - n_post
                if n_pre > 0:
                    pre_loss += loss_t.item() * (n_pre / B)
                    pre_count += 1
                if n_post > 0:
                    post_loss += loss_t.item() * (n_post / B)
                    post_count += 1

        scaler.scale(step_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += step_loss.item() * B
        total_samples += B
        if rank == 0:
            avg_pre = pre_loss / pre_count if pre_count > 0 else 0.0
            avg_post = post_loss / post_count if post_count > 0 else 0.0
            pbar.set_postfix(loss=f'{step_loss.item():.4f}')
            if (total_samples // B) % 10 == 0:
                wandb.log({
                    'train_step_loss': step_loss.item(),
                    'train_pre_loss': avg_pre,
                    'train_post_loss': avg_post,
                    'lr': optimizer.param_groups[0]['lr'],
                })

    if world_size > 1:
        stats = torch.tensor([total_loss, total_samples], device=device, dtype=torch.float64)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_loss, total_samples = stats[0].item(), stats[1].item()

    return total_loss / total_samples


@torch.no_grad()
def validate(model, loader, device, num_steps, init_belief=0.5):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for voxel_maps, gt_pre, gt_post, swept_maps in tqdm(loader, desc='Val'):
        voxel_maps = voxel_maps.to(device, non_blocking=True)
        gt_pre = gt_pre.to(device, non_blocking=True)
        gt_post = gt_post.to(device, non_blocking=True)
        swept_maps = swept_maps.to(device, non_blocking=True)
        B = voxel_maps.shape[0]

        belief = _init_belief(B, device, init_belief)
        has_push = (swept_maps.view(B, num_steps, -1).sum(dim=2) > 0).float()
        cum_push = has_push.cumsum(dim=1).clamp(max=1.0)

        with torch.amp.autocast('cuda'):
            step_loss = 0.0
            for t in range(num_steps):
                voxel = voxel_maps[:, t:t+1]
                swept = swept_maps[:, t:t+1]
                x = torch.cat([belief, voxel, swept], dim=1)

                logit = model(x)
                new_belief = torch.sigmoid(logit)

                mask = cum_push[:, t].view(B, 1, 1, 1)
                gt = gt_pre * (1 - mask) + gt_post * mask

                step_loss = step_loss + loss_fn(logit, gt)
                belief = new_belief

        total_loss += step_loss.item() * B
        total_samples += B

    return total_loss / total_samples


def main(rank, world_size, args):
    device = torch.device(f'cuda:{rank}')

    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    train_loader, val_loader = get_dataloaders(
        args.data_roots, task='UB',
        batch_size=args.batch_size, num_steps=args.num_steps,
        num_val_scenes=args.num_val_scenes, num_workers=args.num_workers,
        rank=rank, world_size=world_size,
    )
    if rank == 0 and val_loader is not None:
        print(f'[UB] Train: {len(train_loader.dataset)} samples, '
              f'Val: {len(val_loader.dataset)} samples')

    model = UNet3DPush(prior=args.prior).to(device)
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

        train_loss = train_one_epoch(model, train_loader, optimizer, device,
                                     args.num_steps, scaler, rank, world_size,
                                     init_belief=args.init_belief)

        val_loss = None
        if rank == 0:
            raw_model = model.module if hasattr(model, 'module') else model
            val_loss = validate(raw_model, val_loader, device, args.num_steps,
                                init_belief=args.init_belief)

            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'lr': optimizer.param_groups[0]['lr'],
            })

            print(f'Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')

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
                    'val_loss': val_loss,
                }, ckpt_path)
                print(f'  -> Saved best checkpoint (val_loss={val_loss:.4f})')
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
