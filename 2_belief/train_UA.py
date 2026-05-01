"""UA 3D UNet trainer with cumulative-mean GT.

train_UAB.py와 거의 동일하되 UA 전용이고 GT만 다음과 같이 바뀜:
    GT_t = (1/|K_t|) Σ_{k∈K_t} v[k]
- t < push_step: v = v_pre[K[t]] (push 전 visibility)
- t >= push_step: v = v_post[K[t]] (push 후 visibility, K_t는 누적 그대로 유지)

기존 모델/loss는 import만 해서 재사용 (model_UAB, loss_UAB 미수정).
init belief은 --init_belief로 명시 (default 0.0).

명령어 :
    cd /home/sylee/codes/DiVE/2_belief
    sudo -E /home/sylee/miniconda3/envs/APOBU/bin/python train_UA.py \
        --data_roots /data/APOBU/beliefmap_low_occlusion_0423 \
                    /data/APOBU/beliefmap_high_occlusion_0423 \
        --save_dir /result/APOBU/U_AB_beliefmap/UA_Kt_TF \
        --wandb_run_name UA_Kt_cumulative_TF \
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

# support_code/ 의 model / loss / dataset 사용
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "support_code"))
from model_UAB import UNet3DPush  # noqa: E402
from loss_UAB import soft_bce_loss as loss_fn  # noqa: E402
from dataset_UA import get_dataloaders  # noqa: E402

os.environ['PYTHONUNBUFFERED'] = '1'


def parse_args():
    parser = argparse.ArgumentParser(
        description='UA 3D UNet trainer with cumulative-mean GT (1/|K_t|) Σ_{k∈K_t} v[k].')
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
    parser.add_argument('--teacher_forcing', action='store_true',
                        help='Shorthand for --teacher_forcing_prob 1.0.')
    parser.add_argument('--teacher_forcing_prob', type=float, default=None,
                        help='Probability of using GT (vs model output) as belief input. '
                             '0.0=self-feed, 1.0=pure TF, in (0,1)=per-element scheduled '
                             'sampling. If unspecified: 1.0 if --teacher_forcing else 0.0.')
    parser.add_argument('--tf_anneal_to', type=float, default=None,
                        help='If set, linearly anneal teacher_forcing_prob to this value '
                             'across epochs. e.g., start=1.0 end=0.0 anneals from full TF '
                             'to full self-feed.')
    parser.add_argument('--best_metric', type=str, default='selffeed',
                        choices=['selffeed', 'tf'],
                        help='Which val loss to use for best checkpoint selection.')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to checkpoint to load model weights from. '
                             'Optimizer/scheduler/scaler are reset (clean fine-tune).')
    parser.add_argument('--ddp_port', type=int, default=29299)
    args = parser.parse_args()
    if args.teacher_forcing_prob is None:
        args.teacher_forcing_prob = 1.0 if args.teacher_forcing else 0.0
    return args


def get_epoch_tf_prob(epoch, total_epochs, prob_start, prob_end):
    """Linear annealing of teacher_forcing_prob across epochs."""
    if prob_end is None:
        return prob_start
    frac = epoch / max(total_epochs - 1, 1)
    return prob_start + frac * (prob_end - prob_start)


def _init_belief(B, device, value=0.0):
    return torch.full((B, 1, 60, 120, 80), value, device=device)


def train_one_epoch(model, loader, optimizer, device, num_steps, scaler,
                    rank, world_size, tf_prob=0.0, init_belief=0.0):
    model.train()
    total_loss = 0.0
    total_samples = 0

    pbar = tqdm(loader, desc='Train') if rank == 0 else loader
    for voxel_maps, swept_maps, per_view_pre, per_view_post in pbar:
        voxel_maps = voxel_maps.to(device, non_blocking=True)
        swept_maps = swept_maps.to(device, non_blocking=True)
        # uint8 그대로 GPU에 두고, step마다 슬라이스해서 float로 캐스팅 (메모리 절약)
        per_view_pre = per_view_pre.to(device, non_blocking=True)
        per_view_post = per_view_post.to(device, non_blocking=True)
        B = voxel_maps.shape[0]

        belief = _init_belief(B, device, init_belief)
        has_push = (swept_maps.view(B, num_steps, -1).sum(dim=2) > 0).float()
        cum_push = has_push.cumsum(dim=1).clamp(max=1.0)

        # Step별 누적합 (streaming: cum_pre/cum_post를 한꺼번에 만들지 않음)
        acc_pre = torch.zeros(B, 60, 120, 80, device=device)
        acc_post = torch.zeros(B, 60, 120, 80, device=device)

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

                # 누적합 업데이트
                acc_pre = acc_pre + per_view_pre[:, t].float()
                acc_post = acc_post + per_view_post[:, t].float()

                mask = cum_push[:, t].view(B, 1, 1, 1)
                gt = ((1.0 - mask) * acc_pre + mask * acc_post) / float(t + 1)

                loss_t = loss_fn(logit, gt)
                step_loss = step_loss + loss_t

                # 다음 step belief 입력 (tf_prob에 따라 GT/self-feed 선택)
                if tf_prob >= 1.0:
                    # 순수 TF: BPTT 차단
                    belief = gt.unsqueeze(1)
                elif tf_prob <= 0.0:
                    # 순수 self-feed: full BPTT
                    belief = torch.sigmoid(logit)
                else:
                    # Mixed: per-element 동전 던져서 선택 (scheduled sampling)
                    sf_belief = torch.sigmoid(logit)
                    use_tf = torch.rand(B, 1, 1, 1, 1, device=device) < tf_prob
                    belief = torch.where(use_tf, gt.unsqueeze(1), sf_belief)

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
def validate(model, loader, device, num_steps, init_belief=0.0):
    """두 모드 동시 평가: self-feed (실제 inference) + teacher-forced (학습 조건).

    두 loss의 갭이 exposure bias 정도를 알려줌.
    """
    model.eval()
    total_loss_sf = 0.0
    total_loss_tf = 0.0
    total_samples = 0

    for voxel_maps, swept_maps, per_view_pre, per_view_post in tqdm(loader, desc='Val'):
        voxel_maps = voxel_maps.to(device, non_blocking=True)
        swept_maps = swept_maps.to(device, non_blocking=True)
        per_view_pre = per_view_pre.to(device, non_blocking=True)
        per_view_post = per_view_post.to(device, non_blocking=True)
        B = voxel_maps.shape[0]

        belief_sf = _init_belief(B, device, init_belief)
        belief_tf = _init_belief(B, device, init_belief)
        has_push = (swept_maps.view(B, num_steps, -1).sum(dim=2) > 0).float()
        cum_push = has_push.cumsum(dim=1).clamp(max=1.0)

        acc_pre = torch.zeros(B, 60, 120, 80, device=device)
        acc_post = torch.zeros(B, 60, 120, 80, device=device)

        with torch.amp.autocast('cuda'):
            step_loss_sf = 0.0
            step_loss_tf = 0.0
            for t in range(num_steps):
                voxel = voxel_maps[:, t:t+1]
                swept = swept_maps[:, t:t+1]

                # Self-feed forward
                x_sf = torch.cat([belief_sf, voxel, swept], dim=1)
                logit_sf = model(x_sf)

                # Teacher-forced forward
                x_tf = torch.cat([belief_tf, voxel, swept], dim=1)
                logit_tf = model(x_tf)

                # GT 계산 (두 모드 공통)
                acc_pre = acc_pre + per_view_pre[:, t].float()
                acc_post = acc_post + per_view_post[:, t].float()
                mask = cum_push[:, t].view(B, 1, 1, 1)
                gt = ((1.0 - mask) * acc_pre + mask * acc_post) / float(t + 1)

                step_loss_sf = step_loss_sf + loss_fn(logit_sf, gt)
                step_loss_tf = step_loss_tf + loss_fn(logit_tf, gt)

                # 다음 step belief 입력
                belief_sf = torch.sigmoid(logit_sf)
                belief_tf = gt.unsqueeze(1)

        total_loss_sf += step_loss_sf.item() * B
        total_loss_tf += step_loss_tf.item() * B
        total_samples += B

    return total_loss_sf / total_samples, total_loss_tf / total_samples


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
        print(f'[UA] Train: {len(train_loader.dataset)} samples, '
              f'Val: {len(val_loader.dataset)} samples')
        print(f'[UA] tf_prob_start={args.teacher_forcing_prob} '
              f'tf_anneal_to={args.tf_anneal_to} best_metric={args.best_metric}')

    model = UNet3DPush(prior=args.prior).to(device)

    # Resume: model weights만 load (clean fine-tune)
    if args.resume_from is not None:
        if rank == 0:
            print(f'[UA] Loading model weights from {args.resume_from}')
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

        train_loss = train_one_epoch(model, train_loader, optimizer, device,
                                     args.num_steps, scaler, rank, world_size,
                                     tf_prob=epoch_tf_prob,
                                     init_belief=args.init_belief)

        val_loss = None
        if rank == 0:
            raw_model = model.module if hasattr(model, 'module') else model
            val_loss_sf, val_loss_tf = validate(raw_model, val_loader, device, args.num_steps,
                                                init_belief=args.init_belief)
            # Best checkpoint 기준은 args.best_metric에 따라 결정
            val_loss = val_loss_tf if args.best_metric == 'tf' else val_loss_sf

            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss_selffeed': val_loss_sf,
                'val_loss_tf': val_loss_tf,
                'tf_prob': epoch_tf_prob,
                'lr': optimizer.param_groups[0]['lr'],
            })

            print(f'Epoch {epoch:03d} | tf_p={epoch_tf_prob:.2f} | '
                  f'Train: {train_loss:.4f} | '
                  f'Val (self-feed): {val_loss_sf:.4f} | Val (TF): {val_loss_tf:.4f}')

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
                    'val_loss_selffeed': val_loss_sf,
                    'val_loss_tf': val_loss_tf,
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
