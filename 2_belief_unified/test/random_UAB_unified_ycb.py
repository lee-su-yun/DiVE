"""Random-schedule belief evolution for the UNIFIED UA+UB model trained on ycb_v3.

train_UAB_unified.py (옵션 A: shared backbone, 2-channel head) 로 학습된 모델 시각화.

Sequence: 하나의 push_X 안에서 pre_occ/{view}.npz 만으로 view-only sequence (push 없음).
GT (학습과 동일):
    UA: GT_A[t] = (1/(t+1)) Σ_{k∈K_t} pre_ray_casting[k]   (cumulative mean of per-view pre-visibility)
    UB: GT_B    = push_visibility_all_marginal.npy          (time-invariant)

Layout per row:
  [RGB | beliefA z=15,35,45 | GT_A z=15,35,45 | beliefB z=15,35,45 | GT_B z=15,35,45]
Last row: reference (v_pre_mean over all 98 views for UA, gt_marginal for UB).

명령어:
    python /home/sylee/codes/DiVE/2_belief_unified/test/random_UAB_unified_ycb.py \\
        --idx 0 --push 1 --split val --occ low --seed 0 --device 0
"""
import os
import sys
import io
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), 'support_code'))
from model_UAB_unified import UNet3DPushUnified  # noqa: E402

D, H, W = 60, 120, 80
NUM_VIEWS = 98
DEFAULT_Z_SLICES = [15, 35, 45]
NUM_STEPS = 10

DATA_ROOTS = {
    'low':  '/result/DiVE_data/beliefmap_low_occlusion_ycb_v3',
    'high': '/data/APOBU/beliefmap_high_occlusion_ycb_v3',
}
DEFAULT_CKPT = '/result/APOBU/DiVE/UAB_unified_smoke_ycb_v3_TFanneal/best.pth'
OUT_DIR = '/home/sylee/codes/DiVE/2_belief_unified/test/result'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--idx', type=int, required=True,
                   help='Index into the chosen split scene list.')
    p.add_argument('--push', type=int, default=1,
                   help='Which push_X to use (1..5).')
    p.add_argument('--split', type=str, default='val', choices=['val', 'train'])
    p.add_argument('--occ', type=str, default='low', choices=['low', 'high'])
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--val_frac', type=float, default=0.1)
    p.add_argument('--num_val_scenes', type=int, default=None)
    p.add_argument('--num_steps', type=int, default=NUM_STEPS)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--ckpt', type=str, default=DEFAULT_CKPT)
    p.add_argument('--label', type=str, default='UAB_unified_ycb_v3')
    p.add_argument('--init_a', type=float, default=0.0,
                   help='Initial belief_A value (training default 0.0).')
    p.add_argument('--init_b', type=float, default=0.0,
                   help='Initial belief_B value (training default 0.0).')
    p.add_argument('--ua_vmax', type=float, default=1.0,
                   help='vmax for UA belief/GT colormap (cum-mean of binary => [0,1]).')
    p.add_argument('--ub_vmax', type=float, default=0.2,
                   help='vmax for UB belief/GT colormap (marginal small).')
    p.add_argument('--out_suffix', type=str, default='')
    p.add_argument('--gt_overlap', action='store_true',
                   help='Also save a PNG with per-step z=15 UA/UB belief slices '
                        'overlaid with the GT object voxels (from pre_gt.npz) in black.')
    p.add_argument('--gt_overlap_z', type=int, default=15,
                   help='Z slice to use for the GT-overlap PNG (default 15).')
    return p.parse_args()


def pick_scene(idx, occ, val_frac, num_val_scenes, split):
    all_scene_names = set()
    for root in DATA_ROOTS.values():
        if not os.path.isdir(root):
            print(f'[pick_scene] skipping missing root: {root}')
            continue
        for d in os.listdir(root):
            if os.path.isdir(os.path.join(root, d)):
                all_scene_names.add(d)
    sorted_scenes = sorted(all_scene_names)
    if num_val_scenes is None:
        num_val_scenes = max(1, int(len(sorted_scenes) * val_frac))
    if split == 'val':
        candidates = sorted_scenes[-num_val_scenes:]
    else:
        candidates = sorted_scenes[:-num_val_scenes]
    if not 0 <= idx < len(candidates):
        raise ValueError(f'idx {idx} out of range (size {len(candidates)})')
    scene_name = candidates[idx]
    scene_dir = os.path.join(DATA_ROOTS[occ], scene_name)
    if not os.path.isdir(scene_dir):
        raise FileNotFoundError(scene_dir)
    print(f'[pick_scene] total={len(sorted_scenes)}, val={num_val_scenes}, '
          f'split={split} ({len(candidates)} candidates)')
    return scene_name, scene_dir


def load_model(ckpt, device):
    model = UNet3DPushUnified().to(device).eval()
    c = torch.load(ckpt, map_location=device)
    model.load_state_dict(c['model_state_dict'])
    vl_sf = c.get('val_loss_selffeed', float('nan'))
    vl_a = c.get('val_loss_a_selffeed', float('nan'))
    vl_b = c.get('val_loss_b_selffeed', float('nan'))
    print(f'  loaded {ckpt}  epoch={c["epoch"]}  '
          f'val_sf={vl_sf:.4f}  (A={vl_a:.4f}  B={vl_b:.4f})')
    return model


def load_views(occ_dir, view_indices):
    arrs = [np.load(os.path.join(occ_dir, f'{int(v):03d}.npz'))['occ']
            for v in view_indices]
    return np.stack(arrs).astype(np.float32)


def load_ray_casting_full(path):
    """Return (98, D, H, W) float32 0/1 visibility from packbits .npy."""
    packed = np.load(path)
    return np.unpackbits(packed, axis=-1).astype(np.float32)


def sample_views(num_steps, seed):
    rng = np.random.default_rng(seed)
    idx = rng.choice(NUM_VIEWS, num_steps, replace=False)
    return np.sort(idx).astype(np.int64)


@torch.no_grad()
def run_schedule(model, voxels, cam_ids, v_pre_full, gt_marginal,
                 init_a, init_b, device):
    """Run unified model for num_steps and return per-step beliefs/GTs/metrics.

    Returns:
        beliefs_a:    list of (D, H, W)  — sigmoid(logit_A) per step
        beliefs_b:    list of (D, H, W)  — sigmoid(logit_B) per step
        gts_a:        list of (D, H, W)  — cumulative mean of v_pre[K_t]
        gts_b:        list of (D, H, W)  — gt_marginal (repeated, for parity)
        entropies_a:  list of float       — Σ 2·beliefA·(1-beliefA) / (t+1)
        sums_b:       list of float       — Σ beliefB        (information-gain proxy)
    """
    num_steps = voxels.shape[0]
    belief_a = torch.full((1, 1, D, H, W), float(init_a), device=device)
    belief_b = torch.full((1, 1, D, H, W), float(init_b), device=device)

    beliefs_a, beliefs_b, gts_a, gts_b = [], [], [], []
    entropies_a, sums_b = [], []
    acc_pre = np.zeros((D, H, W), dtype=np.float32)

    for t in range(num_steps):
        voxel_t = torch.from_numpy(voxels[t]).to(device).view(1, 1, D, H, W)
        x = torch.cat([belief_a, belief_b, voxel_t], dim=1)  # (1, 3, D, H, W)
        logits = model(x)                                    # (1, 2, D, H, W)
        belief_a = torch.sigmoid(logits[:, 0:1])
        belief_b = torch.sigmoid(logits[:, 1:2])

        bA = belief_a.squeeze().cpu().numpy()
        bB = belief_b.squeeze().cpu().numpy()
        beliefs_a.append(bA)
        beliefs_b.append(bB)

        # Per-step scalar metrics (sum over all voxels)
        entropies_a.append(float((2.0 * bA * (1.0 - bA)).sum()) / float(t + 1))
        sums_b.append(float(bB.sum()))

        cam = int(cam_ids[t])
        acc_pre = acc_pre + v_pre_full[cam]
        gts_a.append((acc_pre / float(t + 1)).astype(np.float32))
        gts_b.append(gt_marginal)

    return beliefs_a, beliefs_b, gts_a, gts_b, entropies_a, sums_b


def belief_with_gt_overlay(belief_slice, gt_slice, vmax, cmap_name='viridis'):
    """Return an (H, W, 3) RGB image: belief rendered with `cmap_name`,
    pixels where gt_slice > 0.5 forced to black."""
    cmap = plt.get_cmap(cmap_name)
    normed = np.clip(belief_slice / max(float(vmax), 1e-8), 0.0, 1.0)
    rgb = cmap(normed)[..., :3].copy()
    mask = gt_slice > 0.5
    rgb[mask] = 0.0
    return rgb


def decode_rgb(jpeg_arr, cam_id):
    b = jpeg_arr[cam_id]
    raw = b.tobytes() if hasattr(b, 'tobytes') else bytes(b)
    return np.array(Image.open(io.BytesIO(raw)))


def main():
    args = parse_args()
    dev_str = args.device
    if dev_str.isdigit():
        dev_str = f'cuda:{dev_str}'
    device = torch.device(dev_str)

    # 1. Scene / push
    scene_name, scene_dir = pick_scene(
        args.idx, args.occ, args.val_frac, args.num_val_scenes, args.split)
    print(f'scene: {scene_name} ({args.occ}, {args.split}) -> {scene_dir}')

    push_dir = os.path.join(scene_dir, f'push_{args.push}')
    pre_occ_dir = os.path.join(push_dir, 'pre_occ')
    pre_rc_path = os.path.join(push_dir, 'pre_ray_casting.npy')
    gt_marginal_path = os.path.join(push_dir, 'push_visibility_all_marginal.npy')
    pre_rgb_path = os.path.join(push_dir, 'pre_rgbd.npz')
    pre_gt_path = os.path.join(push_dir, 'pre_gt.npz')
    for p in (pre_occ_dir, pre_rc_path, gt_marginal_path):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    # 2. Sample views (view-only schedule, no push)
    cam_ids = sample_views(args.num_steps, args.seed)
    print(f'sampled views: {cam_ids.tolist()}')
    voxels = load_views(pre_occ_dir, cam_ids)            # (T, D, H, W) float32

    v_pre_full = load_ray_casting_full(pre_rc_path)       # (98, D, H, W)
    gt_marginal = np.load(gt_marginal_path).astype(np.float32)  # (D, H, W)

    # 3. RGB
    if os.path.isfile(pre_rgb_path):
        pre_rgb = np.load(pre_rgb_path, allow_pickle=True)['rgb_jpeg']
    else:
        print(f'[warn] no pre_rgbd.npz at {pre_rgb_path}')
        pre_rgb = None

    # 4. Model
    print(f'[load] {args.label}')
    model = load_model(args.ckpt, device)

    # 5. Run
    beliefs_a, beliefs_b, gts_a, gts_b, entropies_a, sums_b = run_schedule(
        model, voxels, cam_ids, v_pre_full, gt_marginal,
        args.init_a, args.init_b, device)

    print('[per-step scalar metrics]')
    print('  t  | Σ2·UA·(1-UA)/(t+1) |   Σ beliefB   ')
    print('  ---+--------------------+---------------')
    for t in range(args.num_steps):
        print(f'  {t}  |     {entropies_a[t]:>12.4f}   |  {sums_b[t]:>12.4f}')

    # 6. Figure
    z_slices = list(DEFAULT_Z_SLICES)
    nZ = len(z_slices)
    num_rows = args.num_steps + 1            # +1 for reference row
    num_cols = 1 + nZ + nZ + nZ + nZ          # RGB | bA | gtA | bB | gtB
    fig, axes = plt.subplots(num_rows, num_cols,
                             figsize=(2.2 * num_cols, 2.4 * num_rows))

    col_rgb = 0
    cols_bA = [1 + j for j in range(nZ)]
    cols_gA = [1 + nZ + j for j in range(nZ)]
    cols_bB = [1 + 2 * nZ + j for j in range(nZ)]
    cols_gB = [1 + 3 * nZ + j for j in range(nZ)]

    for t in range(args.num_steps):
        cam_id = int(cam_ids[t])
        bA = beliefs_a[t]
        bB = beliefs_b[t]
        gA = gts_a[t]
        gB = gts_b[t]

        ax = axes[t, col_rgb]
        if pre_rgb is not None:
            ax.imshow(decode_rgb(pre_rgb, cam_id))
        ax.set_title(
            f't={t}  cam {cam_id}  |K_t|={t+1}\n'
            f'ΣbA={bA.sum():.1f}  ΣgA={gA.sum():.1f}\n'
            f'ΣbB={bB.sum():.1f}  ΣgB={gB.sum():.1f}\n'
            f'Σ2·UA(1-UA)/(t+1)={entropies_a[t]:.2f}\n'
            f'Σ beliefB        ={sums_b[t]:.2f}',
            fontsize=7)
        ax.axis('off')

        for j, z in enumerate(z_slices):
            ax = axes[t, cols_bA[j]]
            ax.imshow(bA[z, :, :], cmap='viridis', vmin=0,
                      vmax=args.ua_vmax, aspect='auto')
            ax.set_title(f'beliefA z={z}', fontsize=7)
            ax.axis('off')

        for j, z in enumerate(z_slices):
            ax = axes[t, cols_gA[j]]
            ax.imshow(gA[z, :, :], cmap='viridis', vmin=0,
                      vmax=args.ua_vmax, aspect='auto')
            ax.set_title(f'GT_A z={z}', fontsize=7)
            ax.axis('off')

        for j, z in enumerate(z_slices):
            ax = axes[t, cols_bB[j]]
            ax.imshow(bB[z, :, :], cmap='viridis', vmin=0,
                      vmax=args.ub_vmax, aspect='auto')
            ax.set_title(f'beliefB z={z}', fontsize=7)
            ax.axis('off')

        for j, z in enumerate(z_slices):
            ax = axes[t, cols_gB[j]]
            ax.imshow(gB[z, :, :], cmap='viridis', vmin=0,
                      vmax=args.ub_vmax, aspect='auto')
            ax.set_title(f'GT_B z={z}', fontsize=7)
            ax.axis('off')

    # Reference row: v_pre_mean (UA upper bound) and gt_marginal (UB GT itself)
    v_pre_mean = v_pre_full.mean(axis=0)
    row = args.num_steps
    ax = axes[row, col_rgb]
    ax.text(0.5, 0.5,
            f'REFERENCE\n\nUA: mean over 98 views\nΣ={v_pre_mean.sum():.1f}\n\n'
            f'UB: gt_marginal\nΣ={gt_marginal.sum():.1f}',
            ha='center', va='center', fontsize=9, transform=ax.transAxes)
    ax.axis('off')

    for j in range(nZ):
        axes[row, cols_bA[j]].axis('off')
    for j, z in enumerate(z_slices):
        ax = axes[row, cols_gA[j]]
        ax.imshow(v_pre_mean[z, :, :], cmap='viridis', vmin=0,
                  vmax=args.ua_vmax, aspect='auto')
        ax.set_title(f'v_pre_mean z={z}', fontsize=7)
        ax.axis('off')

    for j in range(nZ):
        axes[row, cols_bB[j]].axis('off')
    for j, z in enumerate(z_slices):
        ax = axes[row, cols_gB[j]]
        ax.imshow(gt_marginal[z, :, :], cmap='viridis', vmin=0,
                  vmax=args.ub_vmax, aspect='auto')
        ax.set_title(f'gt_marginal z={z}', fontsize=7)
        ax.axis('off')

    plt.suptitle(
        f'{args.label} random schedule (view-only, no push)  '
        f'init_a={args.init_a} init_b={args.init_b}  '
        f'vmaxA={args.ua_vmax} vmaxB={args.ub_vmax}\n'
        f'{scene_name} ({args.occ}, {args.split})  push_{args.push}  seed={args.seed}\n'
        f'GT_A_t = (1/(t+1)) Σ_{{k∈K_t}} v_pre[k]       '
        f'GT_B = push_visibility_all_marginal.npy (time-invariant)',
        fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(OUT_DIR, exist_ok=True)
    out_png = os.path.join(
        OUT_DIR,
        f'random_UAB_unified_{scene_name}_{args.occ}_{args.split}_'
        f'push{args.push}_seed{args.seed}{args.out_suffix}.png')
    plt.savefig(out_png, dpi=100, bbox_inches='tight')
    print(f'[viz] saved -> {out_png}')

    # 7. Optional: per-step z=15 belief slices with GT object overlaid (black)
    if args.gt_overlap:
        if not os.path.isfile(pre_gt_path):
            print(f'[gt_overlap] skipped: missing {pre_gt_path}')
        else:
            gt_obj = np.load(pre_gt_path)['gt'].astype(np.float32)  # (D, H, W) 0/1
            z = int(args.gt_overlap_z)
            if not 0 <= z < gt_obj.shape[0]:
                raise ValueError(f'--gt_overlap_z {z} out of range [0,{gt_obj.shape[0]})')
            gt_slice = gt_obj[z, :, :]

            fig2, axes2 = plt.subplots(args.num_steps, 2,
                                       figsize=(2.4 * 2, 2.4 * args.num_steps))
            if args.num_steps == 1:
                axes2 = np.array([axes2])
            for t in range(args.num_steps):
                bA_img = belief_with_gt_overlay(
                    beliefs_a[t][z, :, :], gt_slice, args.ua_vmax)
                bB_img = belief_with_gt_overlay(
                    beliefs_b[t][z, :, :], gt_slice, args.ub_vmax)
                ax = axes2[t, 0]
                ax.imshow(bA_img, aspect='auto')
                ax.set_title(f't={t}  cam {int(cam_ids[t])}  '
                             f'beliefA z={z} (+GT)', fontsize=8)
                ax.axis('off')
                ax = axes2[t, 1]
                ax.imshow(bB_img, aspect='auto')
                ax.set_title(f't={t}  cam {int(cam_ids[t])}  '
                             f'beliefB z={z} (+GT)', fontsize=8)
                ax.axis('off')

            plt.suptitle(
                f'{args.label}  GT-overlay (black = pre_gt object) z={z}\n'
                f'{scene_name} ({args.occ}, {args.split})  push_{args.push}  '
                f'seed={args.seed}  vmaxA={args.ua_vmax} vmaxB={args.ub_vmax}',
                fontsize=11)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            out_png_gt = os.path.join(
                OUT_DIR,
                f'random_UAB_unified_{scene_name}_{args.occ}_{args.split}_'
                f'push{args.push}_seed{args.seed}{args.out_suffix}_gtoverlap_z{z}.png')
            plt.savefig(out_png_gt, dpi=100, bbox_inches='tight')
            print(f'[viz] saved -> {out_png_gt}')


if __name__ == '__main__':
    main()
