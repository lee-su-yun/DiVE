# DiVE: Learning Decomposed Visibility for Efficient Active Exploration of Cluttered Scenes

Official code for **DiVE**, Conference on Robot Learning (CoRL) 2026.

[Project page](https://lee-su-yun.github.io/DiVE/) &nbsp;·&nbsp;
[Checkpoints](https://huggingface.co/leesuyun/DiVE) &nbsp;·&nbsp;
Paper (coming soon)

DiVE predicts two decomposed visibility maps from a partial observation — a **view-resolvable**
map (`phi_view`, what a viewpoint change could reveal) and a **push-resolvable** map
(`phi_push`, what a push could reveal). Both are converted to a common expected-visibility-change
scale, so the planner first chooses between looking and pushing, then ranks candidates within the
chosen mode with a lightweight policy. No candidate-wise post-action belief prediction, and no
hand-tuned switching threshold.

## Repository layout

The directories follow the order of the pipeline:

| Directory | Step | Contents |
| --- | --- | --- |
| `data_generation/` | 1 | Isaac Sim scene generation and rollout collection (scripts in `inference/sim/`) |
| `preprocessing/visibility/` | 2a | Ray casting and per-view / marginal visibility ground truth |
| `preprocessing/reward_dataset/` | 2b | Reward datasets for both policies, from a trained DiVE checkpoint |
| `dive/` | 3 | Training the decomposed visibility estimator (`f_DiVE`) |
| `policy/view/`, `policy/push/` | 4 | Training the viewpoint and push selection policies |
| `belief_map/` | 5 | Training the recursive semantic belief estimator (`f_scene`) |
| `inference/` | 6 | Isaac Lab evaluation: DiVE planner + belief estimator, metrics, aggregation |
| `common/` | — | Shared models, datasets and losses |
| `scripts/` | — | Checkpoint download |

The voxel grid is `(D, H, W) = (60, 120, 80)` at 5 mm resolution, with 98 candidate viewpoints on a
7x14 grid and 20 candidate pushes.

## Setup

```bash
git clone https://github.com/lee-su-yun/DiVE.git
cd DiVE
pip install -r requirements.txt
```

Steps 2-5 need only PyTorch. Step 6 (`inference/`) runs inside **Isaac Lab / Isaac Sim 5.1**,
installed separately per NVIDIA's instructions; `trimesh` and optionally `cupy` are used there too.

Download the released checkpoints:

```bash
bash scripts/download_checkpoints.sh          # -> checkpoints/
```

| File | Paper notation | Role |
| --- | --- | --- |
| `DiVE_decomposed_visibility_estimator/best.pth` | `f_DiVE` | View- and push-resolvable visibility maps |
| `view_selection_policy/best.pth` | `pi_view` | Ranks the 98 candidate viewpoints |
| `push_selection_policy/best.pth` | `pi_push` | Ranks the candidate pushes |
| `belief_map_estimator/estimator.pt` | `f_scene` | Recursive semantic belief updater |

## Evaluation

Reproduces the reference run in [`inference/reference_results/REPORT.md`](inference/reference_results/REPORT.md)
(100 episodes, 25-step budget, low occlusion): mIoU 0.9279, occupancy IoU 0.8721,
semantic mIoU 0.8116, 4.92 pushes per episode.

```bash
TEST_SET=/path/to/low_occlusion_test_v1 PY=<isaac lab python> bash inference/run_eval.sh
# -> experiments/low_v1_recursive_step25/REPORT.md   (4 shards in parallel, ~2 h)
```

The high- and extreme-occlusion test sets are not on the Hub. Generate them with the collection
scripts (seed 42, 100 episodes each, see [`data_generation/README.md`](data_generation/README.md)),
then run, for example:

```bash
SCENE_MODE=high TEST_SET=data/high_occlusion_test_v1 PY=<isaac lab python> bash inference/run_eval.sh
```

Single-episode smoke test:

```bash
export PYTHONPATH=$PWD
python -m inference.run_eval \
    --uab_belief   checkpoints/DiVE_decomposed_visibility_estimator/best.pth \
    --ua_reward    checkpoints/view_selection_policy/best.pth \
    --ub_reward    checkpoints/push_selection_policy/best.pth \
    --central_ckpt checkpoints/belief_map_estimator/estimator.pt \
    --n_sem 12 --n_sem_model 15 --scene_mode low --action_budget 25 \
    --test_set_dir /path/to/low_occlusion_test_v1 \
    --episode_start 0 --episode_end 1 --num_episodes 1 \
    --out_dir experiments/smoke --headless --seed 0
python -m inference.aggregate --run_dir experiments/smoke
```

## Data

Scene data is on the Hugging Face Hub at
[`leesuyun/DiVE-data`](https://huggingface.co/datasets/leesuyun/DiVE-data), one `.tar.zst`
archive per scene:

| Path | Contents |
| --- | --- |
| `scenes/low/` | 320 low-occlusion training scenes |
| `scenes/high/` | 320 high-occlusion training scenes |
| `test/low_occlusion_test_v1.tar.zst` | 100 held-out low-occlusion test episodes used for the evaluation above |

The archives hold only what Isaac Sim wrote. Per-view occupancy, ray casting, the visibility
targets and the two reward datasets are derived data: regenerate them with steps 1 and 3 below
instead of downloading them.

```bash
huggingface-cli download leesuyun/DiVE-data --repo-type dataset --local-dir dive_data
mkdir -p data/low
for f in dive_data/scenes/low/*.tar.zst; do tar -I zstd -xf "$f" -C data/low; done
cp dive_data/scenes/low/{fallen,tipped}_log.txt data/low/
```

The extracted layout is documented in [`data_generation/README.md`](data_generation/README.md).

## Training from scratch

Every script takes its input and output paths as arguments — nothing is hardcoded. `<data_root>`
below is a dataset directory produced by step 1 (see [`data_generation/README.md`](data_generation/README.md)
for the expected layout).

### 1. Visibility ground truth

```bash
bash preprocessing/visibility/preprocessing_run_all.sh <data_root> [<data_root> ...]
```

which runs, in order: the fallen/tipped log conversion, `data_preprocessing.py` (unpack the
per-view occupancy and write `pose.npy`), `making_ray_casting_per_view.py` (per-view visibility by
ray casting, before and after every candidate push), and
`making_push_visibility_per_view_all_marginal.py` (the push-resolvable target `phi_push`). Each
script also runs on its own; pass `--help` for the arguments.

### 2. Decomposed visibility estimator

```bash
UAB_DEVICE=0,1 bash dive/train_UAB_unified.sh ycb_v3 <data_root> [<data_root> ...]
```

Input `(B, 3, 60, 120, 80)` = `[phi_view_t, phi_push_t, voxel_t]`, output `(B, 2, 60, 120, 80)`
logits for the two maps, trained with teacher-forcing annealing. Set `SAVE_DIR` to change the
output directory (default `runs/`).

### 3. Reward datasets

One unified rollout produces the training targets for both policies:

```bash
python preprocessing/reward_dataset/UAB_reward_generate_gt.py \
    --data_roots <data_root> [<data_root> ...] \
    --uab_checkpoint runs/UAB_unified_.../best.pth \
    --out_root_ua /path/to/data/ua_reward_dataset \
    --out_root_ub /path/to/data/ub_reward_dataset
```

### 4. Policies

```bash
# viewpoint selection (pi_view)
python policy/view/train_reward_UA.py \
    --reward_roots /path/to/data/ua_reward_dataset \
    --save_dir runs/view_policy --wandb_run_name view_policy \
    --loss_type bce_sigmoid_z --temperature 1.0 --use_seen_mask --device 0

# push selection (pi_push)
python policy/push/train_reward_UB.py \
    --reward_roots /path/to/data/ub_reward_dataset \
    --data_roots <data_root> [<data_root> ...] \
    --save_dir runs/push_policy --wandb_run_name push_policy \
    --loss_type bce_sigmoid_z --temperature 1.0 --beta 0.5 \
    --use_attn --attn_heads 4 --max_steps_per_epoch 3000 --max_val_steps 500 \
    --batch_size 8 --epochs 50 --patience 6 --device 0,1
```

### 5. Belief map estimator

`belief_map/train_finetune.py` finetunes the released `estimator.pt` on new data. The from-scratch
trainer is being prepared for release.

```bash
export PYTHONPATH=$PWD
python -m belief_map.train_finetune \
    --init_from checkpoints/belief_map_estimator/estimator.pt \
    --data_dirs <data_root> [<data_root> ...] \
    --out_dir runs --out_name estimator_new.pt --batch 16 --num_workers 8
```

Input 34 channels = `[alpha/50, beta/50, occ, swept_map, one_hot(sem_obs) x15, sem_belief x15]`;
output 17 = `[alpha, beta, sem_logits x15]`. YCB v3 uses classes 0-11, so evaluation runs with
`--n_sem 12 --n_sem_model 15`.

## Citation

```bibtex
@inproceedings{lee2026dive,
  title     = {Learning Decomposed Visibility for Efficient Active Exploration of Cluttered Scenes},
  author    = {Lee, Suyun and Choi, Minsoo and Gong, Jihwan and Nam, Unghui and Bae, Minji and Shim, Byonghyo},
  booktitle = {Conference on Robot Learning (CoRL)},
  year      = {2026},
}
```

## License

MIT, see [LICENSE](LICENSE).
