# Data generation

Isaac Sim scene generation and rollout collection for the cluttered-shelf dataset. The collection
scripts double as the scene definitions that `inference/` evaluates in, so they live in
`inference/sim/`:

| Script | Split | Active objects |
| --- | --- | --- |
| `inference/sim/scene_low_occlusion.py` | low occlusion (training and test) | 11 |
| `inference/sim/scene_high_occlusion.py` | high occlusion (training and test) | 11 |
| `inference/sim/scene_super_high_occlusion.py` | extreme occlusion (test only) | 12 |

They need Isaac Lab / Isaac Sim 5.1. Each episode is 5 subscenes; every subscene renders all 98
candidate viewpoints before and after each of its 20 candidate pushes, and one valid push is
carried over as the next subscene's starting state.

```bash
export PYTHONPATH=$PWD

# training scenes
python -m inference.sim.scene_low_occlusion  --num_envs 16 --num_episodes 320 \
    --save_path data/beliefmap_low_occlusion_ycb_v3 --headless
python -m inference.sim.scene_high_occlusion --num_envs 16 --num_episodes 320 \
    --save_path data/beliefmap_high_occlusion_ycb_v3 --headless

# held-out test sets: 100 episodes each, seed 42 (the defaults), as used in the paper.
# Only low_occlusion_test_v1 is on the Hub; generate the other two here.
python -m inference.sim.scene_high_occlusion --test_set_mode \
    --save_path data/high_occlusion_test_v1 --headless
python -m inference.sim.scene_super_high_occlusion --test_set_mode \
    --save_path data/super_high_occlusion_test_v1 --headless

# shape-OOD: scale each object's two horizontal axes by U[0.8, 1.2], height unchanged
python -m inference.sim.scene_low_occlusion --test_set_mode --shape_ood \
    --save_path data/low_occlusion_shape_ood_test --headless
```

Pass `--help` for the remaining arguments. You only need these to generate new scenes: the
released ones are on the Hub (see the main README).

Expected layout of a data root, as extracted from the
[`leesuyun/DiVE-data`](https://huggingface.co/datasets/leesuyun/DiVE-data) archives:

```
<data_root>/
├── fallen_log.txt               objects that fell during a push
├── tipped_log.txt               objects that tipped over
└── {scene:09d}/
    ├── episode_log.txt
    └── push_{1..5}/
        ├── camera_poses.npz     98 candidate camera poses
        ├── pre_occ.npz          per-view occupancy, stacked
        ├── pre_semantic_occ.npz per-view semantic labels
        ├── pre_gt.npz           ground-truth occupancy
        ├── semantic_map.npz     ground-truth semantics
        └── post_{00..19}/       the same, after executing candidate push NN
            ├── occ.npz
            ├── post_semantic_occ.npz
            ├── gt.npz
            └── action.npz       push parameters and swept volume
```

`preprocessing/visibility/preprocessing_run_all.sh` adds the derived files next to these: the
unpacked per-view `pre_occ/` and `post_occ/` directories, `pose.npy`, the ray-casting maps and the
push-resolvable target.

The voxel grid is `(D, H, W) = (60, 120, 80)` at 5 mm resolution, with 98 candidate viewpoints
arranged on a 7x14 grid.
