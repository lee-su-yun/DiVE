# DiVE: Distilled View / Push policies for Active Perception

3D voxel beliefmap 위에서 동작하는 active perception 파이프라인.
두 개의 분리된 정책을 학습한다:

- **DiVE-V** (View policy): UA belief 모델로부터 distillation. "다음에 어느 카메라 view를 볼 것인가" — 98 후보.
- **DiVE-P** (Push policy): UB belief 모델로부터 distillation. "다음에 어느 push action을 할 것인가" — 29 후보.

각 belief 모델은 `(belief, voxel, swept_map) → 새 belief` 의 3D UNet (`UNet3DPush`).
RewardNet은 `belief → 후보별 gain logit` 으로 belief 모델 N번 forward 비용을 한 번으로 줄이는 distillation 모델.

---

## Prerequisites

- Python 3.10+
- CUDA-enabled GPU (학습/리워드 생성 모두 GPU 가정; `torch.amp` 기반)
- 시뮬레이터 / 데이터 수집 코드는 별도 (이 레포는 학습·리워드 distillation만 다룸)

```bash
conda create -n DiVE python=3.10 -y
conda activate DiVE
pip install -r requirements.txt
```

---

## 폴더 구조

```
DiVE/
├── 1_preprocessing/    # 원본 시뮬 데이터 → 학습용 voxel/visibility/log
│   ├── convert_fallen_log.py
│   ├── data_preprocessing.py
│   ├── making_push_visibility_per_view_all.py
│   └── making_ray_casting_per_view.py
├── 2_belief/           # belief 모델 학습 (UNet3DPush)
│   ├── train_UA.py     # DiVE-V backbone (cumulative-mean GT)
│   └── train_UB.py     # DiVE-P backbone (push_visibility_all GT)
├── 3_reward_dataset/   # 학습된 belief 모델 rollout으로 reward dataset 생성
│   ├── UA_reward_generate.py    # 98 cam 후보, gain = Σ |Δbelief| (L1)
│   └── UB_reward_generate.py    # 29 push 후보, gain = Σ |Δbelief| (L1)
├── 4_policy/           # RewardNet 학습 (belief → 후보 logits)
│   ├── train_reward_UA.py  # DiVE-V policy
│   └── train_reward_UB.py  # DiVE-P policy
├── support_code/       # model / loss / dataset (로직은 원본과 동일)
│   ├── model_UAB.py        # UNet3DPush
│   ├── loss_UAB.py         # soft_bce_loss
│   ├── dataset_UAB.py      # UA/UB 통합 dataset (push/post 스케줄링 + fallen log 파싱)
│   ├── dataset_UA.py       # UA cumulative-mean GT용 dataset
│   ├── dataset_reward.py   # UA reward dataset
│   ├── dataset_reward_UB.py
│   ├── model_reward.py     # RewardNet (98 cam logit)
│   ├── model_reward_UB.py  # RewardNetUB (29 push logit)
│   └── gen_visibility.py   # ray casting (camera intrinsics + voxel grid 상수 포함)
└── README.md (이 파일)
```

---

## 컨벤션

- **init_belief = 0.0** 로 통일. 모든 학습 / reward 생성 단계에서 동일하게 사용해야 train/inference mismatch가 없다.
  - `train_UA.py`: `--init_belief 0.0` 명시 필요 (default 0.0)
  - `train_UB.py`: `--init_belief 0.0` 명시 필요 (default 0.0)
  - `UA_reward_generate.py`: `--init_belief 0.0` 명시 필요 (default 0.0)
  - `UB_reward_generate.py`: `--init_belief 0.0` 명시 필요 (default 0.0)
- **fallen / tipped log 처리**: `fallen_log_revised.txt`, `tipped_log_revised.txt` 모두 적용 (UA/UB GT 생성, reward dataset filtering)
- **데이터 root**: 멀티 root 지원 (`--data_roots r1 r2 ...`). 보통 low_occlusion + high_occlusion 두 개를 함께.

---

## 실행 순서

각 스크립트는 자체 docstring에도 동일 명령어가 있다. 아래는 전체 흐름.

### Stage 1 — 데이터 전처리

원본 시뮬 데이터에 대해 다음을 차례로 실행. `0423` suffix는 그날 수집본 예시.

```bash
conda activate DiVE
cd 1_preprocessing

# (사전 단계) fallen_log → fallen_log_revised, tipped_log → tipped_log_revised
# 스크립트 안 INPUT/OUTPUT 경로를 데이터셋마다 직접 수정 후 실행.
python convert_fallen_log.py

# 1) OCC 풀어 저장 + pose + visibility (depth-buffer ray casting)
sudo -E python data_preprocessing.py \
    --data-root /data/APOBU/beliefmap_low_occlusion_0423 --workers 32
sudo -E python data_preprocessing.py \
    --data-root /data/APOBU/beliefmap_high_occlusion_0423 --workers 32

# 2) view별 binary visibility (98, 60, 120, 10) packbits 저장
sudo -E python making_ray_casting_per_view.py \
    --data-roots /data/APOBU/beliefmap_low_occlusion_0423 \
                 /data/APOBU/beliefmap_high_occlusion_0423 \
    --workers 32

# 3) UB GT용 scalar push_visibility_all (XOR 기반)
sudo -E python making_push_visibility_per_view_all.py \
    --data-roots /data/APOBU/beliefmap_low_occlusion_0423 \
                 /data/APOBU/beliefmap_high_occlusion_0423 \
    --workers 32
```

생성물 (각 `push_X` 폴더):
- `pre_occ/{000..097}.npz`, `post_NN/post_occ/{000..097}.npz` — view별 occupancy
- `pose.npy`, `pre_visibility.npy`, `post_NN/post_visibility.npy` — pose + view-mean visibility
- `pre_ray_casting.npy`, `post_NN/post_ray_casting.npy` — view별 binary visibility (packbits)
- `push_visibility_all.npy` — UB GT scalar map

### Stage 2 — Belief 모델 학습

```bash
conda activate DiVE
cd 2_belief

# DiVE-V backbone (UA): cumulative-mean GT, init_belief=0.0
sudo -E python train_UA.py \
    --data_roots /data/APOBU/beliefmap_low_occlusion_0423 \
                 /data/APOBU/beliefmap_high_occlusion_0423 \
    --save_dir /result/APOBU/U_AB_beliefmap/UA_Kt_v1 \
    --wandb_run_name UA_Kt_cumulative_TF \
    --init_belief 0.0 \
    --device 2,3 \
    --teacher_forcing

# DiVE-P backbone (UB): push_visibility_all GT, init_belief=0.0
sudo -E python train_UB.py \
    --data_roots /data/APOBU/beliefmap_low_occlusion_0423 \
                 /data/APOBU/beliefmap_high_occlusion_0423 \
    --save_dir /result/APOBU/U_AB_beliefmap/UB_v1 \
    --wandb_run_name UB_v1 \
    --init_belief 0.0 \
    --device 0,1 --epochs 50 --patience 5
```

산출물: 각 `--save_dir/best.pth` (state_dict + epoch + val_loss).

### Stage 3 — Reward dataset 생성

학습된 belief 모델로 `(current_map, gains[N후보])` sample 만들기.

```bash
conda activate DiVE
cd 3_reward_dataset

# DiVE-V: 98 cam 후보 / gain = L1 of belief 변화
python UA_reward_generate.py \
    --data_roots /data/APOBU/beliefmap_low_occlusion_0423 \
                 /data/APOBU/beliefmap_high_occlusion_0423 \
    --ua_checkpoint /result/APOBU/U_AB_beliefmap/UA_Kt_v1/best.pth \
    --out_root /data/APOBU/U_AB_beliefmap/ua_Kt_reward_dataset \
    --init_belief 0.0 \
    --device 0,1,2,3 --chunk 128

# DiVE-P: 29 push 후보 / gain = L1 of belief 변화
python UB_reward_generate.py \
    --data_roots /data/APOBU/beliefmap_low_occlusion_0423 \
                 /data/APOBU/beliefmap_high_occlusion_0423 \
    --ub_checkpoint /result/APOBU/U_AB_beliefmap/UB_v1/best.pth \
    --out_root /data/APOBU/U_AB_beliefmap/ub_reward_dataset \
    --init_belief 0.0 \
    --device 0,1,2,3 --samples_per_scene 64
```

산출물 (sample 단위 npz):
- `current_map`: (1, 60, 120, 80) fp16 — k개 prior cam rollout 후 belief
- `gains`: (98,) 또는 (29,) fp32 — 후보별 gain
- `k`: int32, `prior_cams`: (k,) int32, (UB만) `post_views`: (29,) int32

### Stage 4 — RewardNet (정책) 학습

```bash
conda activate DiVE
cd 4_policy

# DiVE-V policy
python train_reward_UA.py \
    --reward_roots /data/APOBU/U_AB_beliefmap/ua_Kt_reward_dataset \
    --save_dir /result/APOBU/U_AB_beliefmap/reward_Kt_v1 \
    --wandb_run_name reward_Kt_v1_sigz \
    --loss_type bce_sigmoid_z --temperature 1.0 \
    --device 0,1,2,3

# DiVE-P policy (--data_roots 추가 필요: swept_maps on-the-fly 로드용)
python train_reward_UB.py \
    --reward_roots /data/APOBU/U_AB_beliefmap/ub_reward_dataset \
    --data_roots /data/APOBU/beliefmap_low_occlusion_0423 \
                 /data/APOBU/beliefmap_high_occlusion_0423 \
    --save_dir /result/APOBU/U_AB_beliefmap/reward_UB_v1 \
    --wandb_run_name reward_UB_v1_sigz \
    --loss_type bce_sigmoid_z --temperature 1.0 \
    --device 0,1,2,3
```

Loss 옵션 (둘 다 공통):
- `bce_sigmoid_z` (default, 권장): per-sample z-score → sigmoid → BCE. heavy-tail에 안전.
- `bce_topk`: 상위 K vs 나머지 hard label. ranking 정보 손실.
- `kl`: softmax(normalize(g)/T) → KL. `--target_norm`, `--log1p_target`, `--temperature` 같이 튜닝.

모니터링 (wandb):
- `val_top1` / `val_top3`: argmax(gains) 적중률
- `val_p_at_K`, `val_p_at_10`: top-K 교집합 비율
- `val_pred_entropy`: ln(N후보) 가 uniform, 0 근처면 collapse 의심

---

## 데이터 경로 요약

| 단계 | 경로 |
|---|---|
| 시뮬 raw / 전처리 결과 | `/data/APOBU/beliefmap_{low,high}_occlusion_0423/` |
| UA reward dataset | `/data/APOBU/U_AB_beliefmap/ua_Kt_reward_dataset/` |
| UB reward dataset | `/data/APOBU/U_AB_beliefmap/ub_reward_dataset/` |
| Belief / RewardNet 체크포인트 | `/result/APOBU/U_AB_beliefmap/{run_name}/best.pth` |
