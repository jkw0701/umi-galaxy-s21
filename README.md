# Galaxy S21 기반 UMI 프로젝트

> **원본 논문 UMI (Universal Manipulation Interface)** 와의 핵심 차이:
> - 논문: GoPro 광각 카메라 + ORB-SLAM3 + IMU 융합
> - 본 프로젝트: Galaxy **S21** (0.5배율 카메라, 좁은 시야각) + **DROID-SLAM** only, IMU 미사용

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [파이프라인 전체 흐름](#2-파이프라인-전체-흐름)
3. [작업 — 다중 오브젝트 분류 배치](#3-작업--다중-오브젝트-분류-배치)
4. [핵심 트러블슈팅 요약](#4-핵심-트러블슈팅-요약)

---

## 1. 프로젝트 개요

UMI는 스마트폰을 그리퍼에 장착해 사람이 직접 시연한 데이터를 수집하고, Diffusion Policy로 학습해 실제 로봇(Franka)이 모방하도록 하는 프레임워크입니다.

본 프로젝트는 **Galaxy S21** 을 활용해 UMI 파이프라인을 구성한 실험 기록입니다.

### 논문 vs. 본 프로젝트 비교

| 항목 | 원본 UMI (논문) | 본 프로젝트 (S21) |
|------|-----------------|------------------|
| 카메라 | GoPro (광각, 넓은 FOV) | Galaxy S21 0.5배율 (좁은 FOV) |
| SLAM | ORB-SLAM3 + IMU 융합 | DROID-SLAM only |
| IMU | 사용 (VIO) | 미사용 |
| 환경 특징점 | 풍부 (광각으로 넓은 장면) | 제한적 (테이블 너머 배경 포함 주의) |
| 추가 조치 | — | ArUco 마커 + 테이블보 스티커로 특징점 보강 |

---

## 2. 파이프라인 전체 흐름

```
[Galaxy S21 앱으로 데모 데이터 수집]
        ↓  USB → receive_server.py 로 PC에 저장
[세션 폴더 준비]  ← 에피소드 데이터 + gripper_calibration 폴더 필요
        ↓
[Step 1]  run_slam_pipeline_s21.py process-droid
        ↓  DROID-SLAM 실행 → dataset_plan.pkl 생성
[Step 2]  droid_slam_s21/07_generate_replay_buffer.py
        ↓  dataset_plan.pkl → .zarr.zip 변환 (여러 환경 병합 가능)
[서버로 .zarr.zip 복사 후 .zarr로 변환]
        ↓
[Step 3]  train.py  ← Diffusion Policy 학습
        ↓  → .ckpt 파일 생성
[Step 4]  scripts_real/eval_real_umi_ensemble.py
        ↓  Franka 실제 로봇 평가
```

### Step 0 — 데이터 수신

S21 앱으로 촬영한 데이터를 USB로 PC에 수신합니다.

```bash
# 터미널 1: USB 포트 포워딩
adb reverse tcp:8080 tcp:8080

# 터미널 2: 수신 서버 실행 (이 리포 루트에서 실행)
python receive_server.py
```

저장 위치: `~/Downloads/robotdatalearning_local/{session_id}/`

---

### Step 1 — DROID-SLAM 실행

세션 폴더 안에 에피소드 데이터와 `gripper_calibration` 폴더가 준비된 상태에서 실행합니다.

```bash
# 이 리포 루트에서 실행
python run_slam_pipeline_s21.py process-droid \
  --calibration_dir example/calibration_s21 \
  --ref aruco \
  /path/to/session_dir
```

실행 전 폴더 구조 (앱으로 수집한 원본):
```
/path/to/session_dir/
├── gripper_calibration/            ← 그리퍼 캘리브레이션 영상 (사전 촬영)
│   ├── camera_ultrawide.mp4
│   ├── frame_timestamps.csv
│   ├── metadata.json
│   └── sensor_data.jsonl
├── session_20260610_160512/        ← 앱으로 촬영한 데모 1개 (촬영 시각이 폴더명)
│   ├── camera_ultrawide.mp4
│   ├── frame_timestamps.csv
│   ├── metadata.json
│   ├── sensor_data.jsonl
│   └── sync_frames.jsonl
├── session_20260610_161926/        ← 데모 1개 (총 데모 수만큼 폴더 존재)
└── ...
```

실행 후 생성되는 파일:
```
/path/to/session_dir/
├── demos/                                          ← 파이프라인이 자동 생성
│   ├── gripper_calibration_Galaxy_S21_.../         ← 그리퍼 캘리브레이션 처리 결과
│   │   ├── raw_video.mp4
│   │   ├── tag_detection.pkl
│   │   └── gripper_range.json
│   ├── demo_Galaxy_S21_1970.01.16_11.03.33.../    ← 각 에피소드 처리 결과
│   │   ├── raw_video.mp4
│   │   ├── camera_trajectory.csv   ← SLAM 포즈 결과
│   │   ├── tag_detection.pkl       ← ArUco 검출 결과
│   │   ├── tx_slam_tag.json        ← SLAM↔태그 좌표 변환
│   │   ├── frame_timestamps.csv
│   │   ├── sensor_data.jsonl
│   │   ├── metadata.json
│   │   ├── droid_stdout.txt        ← DROID-SLAM 로그
│   │   └── droid_stderr.txt
│   ├── demo_Galaxy_S21_.../
│   └── ...
└── dataset_plan.pkl                ← Step 2 입력 파일
```

> `--calibration_dir`에는 반드시 `example/calibration_s21/` 를 지정합니다.
> S21 0.5배율 카메라의 intrinsics(`s21_intrinsics_1080p.json`)와 ArUco 설정(`aruco_config.yaml`)이 들어 있습니다.

---

### Step 2 — zarr 파일 생성

`dataset_plan.pkl`을 학습용 `.zarr.zip` 파일로 변환합니다.

**단일 환경:**
```bash
python droid_slam_s21/07_generate_replay_buffer.py \
  -o /path/to/session_dir/dataset.zarr.zip \
  /path/to/session_dir
```

**여러 환경 병합** (다환경 학습 시):
```bash
python droid_slam_s21/07_generate_replay_buffer.py \
  /path/to/session_white \
  /path/to/session_green \
  /path/to/session_wine \
  /path/to/session_recovery \
  -o /path/to/output/dataset_combined.zarr.zip
```

---

### Step 3 — 학습

로컬에서 만든 `.zarr.zip`을 학습 서버로 복사한 뒤 `.zarr`로 변환하고 학습을 실행합니다.

```bash
# .zarr.zip → .zarr 변환 (서버에서)
unzip dataset_combined.zarr.zip -d data/dataset_combined.zarr

# 학습 실행
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config-name=train_diffusion_unet_timm_umi_workspace \
  task.dataset_path=data/dataset_combined.zarr \
  training.resume=True
```

결과물: `ckpt/` 폴더에 `.ckpt` 파일 생성

---

### Step 4 — Franka 실제 로봇 평가

```bash
python scripts_real/eval_real_umi_ensemble.py \
  --robot_config=example/eval_franka_robots_config.yaml \
  --mask_mode s21 \
  -i ckpt/latest.ckpt \
  -o eval_data/
```

> `--mask_mode s21`: S21 그리퍼 하단 영역을 마스킹합니다.

---

## 3. 작업 — 다중 오브젝트 분류 배치

> 실험 세팅 및 결과는 추후 상세 작성 예정입니다.

### 작업 1 — 공·큐브 바구니 분류

큐브 2개와 공 2개를 테이블 위 랜덤한 위치에서 집어 각각의 매칭 바구니에 넣는 작업입니다.

- 큐브 2개 → 바구니 A
- 공 2개 → 바구니 B
- 오브젝트 위치, 바구니 위치 모두 랜덤

### 작업 2 — 캔·페트병 분리수거

실제 음료수 캔과 페트병을 종류별 분리수거 바구니에 넣는 작업입니다.

---

## 4. 핵심 트러블슈팅 요약

### T1. 특정 방향 이동 시 SLAM 포즈 진동

| 항목 | 내용 |
|------|------|
| **증상** | 특정 방향으로 이동하는 프레임에서 x, y, z 값이 크게 진동 |
| **원인** | 카메라 시야에 테이블 너머 먼 배경 물체가 포함됨 |
| **해결** | ① DROID-SLAM 파라미터 튜닝 ② 배경 물체 제거 ③ 포즈가 튀는 에피소드 학습 데이터에서 필터링 제외 |
| **교훈** | S21의 좁은 FOV는 배경 오염에 민감 — 작업공간 주변 정리 필수 |

### T2. 단색 테이블보에서 SLAM 특징점 부족

| 항목 | 내용 |
|------|------|
| **증상** | 민무늬 테이블보에서 DROID-SLAM feature tracking 불안정 |
| **원인** | 균일한 색상 → 특징점(keypoint) 부족 |
| **해결** | 테이블보에 다양한 모양의 스티커 부착 (화살표, 네모, 별, 십자가) |

### T3. Recovery 데모 추가로 성공률 향상

| 항목 | 내용 |
|------|------|
| **증상** | 파지 실패 후 복구 동작 없이 에피소드 실패 |
| **해결** | 실패 상황에서의 복구 동작 데모 별도 수집 후 병합 학습 |
| **효과** | 성공률 유의미하게 향상 |

---

## 참고

- **원본 UMI 논문**: [https://umi-gripper.github.io/](https://umi-gripper.github.io/)
- **DROID-SLAM**: [https://github.com/princeton-vl/DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM)
- **Franka 평가 설정**: [example/eval_franka_robots_config.yaml](example/eval_franka_robots_config.yaml)
