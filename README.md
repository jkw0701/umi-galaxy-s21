# Galaxy S21 기반 UMI 프로젝트

> **원본 논문 UMI (Universal Manipulation Interface)** 와의 핵심 차이:
> - 논문: GoPro 광각 카메라 + ORB-SLAM3 + IMU 융합
> - 본 프로젝트: Galaxy **S21** (0.5배율 카메라, 좁은 시야각) + **DROID-SLAM** only, IMU 미사용

---

## 목차

- [실행 환경 및 설치](#실행-환경-및-설치)
1. [프로젝트 개요](#1-프로젝트-개요)
2. [파이프라인 전체 흐름](#2-파이프라인-전체-흐름)
3. [작업 — 다중 오브젝트 분류 배치](#3-작업--다중-오브젝트-분류-배치)
4. [핵심 트러블슈팅 요약](#4-핵심-트러블슈팅-요약) — T1~T4

---

## 실행 환경 및 설치

### 검증된 환경

| 항목 | 버전 |
|------|------|
| GPU | NVIDIA RTX 5060 Ti (16GB) |
| GPU 드라이버 | 595.71 |
| CUDA (드라이버) | 13.2 |
| OS | Ubuntu (Linux) |

> CUDA 커널을 직접 컴파일하는 패키지(`lietorch`, `droid_backends`)가 포함되어 있어 **GPU 드라이버 및 CUDA 버전 조합이 맞아야 한다.** 위 환경에서 검증된 버전 조합이므로, 다른 GPU를 사용할 경우 torch/CUDA 버전을 별도로 맞춰야 할 수 있다.

---

### conda 환경 구성

이 프로젝트는 두 개의 conda 환경을 사용한다.

| 환경 이름 | 용도 |
|-----------|------|
| `umi` | 메인 파이프라인 전체 (데이터 수신, zarr 생성, 학습, 평가) |
| `droid` | DROID-SLAM 실행 전용 (`run_slam_pipeline_s21.py` 내부에서 자동 호출) |

`run_slam_pipeline_s21.py`는 `umi` 환경에서 실행되며, SLAM 단계에서 내부적으로 `conda run -n droid`를 통해 `droid` 환경을 자동으로 호출한다. 사용자가 직접 `droid` 환경을 활성화할 필요는 없다.

---

> **DROID-SLAM 설치 상세 가이드**: [INSTALL_DROID_SLAM.md](INSTALL_DROID_SLAM.md)
> 초기 상태에서 SLAM 실행까지의 전 과정, 발생 가능한 오류 및 해결 방법을 포함한다.

### 환경 1: `umi`

**검증된 주요 패키지 버전:**

| 패키지 | 버전 |
|--------|------|
| Python | 3.9.18 |
| torch | 2.1.0 (CUDA 12.1) |
| torchvision | 0.16.0 |
| numpy | 1.24.4 |
| scipy | 1.11.4 |
| zarr | 2.16.1 |
| opencv-python | 4.7.0 |
| av | 10.0.0 |
| hydra-core | 1.2.0 |
| timm | 0.9.7 |
| diffusers | 0.18.2 |
| einops | 0.6.1 |
| accelerate | 0.24.1 |
| wandb | 0.15.8 |

**설치:**

```bash
conda env create -f conda_environment.yaml
conda activate umi
```

---

### 환경 2: `droid` (DROID-SLAM 전용)

**검증된 주요 패키지 버전:**

| 패키지 | 버전 |
|--------|------|
| Python | 3.10.20 |
| torch | 2.11.0+cu128 |
| CUDA | 12.8 |
| lietorch | 0.2 |

`lietorch`와 `droid_backends`는 CUDA 커널을 직접 컴파일하므로, **torch 및 CUDA 버전이 정확히 일치해야 한다.**

**설치:**

```bash
# 1. DROID-SLAM 저장소 클론
git clone --recursive https://github.com/princeton-vl/DROID-SLAM.git
cd DROID-SLAM

# 2. torch 2.11.0+cu128 기준으로 환경 생성
conda create -n droid python=3.10
conda activate droid
pip install torch==2.11.0+cu128 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 3. lietorch 및 droid_backends 빌드 (CUDA 커널 컴파일 — 수 분 소요)
pip install -e thirdparty/lietorch
python setup.py install

# 4. 모델 가중치 다운로드
# droid.pth 를 DROID-SLAM/ 폴더에 저장
# 다운로드: https://drive.google.com/file/d/1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh
```

> `droid.pth` 경로는 `DROID-SLAM/droid.pth`이어야 한다. 기본값으로 이 경로를 탐색한다.

---

## 1. 프로젝트 개요

UMI는 스마트폰을 그리퍼에 장착해 사람이 직접 시연한 데이터를 수집하고, Diffusion Policy로 학습해 실제 로봇(Franka)이 모방하도록 하는 프레임워크다.

본 프로젝트는 **Galaxy S21** 을 활용해 UMI 파이프라인을 구성한 실험 기록이다.

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

S21 앱으로 촬영한 데이터를 USB로 PC에 수신한다.

```bash
# 터미널 1: USB 포트 포워딩
adb reverse tcp:8080 tcp:8080

# 터미널 2: 수신 서버 실행 (이 리포 루트에서 실행)
python receive_server.py
```

저장 위치: `~/Downloads/robotdatalearning_local/{session_id}/`

---

### Step 1 — DROID-SLAM 실행

명령어 하나로 세션 폴더 안의 **모든 에피소드에 대해 DROID-SLAM을 일괄 실행**한다. 에피소드마다 개별로 실행할 필요가 없다.

```bash
# 이 리포 루트에서 실행
python run_slam_pipeline_s21.py process-droid \
  --calibration_dir example/calibration_s21 \
  --ref aruco \
  /path/to/session_dir
```

`/path/to/session_dir` 아래의 모든 `session_*` 폴더를 자동 탐색해 순차적으로 SLAM을 수행하고, 마지막에 `dataset_plan.pkl`을 생성한다.

#### gripper_calibration 폴더

세션 폴더 안에 **`gripper_calibration`이라는 이름의 폴더가 반드시 1개** 있어야 한다. 이 폴더는 그리퍼의 열림/닫힘 범위를 보정하는 데 사용된다.

- 데이터 수집 시 **그리퍼 캘리브레이션 영상을 1회 별도로 촬영**해야 한다. 촬영 후에는 해당 폴더 이름(`session_20260605_103035` 형식)을 `gripper_calibration`으로 변경하면 된다.
- **정확히 1개만 있으면 된다.** 나머지 에피소드 폴더는 그리퍼 캘리브레이션 처리 없이 SLAM 및 데이터 처리만 수행한다.

실행 전 폴더 구조 (앱으로 수집한 원본):
```
/path/to/session_dir/
├── gripper_calibration/            ← 에피소드 폴더 중 하나를 이름 변경 (1개만 필요)
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

> `--calibration_dir`에는 반드시 `example/calibration_s21/` 를 지정한다.
> S21 0.5배율 카메라의 intrinsics(`s21_intrinsics_1080p.json`)와 ArUco 설정(`aruco_config.yaml`)이 들어 있다.

---

### Step 2 — zarr 파일 생성

`dataset_plan.pkl`을 학습용 `.zarr.zip` 파일로 변환한다.

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

로컬에서 만든 `.zarr.zip`을 학습 서버로 복사한 뒤 `.zarr`로 변환하고 학습을 실행한다.

```bash
# .zarr.zip → .zarr 이름 변경 (서버에서)
cp dataset_combined.zarr.zip data/dataset_combined.zarr

# 학습 실행
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config-name=train_diffusion_unet_timm_umi_workspace \
  task.dataset_path=data/dataset_combined.zarr \
  training.resume=True
```

결과물: `ckpt/` 폴더에 `.ckpt` 파일 생성

---

### Step 4 — Franka 실제 로봇 평가

> Franka 로봇 제어에 필요한 `zerorpc` 패키지는 `conda_environment.yaml`에 포함되어 있지 않다. 실제 로봇 평가 환경에서만 별도 설치가 필요하다.
> ```bash
> pip install zerorpc
> ```

```bash
python scripts_real/eval_real_umi_ensemble.py \
  --robot_config=example/eval_franka_robots_config.yaml \
  --mask_mode s21 \
  -i ckpt/latest.ckpt \
  -o eval_data/
```

> `--mask_mode s21`: S21 그리퍼 하단 영역을 마스킹한다.

---

## 3. 작업 — 다중 오브젝트 분류 배치

> 실험 세팅 및 결과는 추후 상세 작성 예정이다.

### 작업 1 — 공·큐브 바구니 분류

큐브 2개와 공 2개를 테이블 위 랜덤한 위치에서 집어 각각의 매칭 바구니에 넣는 작업이다.

- 큐브 2개 → 바구니 A
- 공 2개 → 바구니 B
- 오브젝트 위치, 바구니 위치 모두 랜덤

### 작업 2 — 캔·페트병 분리수거

실제 음료수 캔과 페트병을 종류별 분리수거 바구니에 넣는 작업이다.

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

### T4. DROID-SLAM 파라미터 튜닝

DROID-SLAM의 궤적 품질은 세 파라미터(filter_thresh, keyframe_thresh, warmup)에 민감하게 반응했으며, 특정 구간의 문제를 해결하면 다른 구간에서 새로운 문제가 발생하는 트레이드오프가 존재했다.

#### 각 파라미터의 역할

| 파라미터 | 역할 |
|----------|------|
| `filter_thresh` | 광학 흐름(optical flow)의 신뢰도 필터. 높일수록 신뢰도 낮은 대응점을 강하게 제거 |
| `keyframe_thresh` | 새 키프레임을 삽입하는 기준 이동량. 낮출수록 키프레임을 더 자주 생성해 빠른 움직임 구간을 촘촘하게 추적 |
| `warmup` | 본격 추적 시작 전 초기 키프레임 그래프를 구성하는 데 사용하는 프레임 수. DROID-SLAM 내부의 Bundle Adjustment(BA)가 수렴하려면 초기 그래프에 충분한 제약 조건이 필요 |

#### 튜닝 과정

**1단계 — 90–150프레임 구간 불안정 해결**

초기 파라미터(filter_thresh=2.4, keyframe_thresh=4.0, warmup=8) 상태에서 로봇이 공을 집어 이동하는 90–150프레임 구간에서 20–30개 에피소드의 Z축이 크게 튀었다.

이 구간은 EE가 테이블 바깥 방향으로 회전하며 물체를 내려놓는 동작이 이루어지는 구간이다. 카메라 시야에서 작업공간인 테이블이 차지하는 비중과 테이블 너머의 배경 공간이 차지하는 비중이 비슷해지면서, DROID-SLAM이 어느 쪽을 기준으로 위치를 추정해야 할지 혼동하여 포즈 추정값이 불안정해진 것으로 분석된다.

해결을 위해 두 가지 조치를 병행했다.

- **환경 통제**: 테이블 너머에 놓인 사물들을 제거해 배경의 특징점 오염을 차단
- **파라미터 조정**:
  - `filter_thresh` 2.4 → 3.4: 저신뢰도 광학 흐름 대응점을 더 강하게 제거
  - `keyframe_thresh` 4.0 → 1.0: 빠른 움직임 구간을 키프레임으로 더 촘촘하게 커버

→ 90–150프레임 구간 불안정 해소. 그러나 `keyframe_thresh`를 1.0으로 낮추자 초반부터 키프레임이 과도하게 생성되어 **0–50프레임 구간에서 60개 에피소드가 새롭게 불안정**해지는 문제가 발생했다.

**2단계 — 전구간 균형 조정**

`filter_thresh`=2.4, `keyframe_thresh`=2.8로 재조정해 두 구간 사이의 균형을 맞췄으나, 초반 불안정은 완전히 해소되지 않았다. 문제의 근본 원인이 다른 곳에 있음을 확인했다.

**3단계 — warmup 파라미터로 근본 해결**

`warmup`=8이면 BA가 수렴하기에 제약 조건이 너무 부족한 상태로, 초기 자세 추정이 잘못된 해로 수렴하고 이 오류가 이후 전체 프레임에 누적·전파된다. `warmup`을 8 → 26으로 높여 초기 그래프가 충분한 키프레임을 확보한 뒤 추적을 시작하게 하자 **120개 에피소드 전 구간에서 불안정 에피소드가 0개**로 완전히 해소됐다.

#### 최종 파라미터

| 파라미터 | 초기값 | 최종값 |
|----------|--------|--------|
| `filter_thresh` | 2.4 | 2.4 |
| `keyframe_thresh` | 4.0 | 2.8 |
| `warmup` | 8 | 26 |

> **핵심 교훈**: 90–150프레임 구간의 불안정은 환경 통제(배경 정리)와 filter/keyframe 파라미터 조정으로 해결했으나, 초반 구간의 근본적인 불안정은 SLAM 초기화 메커니즘인 warmup 부족에서 비롯된 것이었다. warmup을 26으로 높이는 것이 결정적인 해결책이었다.

---

## 참고

- **원본 UMI 논문**: [https://umi-gripper.github.io/](https://umi-gripper.github.io/)
- **DROID-SLAM**: [https://github.com/princeton-vl/DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM)
- **Franka 평가 설정**: [example/eval_franka_robots_config.yaml](example/eval_franka_robots_config.yaml)
