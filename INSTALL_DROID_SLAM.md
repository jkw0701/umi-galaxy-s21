# 설치 가이드

완전 초기 상태에서 파이프라인 실행까지의 전 과정을 기술한다.

설치 단계는 두 종류로 나뉜다:
- **수동 설치** — sudo 권한이 필요하거나 시스템 환경에 따라 다르게 처리해야 하는 단계
- **스크립트 자동화** — `bash install_*.sh` 하나로 완료되는 단계

---

## 목차

1. [사전 요구사항 (수동)](#1-사전-요구사항-수동)
2. [환경 구성 (스크립트)](#2-환경-구성-스크립트)
3. [실행 테스트](#3-실행-테스트)
4. [트러블슈팅](#트러블슈팅)

---

## 1. 사전 요구사항 (수동)

### 검증된 환경

| 항목 | 버전 |
|------|------|
| OS | Ubuntu 22.04 |
| GPU | NVIDIA RTX 5060 Ti (16GB) |
| GPU 드라이버 | 595.71 |
| CUDA (드라이버) | 13.2 |
| CUDA Toolkit | 12.8 |

> CUDA 커널을 직접 컴파일하는 패키지(`lietorch`, `droid_backends`)가 포함되어 있어 **GPU 드라이버 및 CUDA 버전 조합이 맞아야 한다.** 다른 GPU를 사용할 경우 torch/CUDA 버전을 별도로 맞춰야 할 수 있다.

---

### 1-1. GPU 드라이버 및 CUDA 확인

```bash
nvidia-smi                               # GPU 드라이버 및 지원 CUDA 버전 확인
find /usr/local -name nvcc 2>/dev/null   # CUDA Toolkit(컴파일러) 설치 여부 확인
find /opt -name nvcc 2>/dev/null
```

- `nvidia-smi`가 없는 경우 → GPU 드라이버가 설치되지 않은 것이다. 드라이버를 먼저 설치한다.
- `find` 결과가 없는 경우 → CUDA Toolkit이 설치되지 않은 것이다. 1-2로 이동한다.
- `find` 결과가 있는 경우 → 설치 완료. **`install_droid_env.sh`가 이 경로를 자동으로 찾아서 사용하므로 그대로 다음 단계로 넘어간다.**

자동 감지에 실패하는 경우에만 다음과 같이 명시적으로 지정하고 실행한다:

```bash
export CUDA_HOME=/usr/local/cuda-12.8   # find 결과로 확인한 경로
bash install_droid_env.sh
```

---

### 1-2. CUDA Toolkit 설치 (미설치 시만)

`find /usr/local -name nvcc` 결과가 있으면 이 단계를 건너뛴다.

아래는 Ubuntu 22.04 + x86_64 기준이다. 다른 환경은 [CUDA Toolkit 공식 페이지](https://developer.nvidia.com/cuda-downloads)에서 OS/아키텍처에 맞는 명령어를 확인한다.

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-8
```

설치 후 확인:
```bash
find /usr/local -name nvcc 2>/dev/null   # 경로가 출력되면 설치 완료
```

---

### 1-3. Miniforge 설치 (미설치 시만)

`conda --version`이 작동하면 이 단계를 건너뛴다.

Anaconda 대신 Miniforge를 권장한다. 패키지 해결 속도가 빠르고 conda-forge 채널을 기본으로 사용한다.

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
```

설치 후 터미널을 재시작하거나 `source ~/.bashrc`를 실행한다.

---

### 1-4. 이 리포지토리 클론

```bash
git clone https://github.com/jkw0701/umi-galaxy-s21.git ~/umi-galaxy-s21
cd ~/umi-galaxy-s21
```

---

### 1-5. DROID-SLAM 클론

```bash
git clone --recursive https://github.com/princeton-vl/DROID-SLAM.git ~/DROID-SLAM
```

> `--recursive` 플래그가 반드시 필요하다. `thirdparty/lietorch` 등 서브모듈이 함께 클론된다.
> 빠뜨린 경우: `cd ~/DROID-SLAM && git submodule update --init --recursive`

---

### 1-6. 모델 가중치 복사

`droid.pth`는 이 리포지토리에 포함되어 있다 (Git LFS로 관리). `git clone` 시 자동으로 다운로드된다.

DROID-SLAM 실행 시 `~/DROID-SLAM/droid.pth` 경로를 참조하므로, clone 후 한 번만 복사한다:

```bash
cp ~/umi-galaxy-s21/droid.pth ~/DROID-SLAM/droid.pth
```

---

## 2. 환경 구성 (스크립트)

이 프로젝트는 두 개의 conda 환경을 사용한다.

| 환경 | 용도 | 설치 스크립트 |
|------|------|--------------|
| `umi` | 메인 파이프라인 전체 (데이터 수신, zarr 생성, 학습, 평가) | `install_umi_env.sh` |
| `droid` | DROID-SLAM 실행 전용 (파이프라인에서 자동 호출) | `install_droid_env.sh` |

`run_slam_pipeline_s21.py`는 `umi` 환경에서 실행되며, SLAM 단계에서 내부적으로 `conda run -n droid`를 통해 `droid` 환경을 자동 호출한다. **사용자가 직접 `droid` 환경을 활성화할 필요는 없다.**

> **왜 환경을 두 개로 나누는가**
> `lietorch`와 `droid_backends`는 CUDA 커널을 직접 컴파일하는 패키지로, 설치 과정이 복잡하고 빌드 환경에 민감하다. 학습 파이프라인 패키지들(`diffusers`, `zarr` 등)과 같은 환경에 혼합하면 의존성 충돌 위험이 있어 SLAM 전용 환경을 별도로 유지한다.

---

### 2-1. `umi` 환경 설치

```bash
cd ~/umi-galaxy-s21
bash install_umi_env.sh
```

스크립트가 수행하는 작업:

| 단계 | 내용 |
|------|------|
| conda 환경 생성 | `conda_environment.yaml` 기반으로 `umi` 환경 생성 |
| torch CUDA 확인 | yaml 설치 후 CUDA 사용 가능 여부 자동 확인 |
| torch 재설치 (필요 시) | CUDA 비활성화 상태면 torch 2.11.0+cu128 재설치 |
| 최종 확인 | torch, zarr, opencv, wandb, diffusers 임포트 확인 |

완료 시 출력:
```
[OK] torch: 2.11.0+cu128
[OK] zarr: 2.16.1
[OK] opencv: 4.7.0
[OK] wandb: ...
[OK] diffusers: ...
============================================================
 umi environment setup complete.
============================================================
```

> **주의**: `conda_environment.yaml`에서 채널 순서(`conda-forge`가 첫 번째)가 중요하다. 순서가 잘못되면 `av=10.0.0` 패키지 설치가 실패한다.

---

### 2-2. `droid` 환경 설치

```bash
cd ~/umi-galaxy-s21
bash install_droid_env.sh
```

스크립트가 수행하는 작업:

| 단계 | 내용 |
|------|------|
| CUDA 경로 자동 감지 | `CUDA_HOME` 환경변수 → PATH의 nvcc → `/usr/local/cuda*` 순서로 탐색 |
| conda 경로 자동 감지 | `conda info --base`로 자동 감지. 경로 하드코딩 없음 |
| conda 환경 생성 | `droid` (Python 3.10) |
| PyTorch 설치 | torch 2.11.0+cu128 |
| 의존성 설치 | matplotlib, scipy, open3d, opencv 등 |
| lietorch 빌드 | CUDA 커널 직접 컴파일 (`~/DROID-SLAM/thirdparty/lietorch`) |
| droid_backends 빌드 | DROID-SLAM CUDA 백엔드 컴파일 (`~/DROID-SLAM`) |
| torch-scatter 설치 | prebuilt wheel |
| 환경변수 영구 등록 | torch 라이브러리 경로(`LD_LIBRARY_PATH`), droid_slam 모듈 경로(`.pth`) |
| 최종 확인 | droid_backends, lietorch, Droid 클래스 임포트 확인 |

완료 시 출력:
```
[OK] droid_backends
[OK] lietorch
[OK] droid (Droid class importable)
============================================================
 DROID-SLAM environment setup complete.
============================================================
```

> CUDA 커널 컴파일(`lietorch`, `droid_backends`) 단계는 수 분 소요된다.

---

## 3. 실행 테스트

`umi` 환경에서 실행한다. `droid` 환경은 자동으로 호출된다.

```bash
conda activate umi
cd ~/umi-galaxy-s21

python run_slam_pipeline_s21.py process-droid \
  --calibration_dir example/calibration_s21 \
  --ref aruco \
  /path/to/session_dir
```

정상 실행 시 출력:
```
Processing session (DROID-SLAM): /path/to/session_dir
============================================================

--- 00 Process Videos ---
Detected app pre-organized structure.
  → gripper_calibration_Galaxy_S21_.../
  → demo_Galaxy_S21_.../
  ...

--- 01 DROID-SLAM ---
Calib: 782.40 783.44 972.69 541.65 ...
Found N demo video directories
Gripper mask: ON
  [demo_Galaxy_S21_...] Extracting frames (mask=on)...
  [demo_Galaxy_S21_...] 301 frames @ 30.0fps
  [demo_Galaxy_S21_...] Running DROID-SLAM...
  [demo_Galaxy_S21_...] Done -> camera_trajectory.csv
...
Done: N/N succeeded
```

세션 폴더 구조:
```
/path/to/session_dir/
├── gripper_calibration/        ← 별도 촬영 후 폴더명 변경 (1개만 필요)
│   ├── camera_ultrawide.mp4
│   ├── frame_timestamps.csv
│   ├── metadata.json
│   └── sensor_data.jsonl
├── session_20260610_160512/    ← 데모 에피소드 (촬영 시각이 폴더명)
│   └── ...
├── session_20260610_161926/
└── ...
```

---

## 트러블슈팅

| 오류 | 원인 | 해결 |
|------|------|------|
| `CUDA Toolkit not found` | nvcc를 어디서도 찾지 못함 | `find /usr/local -name nvcc`로 위치 확인 후 `export CUDA_HOME=...` 설정 |
| `CUDA not available` (torch) | GPU 드라이버 또는 CUDA 버전 불일치 | `nvidia-smi`와 torch CUDA 버전 확인 |
| `av=10.0.0 is not installable` | conda 채널 순서 문제 | `conda_environment.yaml`에서 `conda-forge`가 첫 번째 채널인지 확인 |
| `wandb` 임포트 오류 | protobuf 버전 충돌 | `pip install protobuf==4.25.8` |
| `ModuleNotFoundError: No module named 'torch'` (lietorch 빌드 시) | `pip install -e` 방식 문제 | 스크립트가 자동으로 `python setup.py develop` 사용 |
| `libc10.so: cannot open shared object file` | torch 라이브러리 경로 미등록 | `install_droid_env.sh` 재실행 |
| `ModuleNotFoundError: No module named 'droid'` | droid_slam 모듈 경로 미등록 | `install_droid_env.sh` 재실행 |
| `ModuleNotFoundError: No module named 'torch_scatter'` | torch-scatter 미설치 | `conda run -n droid pip install torch-scatter -f https://data.pyg.org/whl/torch-2.11.0+cu128.html` |
| `droid.pth not found` | 모델 가중치 미복사 | `~/DROID-SLAM/droid.pth` 위치 확인 |
