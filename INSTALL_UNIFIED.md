# 통합 환경 설치 가이드 (umi_full)

`umi` + `droid` 환경을 하나로 통합한 `umi_full` 환경 기반 설치 가이드.  
SLAM, 학습, 평가 모두 `umi_full` 환경 하나에서 실행한다.

분리 환경 가이드는 [INSTALL_DROID_SLAM.md](INSTALL_DROID_SLAM.md)를 참고한다.

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

---

### 1-1. GPU 드라이버 및 CUDA Toolkit 확인

```bash
nvidia-smi
find /usr/local /opt -name nvcc 2>/dev/null
```

- **경우 A** — `nvidia-smi`가 실행되지 않는 경우: GPU 드라이버 없음. 별도 설치 필요.
- **경우 B** — `nvidia-smi`는 되지만 `find` 결과가 비어있는 경우: CUDA Toolkit 없음. → **1-2로 이동**
- **경우 C** — `find` 결과에 경로가 출력되는 경우 (예: `/usr/local/cuda-12.8/bin/nvcc`): → **1-2 건너뛰고 1-3으로 이동**

---

### 1-2. CUDA Toolkit 설치 (미설치 시만)

Ubuntu 22.04 + x86_64 기준. 다른 환경은 [CUDA Toolkit 공식 페이지](https://developer.nvidia.com/cuda-downloads)를 참고한다.

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-8
```

설치 후 확인:
```bash
find /usr/local /opt -name nvcc 2>/dev/null
```

---

### 1-3. Miniforge 설치 (미설치 시만)

`conda --version`이 작동하면 이 단계를 건너뛴다.

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
```

설치 중 마지막 질문에 반드시 `yes` 입력:
```
Do you wish to update your shell profile to automatically initialize conda?
[yes|no]
>>> yes
```

설치 완료 후 터미널을 새로 열거나:
```bash
source ~/.bashrc
conda --version
```

---

### 1-4. 이 리포지토리 클론

```bash
git clone https://github.com/jkw0701/umi-galaxy-s21.git ~/umi-galaxy-s21
cd ~/umi-galaxy-s21
```

> **주의**: `install_umi_full_env.sh`가 `~/umi-galaxy-s21` 경로를 참조한다. 다른 경로에 클론하면 스크립트가 실패한다.

---

### 1-5. DROID-SLAM 클론

```bash
git clone --recursive https://github.com/princeton-vl/DROID-SLAM.git ~/DROID-SLAM
```

> **주의**: `install_umi_full_env.sh`가 `~/DROID-SLAM` 경로를 참조한다. 다른 경로에 클론하면 스크립트가 실패한다.

`--recursive` 플래그가 반드시 필요하다. 빠뜨린 경우:
```bash
cd ~/DROID-SLAM && git submodule update --init --recursive
```

---

### 1-6. 모델 가중치 복사

`droid.pth`는 이 리포지토리에 포함되어 있다 (Git LFS로 관리). `git clone` 시 자동으로 다운로드된다.

```bash
cp ~/umi-galaxy-s21/droid.pth ~/DROID-SLAM/droid.pth
```

---

## 2. 환경 구성 (스크립트)

`umi_full` 단일 환경에 umi 파이프라인과 DROID-SLAM을 모두 설치한다.

```bash
cd ~/umi-galaxy-s21
bash install_umi_full_env.sh
```

스크립트가 수행하는 작업:

| 단계 | 내용 |
|------|------|
| 1/8 | `conda_environment.yaml` 기반으로 `umi_full` 환경 생성 |
| 2/8 | PyTorch CUDA 검증 |
| 3/8 | droid 전용 추가 패키지 설치 (open3d, pyquaternion) |
| 4/8 | lietorch CUDA 커널 빌드 (`~/DROID-SLAM/thirdparty/lietorch`) |
| 5/8 | droid_backends CUDA 커널 빌드 (`~/DROID-SLAM`) |
| 6/8 | torch-scatter 설치 |
| 7/8 | LD_LIBRARY_PATH, droid_slam 모듈 경로 영구 등록 |
| 8/8 | 전체 패키지 임포트 최종 확인 |

완료 시 출력:
```
[OK] torch: 2.11.0+cu128
[OK] zarr
[OK] cv2
[OK] wandb
[OK] diffusers
[OK] droid_backends
[OK] lietorch
[OK] droid (Droid class importable)
============================================================
 umi_full environment setup complete.
============================================================
```

> CUDA 커널 컴파일(`lietorch`, `droid_backends`) 단계는 수 분 소요된다.

---

## 3. 실행 테스트

`umi_full` 환경 하나에서 모두 실행한다. SLAM도 별도 환경 전환 없이 동작한다.

```bash
conda activate umi_full
cd ~/umi-galaxy-s21

python run_slam_pipeline_s21.py process-droid \
  --calibration_dir example/calibration_s21 \
  --ref aruco \
  /path/to/session_dir
```

---

## 트러블슈팅

| 오류 | 원인 | 해결 |
|------|------|------|
| `CUDA Toolkit not found` | nvcc를 어디서도 찾지 못함 | `find /usr/local -name nvcc`로 위치 확인 후 `export CUDA_HOME=...` 설정 |
| `CUDA not available` (torch) | GPU 드라이버 또는 CUDA 버전 불일치 | `nvidia-smi`와 torch CUDA 버전 확인 |
| `av=10.0.0 is not installable` | conda 채널 순서 문제 | `conda_environment.yaml`에서 `conda-forge`가 첫 번째 채널인지 확인 |
| `libc10.so: cannot open shared object file` | torch 라이브러리 경로 미등록 | `install_umi_full_env.sh` 재실행 |
| `ModuleNotFoundError: No module named 'droid'` | droid_slam 모듈 경로 미등록 | `install_umi_full_env.sh` 재실행 |
| `ModuleNotFoundError: No module named 'torch_scatter'` | torch-scatter 미설치 | `conda run -n umi_full pip install torch-scatter -f https://data.pyg.org/whl/torch-2.11.0+cu128.html` |
| `droid.pth not found` | 모델 가중치 미복사 | `cp ~/umi-galaxy-s21/droid.pth ~/DROID-SLAM/droid.pth` 실행 |
