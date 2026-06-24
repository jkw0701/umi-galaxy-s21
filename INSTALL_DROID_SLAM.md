# DROID-SLAM 설치 가이드

완전 초기 상태(conda 환경 없음, DROID-SLAM 없음)에서 SLAM 실행까지의 전 과정을 기술한다.

---

## 목차

1. [사전 요구사항](#1-사전-요구사항)
2. [DROID-SLAM 설치](#2-droid-slam-설치)
3. [SLAM 실행 테스트](#3-slam-실행-테스트)

---

## 1. 사전 요구사항

### 검증된 환경

| 항목 | 버전 |
|------|------|
| OS | Ubuntu 22.04 |
| GPU | NVIDIA RTX 5060 Ti (16GB) |
| GPU 드라이버 | 595.71 |
| CUDA (드라이버) | 13.2 |
| CUDA Toolkit | 12.8 |
| Miniforge | 최신 버전 |

### CUDA Toolkit 설치

GPU 드라이버가 설치된 상태에서 CUDA Toolkit을 별도로 설치해야 한다. `nvidia-smi`에 표시되는 CUDA 버전은 드라이버 레벨이며, 컴파일에 필요한 `nvcc`는 Toolkit 설치 후에만 사용 가능하다.

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-8
```

설치 확인:

```bash
/usr/local/cuda/bin/nvcc --version
```

> 설치 후 `/usr/local/cuda`는 `/usr/local/cuda-12.8`을 가리키는 심볼릭 링크로 자동 생성된다. 설치 스크립트는 이 링크를 통해 CUDA 경로를 자동 감지하므로 버전을 직접 지정할 필요가 없다.

### Miniforge 설치

Anaconda 대신 Miniforge를 권장한다. 패키지 해결 속도가 빠르고 conda-forge 채널을 기본으로 사용한다.

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
```

설치 후 터미널을 재시작하거나 `source ~/.bashrc`를 실행한다.

---

## 2. DROID-SLAM 설치

DROID-SLAM은 별도의 `droid` conda 환경에서 실행된다. `run_slam_pipeline_s21.py`가 내부적으로 `conda run -n droid`를 통해 자동 호출하므로, 사용자가 직접 `droid` 환경을 활성화할 필요는 없다.

### 2-1. 저장소 클론

```bash
git clone --recursive https://github.com/princeton-vl/DROID-SLAM.git ~/DROID-SLAM
```

> `--recursive` 플래그가 반드시 필요하다. `thirdparty/lietorch` 등 서브모듈이 함께 클론된다.
> 빠뜨린 경우: `cd ~/DROID-SLAM && git submodule update --init --recursive`

### 2-2. 모델 가중치 복사

`droid.pth`를 DROID-SLAM 폴더에 복사한다.

```bash
cp /path/to/droid.pth ~/DROID-SLAM/droid.pth
```

> 네트워크가 가능한 환경: https://drive.google.com/file/d/1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh

### 2-3. 설치 스크립트 실행

이 리포 루트에서 아래 명령어 하나로 `droid` conda 환경 구성 전체를 자동으로 완료한다.

```bash
bash install_droid_env.sh
```

스크립트가 수행하는 작업:

| 단계 | 내용 |
|------|------|
| CUDA 경로 감지 | `/usr/local/cuda` 심볼릭 링크를 통해 자동 감지. `CUDA_HOME` 환경변수가 이미 설정된 경우 그대로 사용 |
| conda 경로 감지 | `conda info --base`로 자동 감지. 경로 하드코딩 없음 |
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

## 3. SLAM 실행 테스트

`umi` 환경에서 실행한다. `droid` 환경은 자동으로 호출된다.

```bash
conda activate umi
cd ~/umi-galaxy-s21   # 이 리포 루트

python run_slam_pipeline_s21.py process-droid \
  --calibration_dir example/calibration_s21 \
  --ref aruco \
  /path/to/session_dir
```

### 정상 실행 시 출력 예시

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

### 세션 폴더 구조

실행 전 세션 폴더는 아래 구조여야 한다.

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
| `CUDA not found` | CUDA Toolkit 미설치 또는 `/usr/local/cuda` 심볼릭 링크 없음 | CUDA Toolkit 설치 확인, `ls /usr/local/cuda` |
| `nvcc not found` | PATH에 nvcc 없음 | `export PATH=/usr/local/cuda/bin:$PATH` 후 재시도 |
| `CUDA not available` (torch) | GPU 드라이버 또는 CUDA 버전 불일치 | `nvidia-smi`와 torch CUDA 버전 확인 |
| `ModuleNotFoundError: No module named 'torch'` (lietorch 빌드 시) | `pip install -e` 방식 문제 | 스크립트가 자동으로 `python setup.py develop` 사용 |
| `libc10.so: cannot open shared object file` | torch 라이브러리 경로 미등록 | 스크립트 재실행 (환경변수 등록 단계 포함) |
| `ModuleNotFoundError: No module named 'droid'` | droid_slam 모듈 경로 미등록 | 스크립트 재실행 (`.pth` 파일 등록 단계 포함) |
| `ModuleNotFoundError: No module named 'torch_scatter'` | torch-scatter 미설치 | `conda run -n droid pip install torch-scatter -f https://data.pyg.org/whl/torch-2.11.0+cu128.html` |
| `droid.pth not found` | 모델 가중치 미복사 | `~/DROID-SLAM/droid.pth` 위치 확인 |
