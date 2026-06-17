# DROID-SLAM 설치 가이드

완전 초기 상태(conda 환경 없음, DROID-SLAM 없음)에서 SLAM 실행까지의 전 과정을 기술한다.

---

## 목차

1. [사전 요구사항](#1-사전-요구사항)
2. [umi conda 환경 설치](#2-umi-conda-환경-설치)
3. [DROID-SLAM 설치](#3-droid-slam-설치)
4. [SLAM 실행 테스트](#4-slam-실행-테스트)

---

## 1. 사전 요구사항

### 검증된 환경

| 항목 | 버전 |
|------|------|
| OS | Ubuntu 22.04 |
| GPU | NVIDIA RTX 5060 Ti (16GB) |
| GPU 드라이버 | 595.71 |
| CUDA (드라이버) | 13.2 |
| CUDA Toolkit | 12.8 (`/usr/local/cuda-12.8`) |
| Miniforge | 최신 버전 |

### Miniforge 설치

Anaconda 대신 Miniforge를 권장한다. 패키지 해결 속도가 빠르고 conda-forge 채널을 기본으로 사용한다.

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
```

설치 후 터미널을 재시작하거나 `source ~/.bashrc`를 실행한다.

---

## 2. umi conda 환경 설치

메인 파이프라인 전체(SLAM, zarr 생성, 학습)에서 사용하는 환경이다.

```bash
conda env create -f conda_environment.yaml -n umi
conda activate umi
```

torch는 conda yaml의 pip 섹션에 `--extra-index-url`로 지정되어 있으나, 환경에 따라 자동 적용이 안 될 수 있다. 그 경우 아래를 추가 실행한다:

```bash
pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
```

> **주의**: 채널 순서(`conda-forge`가 첫 번째)가 중요하다. 순서가 잘못되면 `av=10.0.0` 패키지 설치가 실패한다.

### 설치 확인

```bash
python -c "import torch; print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
python -c "import zarr; print('zarr:', zarr.__version__)"
python -c "import cv2; print('opencv:', cv2.__version__)"
python -c "import wandb; print('wandb:', wandb.__version__)"
```

예상 출력:
```
torch: 2.11.0+cu128
CUDA: True
zarr: 2.16.1
opencv: 4.7.0
wandb: 0.25.1
```

---

## 3. DROID-SLAM 설치

DROID-SLAM은 별도의 `droid` conda 환경에서 실행된다. `run_slam_pipeline_s21.py`가 내부적으로 `conda run -n droid`를 통해 자동 호출하므로, 사용자가 직접 `droid` 환경을 활성화할 필요는 없다.

### 3-1. 저장소 클론

```bash
cd ~
git clone --recursive https://github.com/princeton-vl/DROID-SLAM.git
cd DROID-SLAM
```

> `--recursive` 플래그가 반드시 필요하다. `thirdparty/lietorch` 등 서브모듈이 함께 클론된다.
> 빠뜨린 경우: `git submodule update --init --recursive`

### 3-2. conda 환경 생성 및 torch 설치

```bash
conda create -n droid python=3.10 -y
conda activate droid
pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 torchaudio==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
```

> ROS2 관련 경고(`generate-parameter-library-py requires pyyaml` 등)가 뜰 수 있으나 무시해도 된다.

### 설치 확인 (다음 단계 전 반드시 확인)

```bash
python -c "import torch; print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

예상 출력:
```
torch: 2.11.0+cu128
CUDA available: True
```

### 3-3. 기타 의존성 설치

```bash
pip install matplotlib scipy tqdm scikit-learn open3d opencv-python pyquaternion numpy==1.26.4
```

> ROS2 관련 경고(`generate-parameter-library-py requires typeguard` 등)가 뜰 수 있으나 무시해도 된다.

### 설치 확인

```bash
python -c "import numpy; print('numpy:', numpy.__version__)"
python -c "import cv2; print('opencv:', cv2.__version__)"
python -c "import open3d; print('open3d:', open3d.__version__)"
```

예상 출력:
```
numpy: 1.26.4
opencv: 4.11.0
open3d: 0.19.0
```

### 3-4. lietorch 빌드

`lietorch`는 CUDA 커널을 직접 컴파일하는 패키지다.

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
cd ~/DROID-SLAM
python setup.py develop   # thirdparty/lietorch
```

> `pip install -e thirdparty/lietorch`는 `ModuleNotFoundError: No module named 'torch'` 오류가 발생할 수 있다. `python setup.py develop`을 사용한다.

### 3-5. droid_backends 빌드

DROID-SLAM의 CUDA 백엔드를 빌드한다.

```bash
cd ~/DROID-SLAM
python setup.py install
```

### 3-6. torch-scatter 설치

```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.11.0+cu128.html
```

### 3-7. 영구 환경변수 등록

`libc10.so` 등 torch 공유 라이브러리 경로와 DROID-SLAM Python 모듈 경로를 conda 환경에 영구 등록한다. 이 설정이 없으면 `droid_backends` 임포트 시 오류가 발생한다.

```bash
# torch 라이브러리 경로 등록 (libc10.so 등)
mkdir -p /home/$USER/miniforge3/envs/droid/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=/home/$USER/miniforge3/envs/droid/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH' \
  > /home/$USER/miniforge3/envs/droid/etc/conda/activate.d/torch_lib.sh

# DROID-SLAM Python 모듈 경로 등록
echo "/home/$USER/DROID-SLAM/droid_slam" \
  > /home/$USER/miniforge3/envs/droid/lib/python3.10/site-packages/droid_slam.pth
```

> `.pth` 파일은 Python이 직접 읽는 텍스트 파일이므로 `$USER`가 실제 경로로 치환되어야 한다. 작은따옴표(`'`) 대신 큰따옴표(`"`)를 사용한다.

### 3-8. 모델 가중치 복사

`droid.pth`를 DROID-SLAM 폴더에 복사한다.

```bash
cp /path/to/droid.pth ~/DROID-SLAM/droid.pth
```

> 네트워크 차단 환경의 경우 위와 같이 로컬에서 복사한다.
> 네트워크가 가능한 환경: https://drive.google.com/file/d/1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh

### 3-9. 설치 최종 확인

새 터미널을 열어서 확인한다.

```bash
conda activate droid
python -c "import droid_backends; print('droid_backends: OK')"
python -c "from droid import Droid; print('droid: OK')"
python -c "import lietorch; print('lietorch: OK')"
```

세 개 모두 OK가 출력되면 설치 완료다.

---

## 4. SLAM 실행 테스트

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
| `av=10.0.0 is not installable` | conda 채널 순서 문제 | `conda-forge`를 첫 번째 채널로 설정 (`conda-forge` 채널에만 Python 3.10용 빌드가 존재) |
| `wandb` 임포트 오류 | protobuf 버전 충돌 | `pip install protobuf==4.25.8` |
| `ModuleNotFoundError: No module named 'torch'` (lietorch 빌드 시) | `pip install -e` 방식 문제 | `python setup.py develop` 사용 |
| `libc10.so: cannot open shared object file` | torch 라이브러리 경로 미등록 | [3-7 영구 환경변수 등록](#3-7-영구-환경변수-등록) |
| `ModuleNotFoundError: No module named 'droid'` | DROID-SLAM 모듈 경로 미등록 | [3-7 영구 환경변수 등록](#3-7-영구-환경변수-등록) |
| `ModuleNotFoundError: No module named 'torch_scatter'` | torch-scatter 미설치 | [3-6 torch-scatter 설치](#3-6-torch-scatter-설치) |
| `droid.pth not found` | DROID-SLAM 경로 설정 오류 | `~/DROID-SLAM/droid.pth` 위치 확인 |
