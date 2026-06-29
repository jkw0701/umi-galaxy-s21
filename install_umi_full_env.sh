#!/usr/bin/env bash
# umi_full conda 환경 설치 스크립트
# umi + droid slam 통합 환경 (학습, 평가, SLAM 모두 포함)
#
# 사용법:
#   bash install_umi_full_env.sh
#
# 사전 요구사항:
#   - CUDA Toolkit 설치 완료 (nvcc 사용 가능)
#   - Miniforge(또는 Anaconda) 설치
#   - DROID-SLAM 클론 완료: ~/DROID-SLAM
#   - umi-galaxy-s21 클론 완료: ~/umi-galaxy-s21
#
# 검증 환경: Ubuntu 22.04, CUDA 12.8, RTX 5060 Ti
set -e

echo "============================================================"
echo " umi_full environment setup (umi + droid slam unified)"
echo "============================================================"

# ── 1. CUDA 경로 자동 감지 ────────────────────────────────────────────────
find_cuda_home() {
    if [ -n "$CUDA_HOME" ] && [ -x "$CUDA_HOME/bin/nvcc" ]; then
        echo "$CUDA_HOME"; return
    fi
    local nvcc_path
    nvcc_path=$(command -v nvcc 2>/dev/null)
    if [ -n "$nvcc_path" ]; then
        echo "$(dirname "$(dirname "$nvcc_path")")"; return
    fi
    for d in $(ls -d /usr/local/cuda* 2>/dev/null | sort -rV); do
        if [ -x "$d/bin/nvcc" ]; then
            echo "$d"; return
        fi
    done
    echo ""
}

CUDA_HOME=$(find_cuda_home)
if [ -z "$CUDA_HOME" ]; then
    echo "[ERROR] CUDA Toolkit not found. nvcc를 찾을 수 없습니다."
    echo "  export CUDA_HOME=/path/to/cuda && bash install_umi_full_env.sh"
    exit 1
fi
export CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
echo "[INFO] CUDA_HOME: $CUDA_HOME"
echo "[INFO] nvcc: $(nvcc --version | grep release)"

# ── 2. conda 경로 자동 감지 ──────────────────────────────────────────────
CONDA_BASE=$(conda info --base 2>/dev/null)
if [ -z "$CONDA_BASE" ]; then
    echo "[ERROR] conda not found. Miniforge를 먼저 설치하세요."
    exit 1
fi
echo "[INFO] conda base: $CONDA_BASE"

# ── 3. DROID-SLAM 경로 확인 ──────────────────────────────────────────────
DROID_DIR="$HOME/DROID-SLAM"
if [ ! -d "$DROID_DIR" ]; then
    echo "[ERROR] DROID-SLAM not found at $DROID_DIR"
    echo "  Run: git clone --recursive https://github.com/princeton-vl/DROID-SLAM.git ~/DROID-SLAM"
    exit 1
fi
if [ ! -d "$DROID_DIR/thirdparty/lietorch" ]; then
    echo "[ERROR] thirdparty/lietorch not found."
    echo "  Run: cd ~/DROID-SLAM && git submodule update --init --recursive"
    exit 1
fi
echo "[INFO] DROID-SLAM: $DROID_DIR"

# ── 4. umi-galaxy-s21 경로 확인 ──────────────────────────────────────────
UMI_DIR="$HOME/umi-galaxy-s21"
if [ ! -d "$UMI_DIR" ]; then
    echo "[ERROR] umi-galaxy-s21 not found at $UMI_DIR"
    exit 1
fi
echo "[INFO] umi-galaxy-s21: $UMI_DIR"
echo ""

# ── 5. conda 환경 생성 (umi conda_environment.yaml 기반) ─────────────────
echo "==> [1/8] Creating conda environment 'umi_full' from conda_environment.yaml..."
conda env create -n umi_full -f "$UMI_DIR/conda_environment.yaml"
echo ""

# ── 6. PyTorch CUDA 검증 ─────────────────────────────────────────────────
echo "==> [2/8] Verifying PyTorch CUDA..."
conda run -n umi_full python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available'
print('[OK] torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())
"
echo ""

# ── 7. droid 전용 의존성 설치 ────────────────────────────────────────────
echo "==> [3/8] Installing droid-specific dependencies..."
conda run -n umi_full pip install \
    open3d pyquaternion
echo ""

# ── 8. lietorch 빌드 ──────────────────────────────────────────────────────
echo "==> [4/8] Building lietorch (CUDA kernel compile — a few minutes)..."
conda run -n umi_full --no-capture-output \
    bash -c "export CUDA_HOME=$CUDA_HOME && export PATH=$CUDA_HOME/bin:\$PATH && \
             cd $DROID_DIR/thirdparty/lietorch && python setup.py develop"

conda run -n umi_full python -c "import lietorch; print('[OK] lietorch')"
echo ""

# ── 9. droid_backends 빌드 ────────────────────────────────────────────────
echo "==> [5/8] Building droid_backends..."
conda run -n umi_full --no-capture-output \
    bash -c "export CUDA_HOME=$CUDA_HOME && export PATH=$CUDA_HOME/bin:\$PATH && \
             cd $DROID_DIR && python setup.py install"
echo ""

# ── 10. torch-scatter 설치 ────────────────────────────────────────────────
echo "==> [6/8] Installing torch-scatter..."
conda run -n umi_full pip install torch-scatter \
    -f https://data.pyg.org/whl/torch-2.11.0+cu128.html
echo ""

# ── 11. 환경변수 영구 등록 ────────────────────────────────────────────────
echo "==> [7/8] Registering permanent environment variables..."

ENV_DIR="$CONDA_BASE/envs/umi_full"
SITE_PACKAGES=$(find "$ENV_DIR/lib" -maxdepth 2 -type d -name "site-packages" | head -1)
if [ -z "$SITE_PACKAGES" ]; then
    echo "[ERROR] site-packages not found in $ENV_DIR/lib"
    exit 1
fi
echo "[INFO] site-packages: $SITE_PACKAGES"

TORCH_LIB="$SITE_PACKAGES/torch/lib"

ACTIVATE_D="$ENV_DIR/etc/conda/activate.d"
mkdir -p "$ACTIVATE_D"
cat > "$ACTIVATE_D/umi_full_env.sh" << EOF
export LD_LIBRARY_PATH=$TORCH_LIB:\$LD_LIBRARY_PATH
EOF
echo "[OK] LD_LIBRARY_PATH registered: $ACTIVATE_D/umi_full_env.sh"

PTH_PATH="$SITE_PACKAGES/droid_slam.pth"
echo "$DROID_DIR/droid_slam" > "$PTH_PATH"
echo "[OK] droid_slam.pth registered: $PTH_PATH"
echo ""

# ── 12. 최종 확인 ─────────────────────────────────────────────────────────
echo "==> [8/8] Final verification..."
conda run -n umi_full \
    bash -c "export LD_LIBRARY_PATH=$TORCH_LIB:\$LD_LIBRARY_PATH && \
             export PYTHONPATH=$DROID_DIR/droid_slam:\$PYTHONPATH && \
             python -c \"
import torch;          print('[OK] torch:', torch.__version__)
import zarr;           print('[OK] zarr')
import cv2;            print('[OK] cv2')
import wandb;          print('[OK] wandb')
import diffusers;      print('[OK] diffusers')
import droid_backends; print('[OK] droid_backends')
import lietorch;       print('[OK] lietorch')
from droid import Droid; print('[OK] droid (Droid class importable)')
\""

echo ""
echo "============================================================"
echo " umi_full environment setup complete."
echo ""
echo " To activate:  conda activate umi_full"
echo " SLAM, 학습, 평가 모두 이 환경에서 실행 가능합니다."
echo "============================================================"
