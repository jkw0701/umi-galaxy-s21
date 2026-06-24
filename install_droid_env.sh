#!/usr/bin/env bash
# DROID-SLAM conda 환경 설치 스크립트
#
# 사용법:
#   bash install_droid_env.sh
#
# 사전 요구사항:
#   - CUDA Toolkit 설치 (/usr/local/cuda 심볼릭 링크 존재)
#   - Miniforge(또는 Anaconda) 설치
#   - DROID-SLAM 클론 완료: ~/DROID-SLAM
#
# 검증 환경: Ubuntu 22.04, CUDA 12.8, RTX 5060 Ti
set -e

echo "============================================================"
echo " DROID-SLAM environment setup"
echo "============================================================"

# ── 1. CUDA 경로 자동 감지 ────────────────────────────────────────────────
if [ -n "$CUDA_HOME" ] && [ -d "$CUDA_HOME" ]; then
    echo "[INFO] Using CUDA_HOME from environment: $CUDA_HOME"
elif [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME=/usr/local/cuda
    echo "[INFO] CUDA detected: $CUDA_HOME ($(readlink -f $CUDA_HOME))"
else
    echo "[ERROR] CUDA Toolkit not found."
    echo "  Install: sudo apt-get install cuda-toolkit-12-8"
    echo "  Or set:  export CUDA_HOME=/path/to/cuda"
    exit 1
fi
export PATH=$CUDA_HOME/bin:$PATH

if ! command -v nvcc &> /dev/null; then
    echo "[ERROR] nvcc not found at $CUDA_HOME/bin. Check CUDA Toolkit installation."
    exit 1
fi
echo "[INFO] nvcc: $(nvcc --version | grep release)"

# ── 2. conda 경로 자동 감지 ──────────────────────────────────────────────
CONDA_BASE=$(conda info --base 2>/dev/null)
if [ -z "$CONDA_BASE" ]; then
    echo "[ERROR] conda not found. Install Miniforge first."
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
    echo "[ERROR] thirdparty/lietorch not found. Missing --recursive flag during clone?"
    echo "  Run: cd ~/DROID-SLAM && git submodule update --init --recursive"
    exit 1
fi
echo "[INFO] DROID-SLAM: $DROID_DIR"
echo ""

# ── 4. conda 환경 생성 ────────────────────────────────────────────────────
echo "==> [1/6] Creating conda environment 'droid' (python 3.10)..."
conda create -n droid python=3.10 -y
echo ""

# ── 5. PyTorch 설치 ───────────────────────────────────────────────────────
echo "==> [2/6] Installing PyTorch 2.11.0+cu128..."
conda run -n droid pip install \
    torch==2.11.0+cu128 torchvision==0.26.0+cu128 torchaudio==2.11.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

conda run -n droid python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available — check GPU driver and CUDA Toolkit version'
print('[OK] torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())
"
echo ""

# ── 6. 기타 의존성 설치 ───────────────────────────────────────────────────
echo "==> [3/6] Installing dependencies..."
conda run -n droid pip install \
    matplotlib scipy tqdm scikit-learn open3d opencv-python pyquaternion numpy==1.26.4
echo ""

# ── 7. lietorch 빌드 ──────────────────────────────────────────────────────
echo "==> [4/6] Building lietorch (CUDA kernel compile — a few minutes)..."
conda run -n droid --no-capture-output \
    bash -c "export CUDA_HOME=$CUDA_HOME && export PATH=$CUDA_HOME/bin:\$PATH && \
             cd $DROID_DIR/thirdparty/lietorch && python setup.py develop"

conda run -n droid python -c "import lietorch; print('[OK] lietorch')"
echo ""

# ── 8. droid_backends 빌드 ────────────────────────────────────────────────
echo "==> [5/6] Building droid_backends..."
conda run -n droid --no-capture-output \
    bash -c "export CUDA_HOME=$CUDA_HOME && export PATH=$CUDA_HOME/bin:\$PATH && \
             cd $DROID_DIR && python setup.py install"

conda run -n droid python -c "import droid_backends; print('[OK] droid_backends')"
echo ""

# ── 9. torch-scatter 설치 ─────────────────────────────────────────────────
echo "==> [6/6] Installing torch-scatter..."
conda run -n droid pip install torch-scatter \
    -f https://data.pyg.org/whl/torch-2.11.0+cu128.html
echo ""

# ── 10. 환경변수 영구 등록 ────────────────────────────────────────────────
echo "==> Registering permanent environment variables..."

DROID_ENV_DIR="$CONDA_BASE/envs/droid"

# conda activate 시 자동으로 LD_LIBRARY_PATH에 torch 라이브러리 경로 추가
ACTIVATE_D="$DROID_ENV_DIR/etc/conda/activate.d"
mkdir -p "$ACTIVATE_D"
cat > "$ACTIVATE_D/droid_env.sh" << EOF
export LD_LIBRARY_PATH=$DROID_ENV_DIR/lib/python3.10/site-packages/torch/lib:\$LD_LIBRARY_PATH
EOF
echo "[OK] LD_LIBRARY_PATH registered: $ACTIVATE_D/droid_env.sh"

# Python이 droid_slam 모듈을 찾을 수 있도록 .pth 파일 등록
PTH_PATH="$DROID_ENV_DIR/lib/python3.10/site-packages/droid_slam.pth"
echo "$DROID_DIR/droid_slam" > "$PTH_PATH"
echo "[OK] droid_slam.pth registered: $PTH_PATH"
echo ""

# ── 11. 최종 확인 ─────────────────────────────────────────────────────────
echo "==> Final verification..."
conda run -n droid python -c "
import droid_backends; print('[OK] droid_backends')
import lietorch;       print('[OK] lietorch')
from droid import Droid; print('[OK] droid (Droid class importable)')
"

echo ""
echo "============================================================"
echo " DROID-SLAM environment setup complete."
echo ""
echo " To test manually:  conda activate droid"
echo " In the pipeline:   run_slam_pipeline_s21.py calls droid"
echo "                    automatically via 'conda run -n droid'"
echo "============================================================"
