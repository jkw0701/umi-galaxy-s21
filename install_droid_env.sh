#!/usr/bin/env bash
# DROID-SLAM conda 환경 설치 스크립트
#
# 사용법:
#   bash install_droid_env.sh
#
# 사전 요구사항:
#   - CUDA Toolkit 설치 완료 (nvcc 사용 가능)
#   - Miniforge(또는 Anaconda) 설치
#   - DROID-SLAM 클론 완료: ~/DROID-SLAM
#
# 검증 환경: Ubuntu 22.04, CUDA 12.8, RTX 5060 Ti
set -e

echo "============================================================"
echo " DROID-SLAM environment setup"
echo "============================================================"

# ── 1. CUDA 경로 자동 감지 ────────────────────────────────────────────────
# 우선순위:
#   1) 이미 CUDA_HOME 환경변수가 설정된 경우 그대로 사용
#   2) nvcc가 PATH에 있는 경우 그 위치에서 역으로 CUDA_HOME 추론
#   3) /usr/local/cuda* 중 nvcc가 존재하는 디렉토리 탐색
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
    echo "  CUDA Toolkit을 설치하거나 CUDA_HOME을 직접 지정하세요:"
    echo "  export CUDA_HOME=/path/to/cuda && bash install_droid_env.sh"
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
    echo "[ERROR] thirdparty/lietorch not found. --recursive 플래그 없이 클론했나요?"
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
assert torch.cuda.is_available(), 'CUDA not available — GPU 드라이버와 CUDA Toolkit 버전을 확인하세요'
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
echo ""

# ── 9. torch-scatter 설치 ─────────────────────────────────────────────────
echo "==> [6/6] Installing torch-scatter..."
conda run -n droid pip install torch-scatter \
    -f https://data.pyg.org/whl/torch-2.11.0+cu128.html
echo ""

# ── 10. 환경변수 영구 등록 ────────────────────────────────────────────────
echo "==> Registering permanent environment variables..."

DROID_ENV_DIR="$CONDA_BASE/envs/droid"

# site-packages 경로를 find로 직접 탐색
SITE_PACKAGES=$(find "$DROID_ENV_DIR/lib" -maxdepth 2 -type d -name "site-packages" | head -1)
if [ -z "$SITE_PACKAGES" ]; then
    echo "[ERROR] site-packages not found in $DROID_ENV_DIR/lib"
    exit 1
fi
echo "[INFO] site-packages: $SITE_PACKAGES"

TORCH_LIB="$SITE_PACKAGES/torch/lib"

# conda activate 시 자동으로 LD_LIBRARY_PATH에 torch 라이브러리 경로 추가
ACTIVATE_D="$DROID_ENV_DIR/etc/conda/activate.d"
mkdir -p "$ACTIVATE_D"
cat > "$ACTIVATE_D/droid_env.sh" << EOF
export LD_LIBRARY_PATH=$TORCH_LIB:\$LD_LIBRARY_PATH
EOF
echo "[OK] LD_LIBRARY_PATH registered: $ACTIVATE_D/droid_env.sh"

# Python이 droid_slam 모듈을 찾을 수 있도록 .pth 파일 등록
PTH_PATH="$SITE_PACKAGES/droid_slam.pth"
echo "$DROID_DIR/droid_slam" > "$PTH_PATH"
echo "[OK] droid_slam.pth registered: $PTH_PATH"
echo ""

# ── 11. 최종 확인 ─────────────────────────────────────────────────────────
# conda run은 activate.d 훅을 트리거하지 않으므로 LD_LIBRARY_PATH를 직접 주입
echo "==> Final verification..."
conda run -n droid \
    bash -c "export LD_LIBRARY_PATH=$TORCH_LIB:\$LD_LIBRARY_PATH && \
             export PYTHONPATH=$DROID_DIR/droid_slam:\$PYTHONPATH && \
             python -c \"
import droid_backends; print('[OK] droid_backends')
import lietorch;       print('[OK] lietorch')
from droid import Droid; print('[OK] droid (Droid class importable)')
\""

echo ""
echo "============================================================"
echo " DROID-SLAM environment setup complete."
echo ""
echo " To test manually:  conda activate droid"
echo " In the pipeline:   run_slam_pipeline_s21.py calls droid"
echo "                    automatically via 'conda run -n droid'"
echo "============================================================"
