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
#   - DROID-SLAM 클론 완료: umi-galaxy-s21과 같은 부모 디렉토리 아래에 DROID-SLAM 폴더로 클론
#     예) ~/foo/umi-galaxy-s21  →  ~/foo/DROID-SLAM
#
# 검증 환경: Ubuntu 22.04, CUDA 12.8, RTX 5060 Ti
set -e

echo "============================================================"
echo " umi_full environment setup (umi + droid slam unified)"
echo "============================================================"

# ── 1. CUDA 경로 자동 감지 ────────────────────────────────────────────────
find_cuda_home() {
    # 1) 명시적으로 CUDA_HOME이 설정된 경우 우선 사용 (심링크도 실제 경로로 resolve)
    if [ -n "$CUDA_HOME" ] && [ -x "$CUDA_HOME/bin/nvcc" ]; then
        readlink -f "$CUDA_HOME" 2>/dev/null || echo "$CUDA_HOME"; return
    fi
    # 2) /usr/local/cuda-X.Y 형태의 실제 디렉토리만 탐색 (심링크 제외), 가장 높은 버전 우선
    #    심링크(cuda, cuda-12 등)는 건너뜀 → 버전 비교가 명확한 실제 경로만 사용
    for d in $(ls -d /usr/local/cuda-[0-9]* 2>/dev/null | sort -rV); do
        if [ -x "$d/bin/nvcc" ]; then
            echo "$d"; return
        fi
    done
    # 3) PATH의 nvcc (단, /usr/bin은 시스템 패키지이므로 제외)
    local nvcc_path
    nvcc_path=$(command -v nvcc 2>/dev/null)
    if [ -n "$nvcc_path" ] && [[ "$nvcc_path" != /usr/bin/* ]]; then
        readlink -f "$(dirname "$(dirname "$nvcc_path")")" 2>/dev/null || \
            echo "$(dirname "$(dirname "$nvcc_path")")"; return
    fi
    # 4) 마지막 수단: find로 전체 탐색
    local found
    found=$(find /usr/local /opt -name nvcc -not -path "/usr/bin/*" 2>/dev/null | head -1)
    if [ -n "$found" ]; then
        echo "$(dirname "$(dirname "$found")")"; return
    fi
    echo ""
}

# PATH에서 /usr/bin을 제거하여 시스템 nvcc(11.5 등)가 우선되는 것을 방지
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '^/usr/bin$' | tr '\n' ':' | sed 's/:$//')

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

# ── 3. umi-galaxy-s21 경로 확인 (스크립트 위치 기준 자동 감지) ──────────
UMI_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ ! -f "$UMI_DIR/conda_environment.yaml" ]; then
    echo "[ERROR] conda_environment.yaml not found at $UMI_DIR"
    echo "  이 스크립트는 umi-galaxy-s21 리포지토리 루트에서 실행해야 합니다."
    exit 1
fi
echo "[INFO] umi-galaxy-s21: $UMI_DIR"

# ── 4. DROID-SLAM 경로 확인 (umi-galaxy-s21과 같은 부모 디렉토리) ────────
PARENT_DIR="$(dirname "$UMI_DIR")"
DROID_DIR="$PARENT_DIR/DROID-SLAM"
if [ ! -d "$DROID_DIR" ]; then
    echo "[ERROR] DROID-SLAM not found at $DROID_DIR"
    echo "  umi-galaxy-s21과 같은 디렉토리에 DROID-SLAM을 클론하세요:"
    echo "  git clone --recursive https://github.com/princeton-vl/DROID-SLAM.git $DROID_DIR"
    exit 1
fi
if [ ! -d "$DROID_DIR/thirdparty/lietorch" ]; then
    echo "[ERROR] thirdparty/lietorch not found."
    echo "  Run: cd $DROID_DIR && git submodule update --init --recursive"
    exit 1
fi
echo "[INFO] DROID-SLAM: $DROID_DIR"
echo ""

# ── 5. conda 환경 생성 (umi conda_environment.yaml 기반) ─────────────────
echo "==> [1/9] Creating conda environment 'umi_full' from conda_environment.yaml..."
conda env create -n umi_full -f "$UMI_DIR/conda_environment.yaml"
echo ""

# ── 6. PyTorch CUDA 검증 ─────────────────────────────────────────────────
echo "==> [2/9] Verifying PyTorch CUDA..."
conda run -n umi_full python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available'
print('[OK] torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())
"
echo ""

# ── 7. droid 전용 의존성 설치 ────────────────────────────────────────────
echo "==> [3/9] Installing droid-specific dependencies..."
conda run -n umi_full pip install \
    open3d pyquaternion
echo ""

# ── 8. lietorch setup.py 아키텍처 패치 ───────────────────────────────────
# 원본 setup.py는 sm_60~sm_75까지만 포함 → RTX 30/40/50 시리즈 누락
# sm_60/61/70 제거 후 sm_86(RTX30), sm_89(RTX40), compute_89 PTX(RTX50 JIT) 추가
echo "==> [4/9] Patching lietorch setup.py for modern GPU architectures..."
SETUP_PY="$DROID_DIR/thirdparty/lietorch/setup.py"
python3 -c "
import re

setup_path = '$SETUP_PY'
with open(setup_path) as f:
    content = f.read()

# 제거: sm_60, sm_61, sm_70 (Pascal/Volta, 2016~2017년)
for old in [
    \"                    '-gencode=arch=compute_60,code=sm_60', \n\",
    \"                    '-gencode=arch=compute_61,code=sm_61', \n\",
    \"                    '-gencode=arch=compute_70,code=sm_70', \n\",
]:
    content = content.replace(old, '')

# 추가: sm_86(RTX30), sm_89(RTX40), compute_89 PTX(RTX50 JIT fallback)
addition = (
    \"                    '-gencode=arch=compute_80,code=sm_80',\n\"
    \"                    '-gencode=arch=compute_86,code=sm_86',\n\"
    \"                    '-gencode=arch=compute_89,code=sm_89',\n\"
    \"                    '-gencode=arch=compute_89,code=compute_89',\n\"
)
anchor = \"                    '-gencode=arch=compute_75,code=compute_75',\n\"
if 'compute_89' not in content:
    content = content.replace(anchor, anchor + addition)

with open(setup_path, 'w') as f:
    f.write(content)

arches = re.findall(r'code=(sm_\d+|compute_\d+)', content)
print('[OK] lietorch setup.py patched')
print('[INFO] target architectures:', ', '.join(dict.fromkeys(arches)))
"
echo ""

# ── 9. lietorch 빌드 ──────────────────────────────────────────────────────
echo "==> [5/8] Building lietorch (CUDA kernel compile — a few minutes)..."
conda run -n umi_full --no-capture-output \
    bash -c "export CUDA_HOME=$CUDA_HOME && export PATH=$CUDA_HOME/bin:\$PATH && \
             cd $DROID_DIR/thirdparty/lietorch && \
             python setup.py build_ext --inplace && \
             pip install -e . --no-build-isolation"

conda run -n umi_full python -c "import lietorch; print('[OK] lietorch')"
echo ""

# ── 10. droid_backends 빌드 ───────────────────────────────────────────────
echo "==> [6/8] Building droid_backends..."
conda run -n umi_full --no-capture-output \
    bash -c "export CUDA_HOME=$CUDA_HOME && export PATH=$CUDA_HOME/bin:\$PATH && \
             cd $DROID_DIR && python setup.py install --no-build-isolation 2>/dev/null || \
             pip install . --no-build-isolation"
echo ""

# ── 11. torch-scatter 설치 ────────────────────────────────────────────────
echo "==> [7/8] Installing torch-scatter..."
conda run -n umi_full pip install torch-scatter \
    -f https://data.pyg.org/whl/torch-2.11.0+cu128.html
echo ""

# ── 12. 환경변수 영구 등록 ────────────────────────────────────────────────
echo "==> [8/8] Registering permanent environment variables..."

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
echo "==> [9/9] Final verification..."
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
