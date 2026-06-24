#!/usr/bin/env bash
# umi conda 환경 설치 스크립트
#
# 사용법:
#   bash install_umi_env.sh
#
# 사전 요구사항:
#   - Miniforge(또는 Anaconda) 설치
#   - NVIDIA GPU 드라이버 설치
#   - 이 스크립트는 리포지토리 루트에서 실행한다
set -e

echo "============================================================"
echo " umi environment setup"
echo "============================================================"

# ── 1. conda 경로 자동 감지 ──────────────────────────────────────────────
CONDA_BASE=$(conda info --base 2>/dev/null)
if [ -z "$CONDA_BASE" ]; then
    echo "[ERROR] conda not found. Install Miniforge first."
    exit 1
fi
echo "[INFO] conda base: $CONDA_BASE"

# ── 2. conda_environment.yaml 존재 확인 ──────────────────────────────────
YAML_PATH="$(cd "$(dirname "$0")" && pwd)/conda_environment.yaml"
if [ ! -f "$YAML_PATH" ]; then
    echo "[ERROR] conda_environment.yaml not found at $YAML_PATH"
    echo "  Run this script from the repository root."
    exit 1
fi

# ── 3. umi 환경 생성 ──────────────────────────────────────────────────────
echo ""
echo "==> [1/3] Creating conda environment 'umi' from conda_environment.yaml..."
conda env create -f "$YAML_PATH" -n umi
echo ""

# ── 4. torch CUDA 설치 확인 및 보완 ──────────────────────────────────────
echo "==> [2/3] Verifying torch CUDA installation..."
TORCH_OK=$(conda run -n umi python -c "
import torch
ok = torch.cuda.is_available()
print('ok' if ok else 'fail')
print(torch.__version__)
" 2>/dev/null || echo "fail")

if echo "$TORCH_OK" | grep -q "^ok"; then
    TORCH_VER=$(echo "$TORCH_OK" | tail -1)
    echo "[OK] torch $TORCH_VER with CUDA"
else
    echo "[WARN] torch CUDA not available. Reinstalling torch with CUDA 12.8..."
    conda run -n umi pip install \
        torch==2.11.0+cu128 torchvision==0.26.0+cu128 \
        --index-url https://download.pytorch.org/whl/cu128

    conda run -n umi python -c "
import torch
assert torch.cuda.is_available(), 'CUDA still not available after reinstall'
print('[OK] torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())
"
fi
echo ""

# ── 5. 최종 확인 ──────────────────────────────────────────────────────────
echo "==> [3/3] Final verification..."
conda run -n umi python -c "
import torch;    print('[OK] torch:', torch.__version__)
import zarr;     print('[OK] zarr:', zarr.__version__)
import cv2;      print('[OK] opencv:', cv2.__version__)
import wandb;    print('[OK] wandb:', wandb.__version__)
import diffusers;print('[OK] diffusers:', diffusers.__version__)
"

echo ""
echo "============================================================"
echo " umi environment setup complete."
echo " Run 'conda activate umi' to use."
echo "============================================================"
