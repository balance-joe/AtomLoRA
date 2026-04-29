#!/bin/bash
# AtomLoRA 安装脚本
# 用法: bash install.sh [cpu|cu118|cu121|cu124]
#   cpu    - 仅 CPU（默认）
#   cu118  - CUDA 11.8
#   cu121  - CUDA 12.1
#   cu124  - CUDA 12.4

set -e

MODE=${1:-cpu}

echo "=== AtomLoRA 安装 (模式: $MODE) ==="

# 安装 PyTorch
case "$MODE" in
    cpu)
        echo "[1/3] 安装 PyTorch (CPU)..."
        pip install torch --index-url https://download.pytorch.org/whl/cpu
        ;;
    cu118)
        echo "[1/3] 安装 PyTorch (CUDA 11.8)..."
        pip install torch --index-url https://download.pytorch.org/whl/cu118
        ;;
    cu121)
        echo "[1/3] 安装 PyTorch (CUDA 12.1)..."
        pip install torch --index-url https://download.pytorch.org/whl/cu121
        ;;
    cu124)
        echo "[1/3] 安装 PyTorch (CUDA 12.4)..."
        pip install torch --index-url https://download.pytorch.org/whl/cu124
        ;;
    *)
        echo "未知模式: $MODE"
        echo "用法: bash install.sh [cpu|cu118|cu121|cu124]"
        echo ""
        echo "查看 CUDA 版本: nvidia-smi"
        exit 1
        ;;
esac

# 验证 PyTorch
echo "[2/3] 验证 PyTorch..."
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')" || {
    echo "[ERROR] PyTorch 导入失败"
    exit 1
}

# 安装其他依赖
echo "[3/3] 安装项目依赖..."
pip install -r requirements.txt || {
    echo ""
    echo "[ERROR] 依赖安装失败"
    echo "[HINT] 尝试: pip install --upgrade pip && pip install -r requirements.txt"
    exit 1
}

# 以开发模式安装 AtomLoRA
echo "[+] 安装 atomlora 命令..."
pip install -e . || {
    echo "[ERROR] atomlora 安装失败"
    exit 1
}

echo ""
echo "=== 安装完成 ==="
echo "验证: atomlora --help"
echo "训练: atomlora train --config configs/demo.yaml"
