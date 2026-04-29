@echo off
REM AtomLoRA 安装脚本 (Windows)
REM 用法: install.bat [cpu|cu118|cu121|cu124]

set MODE=%1
if "%MODE%"=="" set MODE=cpu

echo === AtomLoRA 安装 (模式: %MODE%) ===

if "%MODE%"=="cpu" (
    echo [1/3] 安装 PyTorch (CPU)...
    pip install torch --index-url https://download.pytorch.org/whl/cpu
) else if "%MODE%"=="cu118" (
    echo [1/3] 安装 PyTorch (CUDA 11.8)...
    pip install torch --index-url https://download.pytorch.org/whl/cu118
) else if "%MODE%"=="cu121" (
    echo [1/3] 安装 PyTorch (CUDA 12.1)...
    pip install torch --index-url https://download.pytorch.org/whl/cu121
) else if "%MODE%"=="cu124" (
    echo [1/3] 安装 PyTorch (CUDA 12.4)...
    pip install torch --index-url https://download.pytorch.org/whl/cu124
) else (
    echo 未知模式: %MODE%
    echo 用法: install.bat [cpu^|cu118^|cu121^|cu124]
    echo.
    echo 查看 CUDA 版本: nvidia-smi
    exit /b 1
)

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] PyTorch 安装失败
    echo [HINT] 常见原因:
    echo   1. CUDA 版本不匹配 — 运行 nvidia-smi 查看版本
    echo   2. 网络问题 — 尝试 pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
    echo   3. pip 版本过旧 — 运行 pip install --upgrade pip
    exit /b 1
)

echo [2/3] 验证 PyTorch...
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
if %errorlevel% neq 0 (
    echo [ERROR] PyTorch 导入失败
    exit /b 1
)

echo [3/3] 安装项目依赖...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] 依赖安装失败
    echo [HINT] 尝试: pip install --upgrade pip
    exit /b 1
)

echo [+] 安装 atomlora 命令...
pip install -e .

echo.
echo === 安装完成 ===
echo 验证: atomlora --help
echo 训练: atomlora train --config configs/demo.yaml
