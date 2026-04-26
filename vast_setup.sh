#!/bin/bash
# ============================================================
# GVCC 环境配置脚本
# 适用于 Vast.ai PyTorch 镜像 (CUDA 12.x)
# ============================================================

set -e
echo "======================================================"
echo " GVCC 环境配置开始"
echo "======================================================"

# —— 1. 基础工具 ————————————————————————————————————————————
echo ""
echo "[1/7] 安装系统工具..."
apt-get update -q
apt-get install -y -q git ffmpeg

# —— 2. 克隆项目 ————————————————————————————————————————————
echo ""
echo "[2/7] 克隆 GVCC 项目..."
cd /workspace

if [ -d "GVCC" ]; then
    echo "  目录已存在，执行 git pull..."
    cd GVCC && git pull && cd ..
else
    git clone https://github.com/CCdydy/GVCC.git
fi

# —— 3. 安装 Python 依赖 ————————————————————————————————————
echo ""
echo "[3/7] 安装 Python 依赖..."
cd /workspace/GVCC

# 跳过 flash_attn (部分显卡不兼容) 和 gradio (不需要)
grep -v -E "flash_attn|gradio|dashscope" requirements.txt > /tmp/req_filtered.txt
pip install -r /tmp/req_filtered.txt -q
pip install einops -q

echo "  Python 依赖安装完成"

# —— 4. 下载模型 ————————————————————————————————————————————
# 三个实验各需要一个 14B 模型，各约 30GB
# 按需下载：只下载你要跑的实验的模型
echo ""
echo "[4/7] 下载 Wan2.1 模型..."

# ---- I2V-14B (Native 格式) ----
I2V_DIR="/workspace/GVCC/exp_i2v/Wan2.1-I2V-14B-720P"
if [ -d "$I2V_DIR" ] && [ "$(ls $I2V_DIR/*.safetensors 2>/dev/null | wc -l)" -ge 1 ]; then
    echo "  I2V 模型已存在，跳过"
else
    echo "  下载 Wan2.1-I2V-14B-720P (~30GB)..."
    python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Wan-AI/Wan2.1-I2V-14B-720P',
    local_dir='$I2V_DIR',
    max_workers=1
)
print('  I2V 模型下载完成!')
"
fi

# ---- T2V-14B (Diffusers 格式) ----
T2V_DIR="/workspace/GVCC/exp_t2v/Wan2.1-T2V-14B-Diffusers"
if [ -d "$T2V_DIR" ] && [ "$(ls $T2V_DIR/ 2>/dev/null | wc -l)" -ge 5 ]; then
    echo "  T2V 模型已存在，跳过"
else
    echo "  下载 Wan2.1-T2V-14B (Diffusers 格式, ~30GB)..."
    python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Wan-AI/Wan2.1-T2V-14B-Diffusers',
    local_dir='$T2V_DIR',
    max_workers=1
)
print('  T2V 模型下载完成!')
"
fi

# ---- FLF2V-14B (Native 格式) ----
FLF2V_DIR="/workspace/GVCC/exp_flf2v/Wan2.1-FLF2V-14B-720P"
if [ -d "$FLF2V_DIR" ] && [ "$(ls $FLF2V_DIR/*.safetensors 2>/dev/null | wc -l)" -ge 1 ]; then
    echo "  FLF2V 模型已存在，跳过"
else
    echo "  下载 Wan2.1-FLF2V-14B-720P (~30GB)..."
    python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Wan-AI/Wan2.1-FLF2V-14B-720P',
    local_dir='$FLF2V_DIR',
    max_workers=1
)
print('  FLF2V 模型下载完成!')
"
fi

# —— 5. 准备 UVG 数据集 —————————————————————————————————————
echo ""
echo "[5/7] 准备数据集目录..."
DATA_DIR="/workspace/GVCC/data/uvg"
mkdir -p "$DATA_DIR"

# 检查是否已有 YUV 文件
YUV_COUNT=$(find "$DATA_DIR" -name "*.yuv" 2>/dev/null | wc -l)
if [ "$YUV_COUNT" -gt 0 ]; then
    echo "  找到 $YUV_COUNT 个 YUV 文件"
else
    echo ""
    echo "  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "  !  未找到 UVG 数据集!                          !"
    echo "  !  请手动上传 .yuv 文件到:                     !"
    echo "  !  /workspace/GVCC/data/uvg/             !"
    echo "  !                                              !"
    echo "  !  文件名示例:                                 !"
    echo "  !  Beauty_1920x1080_120fps_420_8bit_YUV.yuv    !"
    echo "  !  Bosphorus_1920x1080_120fps_420_8bit_YUV.yuv !"
    echo "  !  ...                                         !"
    echo "  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
fi

# —— 6. 给脚本加执行权限 ———————————————————————————————————
echo ""
echo "[6/7] 设置脚本权限..."
chmod +x /workspace/GVCC/exp_i2v/run.sh
chmod +x /workspace/GVCC/exp_i2v/run_1080p.sh
chmod +x /workspace/GVCC/exp_t2v/run_t2v.sh
chmod +x /workspace/GVCC/exp_t2v/run_t2v_1080p.sh
chmod +x /workspace/GVCC/exp_flf2v/run_flf2v.sh
chmod +x /workspace/GVCC/exp_flf2v/run_flf2v_1080p.sh

# —— 7. 验证环境 ———————————————————————————————————————————
echo ""
echo "[7/7] 验证环境..."
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA 可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'  显存: {mem:.1f} GB')
    if mem < 45:
        print(f'  ⚠ 显存不足 48GB，720p 可能 OOM')
    if mem < 70:
        print(f'  ⚠ 显存不足 70GB，1080p 可能 OOM')
import einops, lpips, pytorch_msssim, compressai
print('  einops / lpips / pytorch_msssim / compressai: OK')
import diffusers, transformers
print(f'  diffusers: {diffusers.__version__}')
print(f'  transformers: {transformers.__version__}')
"

echo ""
echo "======================================================"
echo " 配置完成!"
echo ""
echo " 数据集放在: /workspace/GVCC/data/uvg/"
echo ""
echo " 运行实验:"
echo "   cd /workspace/GVCC"
echo ""
echo "   # I2V 720p (需要 ~48GB 显存)"
echo "   bash exp_i2v/run.sh"
echo ""
echo "   # T2V 720p"
echo "   bash exp_t2v/run_t2v.sh"
echo ""
echo "   # FLF2V 720p"
echo "   bash exp_flf2v/run_flf2v.sh"
echo ""
echo "   # 1080p 版本 (需要 ~70GB 显存)"
echo "   bash exp_i2v/run_1080p.sh"
echo "   bash exp_t2v/run_t2v_1080p.sh"
echo "   bash exp_flf2v/run_flf2v_1080p.sh"
echo "======================================================"
