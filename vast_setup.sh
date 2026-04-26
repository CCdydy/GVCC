#!/bin/bash
# ============================================================
# GVCC environment setup script
# Target: Vast.ai PyTorch image (CUDA 12.x or 13.x)
# ============================================================

set -e
echo "======================================================"
echo " GVCC environment setup"
echo "======================================================"

# ---- 1. System tools --------------------------------------
echo ""
echo "[1/7] Installing system tools..."
apt-get update -q
apt-get install -y -q git ffmpeg

# ---- 2. Clone the repository ------------------------------
echo ""
echo "[2/7] Cloning GVCC..."
cd /workspace

if [ -d "GVCC" ]; then
    echo "  Directory exists, running git pull..."
    cd GVCC && git pull && cd ..
else
    git clone https://github.com/CCdydy/GVCC.git
fi

# ---- 3. Python dependencies -------------------------------
echo ""
echo "[3/7] Installing Python dependencies..."
cd /workspace/GVCC

# Skip flash_attn (not always compatible) — code falls back to PyTorch SDPA.
grep -v -E "flash_attn" requirements.txt > /tmp/req_filtered.txt
pip install -r /tmp/req_filtered.txt -q

echo "  Python dependencies installed."

# ---- 4. Download Wan2.1 weights ---------------------------
# Three 14B models, ~30 GB each. Comment out any you do not need.
echo ""
echo "[4/7] Downloading Wan2.1 weights..."

# I2V-14B (native format)
I2V_DIR="/workspace/GVCC/exp_i2v/Wan2.1-I2V-14B-720P"
if [ -d "$I2V_DIR" ] && [ "$(ls $I2V_DIR/*.safetensors 2>/dev/null | wc -l)" -ge 1 ]; then
    echo "  I2V weights already present, skipping."
else
    echo "  Downloading Wan2.1-I2V-14B-720P (~30 GB)..."
    python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Wan-AI/Wan2.1-I2V-14B-720P',
    local_dir='$I2V_DIR',
    max_workers=1,
)
print('  I2V download complete.')
"
fi

# T2V-14B (diffusers format)
T2V_DIR="/workspace/GVCC/exp_t2v/Wan2.1-T2V-14B-Diffusers"
if [ -d "$T2V_DIR" ] && [ "$(ls $T2V_DIR/ 2>/dev/null | wc -l)" -ge 5 ]; then
    echo "  T2V weights already present, skipping."
else
    echo "  Downloading Wan2.1-T2V-14B (diffusers format, ~30 GB)..."
    python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Wan-AI/Wan2.1-T2V-14B-Diffusers',
    local_dir='$T2V_DIR',
    max_workers=1,
)
print('  T2V download complete.')
"
fi

# FLF2V-14B (native format)
FLF2V_DIR="/workspace/GVCC/exp_flf2v/Wan2.1-FLF2V-14B-720P"
if [ -d "$FLF2V_DIR" ] && [ "$(ls $FLF2V_DIR/*.safetensors 2>/dev/null | wc -l)" -ge 1 ]; then
    echo "  FLF2V weights already present, skipping."
else
    echo "  Downloading Wan2.1-FLF2V-14B-720P (~30 GB)..."
    python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Wan-AI/Wan2.1-FLF2V-14B-720P',
    local_dir='$FLF2V_DIR',
    max_workers=1,
)
print('  FLF2V download complete.')
"
fi

# ---- 5. UVG dataset directory -----------------------------
echo ""
echo "[5/7] Preparing dataset directory..."
DATA_DIR="/workspace/GVCC/data/uvg"
mkdir -p "$DATA_DIR"

YUV_COUNT=$(find "$DATA_DIR" -name "*.yuv" 2>/dev/null | wc -l)
if [ "$YUV_COUNT" -gt 0 ]; then
    echo "  Found $YUV_COUNT .yuv file(s) in $DATA_DIR"
else
    echo ""
    echo "  ----------------------------------------------------"
    echo "  UVG dataset not found."
    echo "  Please place .yuv files under:"
    echo "    $DATA_DIR"
    echo ""
    echo "  Expected filenames, e.g.:"
    echo "    Beauty_1920x1080_120fps_420_8bit_YUV.yuv"
    echo "    Bosphorus_1920x1080_120fps_420_8bit_YUV.yuv"
    echo "    ..."
    echo "  ----------------------------------------------------"
fi

# ---- 6. Make run scripts executable -----------------------
echo ""
echo "[6/7] Setting executable bit on run scripts..."
chmod +x /workspace/GVCC/exp_i2v/run.sh
chmod +x /workspace/GVCC/exp_i2v/run_1080p.sh
chmod +x /workspace/GVCC/exp_t2v/run_t2v.sh
chmod +x /workspace/GVCC/exp_t2v/run_t2v_1080p.sh
chmod +x /workspace/GVCC/exp_flf2v/run_flf2v.sh
chmod +x /workspace/GVCC/exp_flf2v/run_flf2v_1080p.sh

# ---- 7. Environment sanity check --------------------------
echo ""
echo "[7/7] Verifying environment..."
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'  VRAM: {mem:.1f} GB')
    if mem < 45:
        print('  WARNING: less than 48 GB VRAM, 720p runs may OOM.')
    if mem < 70:
        print('  WARNING: less than 70 GB VRAM, 1080p runs may OOM.')
import einops, lpips, pytorch_msssim, compressai
print('  einops / lpips / pytorch_msssim / compressai: OK')
import diffusers, transformers
print(f'  diffusers: {diffusers.__version__}')
print(f'  transformers: {transformers.__version__}')
"

echo ""
echo "======================================================"
echo " Setup complete."
echo ""
echo " Place UVG .yuv files under: /workspace/GVCC/data/uvg/"
echo ""
echo " Run experiments:"
echo "   cd /workspace/GVCC"
echo ""
echo "   # 720p (requires ~48 GB VRAM)"
echo "   bash exp_i2v/run.sh"
echo "   bash exp_t2v/run_t2v.sh"
echo "   bash exp_flf2v/run_flf2v.sh"
echo ""
echo "   # 1080p (requires ~70 GB VRAM)"
echo "   bash exp_i2v/run_1080p.sh"
echo "   bash exp_t2v/run_t2v_1080p.sh"
echo "   bash exp_flf2v/run_flf2v_1080p.sh"
echo "======================================================"
