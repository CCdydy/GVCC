#!/bin/bash
# Download Wan2.1 FLF2V 14B model (First-Last-Frame to Video)
# Source: https://huggingface.co/Wan-AI/Wan2.1-FLF2V-14B-720P

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TARGET_DIR="${SCRIPT_DIR}/Wan2.1-FLF2V-14B-720P"

python -m pip install -U huggingface_hub

# Remove broken symlink target if present
if [ -L "$TARGET_DIR" ] && [ ! -d "$TARGET_DIR" ]; then
    echo "Removing broken symlink at $TARGET_DIR"
    rm -f "$TARGET_DIR"
fi

# Clean stale download cache to avoid race-condition FileNotFoundError
# (huggingface_hub #2374, #2355)
rm -rf "${TARGET_DIR}/.cache"
mkdir -p "${TARGET_DIR}"

python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'Wan-AI/Wan2.1-FLF2V-14B-720P',
    local_dir='${TARGET_DIR}',
    max_workers=1,
)
"
