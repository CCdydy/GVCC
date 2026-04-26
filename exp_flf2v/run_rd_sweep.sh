#!/bin/bash
# FLF2V R-D Sweep — 720p, joint (M, K, ref_quality) scaling
# Traces a rate-distortion curve on UVG 720p (3 GOPs per sequence)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "============================================================"
echo " FLF2V R-D Sweep — 720p"
echo " Started: $(date)"
echo "============================================================"

python ${SCRIPT_DIR}/run_rd_sweep.py \
    --wan_ckpt ${SCRIPT_DIR}/Wan2.1-FLF2V-14B-720P \
    --data_dir ${PROJECT_DIR}/data/uvg \
    --output_dir ${SCRIPT_DIR}/rd_sweep_720p \
    --height 720 --width 1280 \
    --num_gops 3 \
    --steps 20 --ddim_tail 3 \
    --g_scale 3.0 \
    --ref_codec compressai \
    --seed 42 \
    "$@"

echo "============================================================"
echo " FLF2V R-D Sweep finished: $(date)"
echo "============================================================"
