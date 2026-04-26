#!/bin/bash
# T2V 1.3B R-D Sweep — 720p, local (32GB VRAM)
# Joint (M, K) proportional scaling on UVG

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "============================================================"
echo " T2V 1.3B R-D Sweep — 720p (local)"
echo " Started: $(date)"
echo "============================================================"

conda run python3 ${SCRIPT_DIR}/run_rd_sweep_t2v_1_3b.py \
    --wan_ckpt ${PROJECT_DIR}/checkpoints/Wan2.1-T2V-1.3B-Diffusers \
    --data_dir ${PROJECT_DIR}/data/uvg \
    --output_dir ${SCRIPT_DIR}/results \
    --height 480 --width 848 \
    --num_frames 33 \
    --num_gops 3 \
    --steps 20 --ddim_tail 3 \
    --g_scale 3.0 \
    --guidance_scale 1.0 \
    --flow_shift 3.0 \
    --seed 42 \
    "$@"

echo "============================================================"
echo " Finished: $(date)"
echo "============================================================"
