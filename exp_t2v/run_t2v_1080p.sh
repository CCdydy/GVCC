#!/bin/bash
# T2V Unconditional Compression — 1080p full UVG (all frames, same as DCVC-RT benchmark)
# Uses Wan2.1 T2V 14B with empty prompt

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "============================================================"
echo " T2V 1080p — Full UVG benchmark (all frames per sequence)"
echo " Started: $(date)"
echo "============================================================"

python ${SCRIPT_DIR}/run_t2v_experiment.py \
    --wan_ckpt ${SCRIPT_DIR}/Wan2.1-T2V-14B-Diffusers \
    --data_dir ${PROJECT_DIR}/data/uvg \
    --output_dir ${SCRIPT_DIR}/results_1080p \
    --height 1080 --width 1920 \
    --num_frames 33 \
    --num_gops 0 \
    --M 80 --K 16384 \
    --steps 20 --ddim_tail 3 \
    --g_scale 3.0 \
    --guidance_scale 1.0 \
    --flow_shift 3.0 \
    --seed 42 \
    "$@"

echo "============================================================"
echo " T2V 1080p finished: $(date)"
echo "============================================================"
