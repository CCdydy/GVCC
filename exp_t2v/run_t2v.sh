#!/bin/bash
# T2V Unconditional Compression — 720p full UVG
# Uses Wan2.1 T2V 14B with empty prompt

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

python ${SCRIPT_DIR}/run_t2v_experiment.py \
    --wan_ckpt ${SCRIPT_DIR}/Wan2.1-T2V-14B-Diffusers \
    --data_dir ${PROJECT_DIR}/data/uvg \
    --output_dir ${SCRIPT_DIR}/results_720p \
    --height 720 --width 1280 \
    --num_frames 33 \
    --num_gops 3 \
    --M 64 --K 16384 \
    --steps 20 --ddim_tail 3 \
    --g_scale 3.0 \
    --guidance_scale 1.0 \
    --flow_shift 3.0 \
    --seed 42 \
    "$@"
