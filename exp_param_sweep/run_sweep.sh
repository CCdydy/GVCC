#!/bin/bash
# T2V Parameter Sweep — find optimal M, K, num_frames, steps, g_scale
# Baseline: 720p, Beauty, 1 GOP
#
# Usage:
#   bash run_sweep.sh                              # all sweeps
#   bash run_sweep.sh --sweeps M K                 # only M and K
#   bash run_sweep.sh --sweep_M 32,64,128          # custom M range
#   bash run_sweep.sh --sequence Jockey            # different sequence

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

python ${SCRIPT_DIR}/run_sweep.py \
    --wan_ckpt ${PROJECT_DIR}/exp_t2v/Wan2.1-T2V-14B-Diffusers \
    --data_dir ${PROJECT_DIR}/data/uvg \
    --output_dir ${SCRIPT_DIR}/outputs \
    --sequence Beauty \
    --height 720 --width 1280 \
    --seed 42 \
    --guidance_scale 1.0 \
    --flow_shift 3.0 \
    --ddim_tail 3 \
    --base_M 64 --base_K 16384 --base_num_frames 33 --base_steps 20 --base_g_scale 3.0 \
    --sweep_M "16,32,64,128,256" \
    --sweep_K "1024,4096,16384,65536" \
    --sweep_num_frames "17,33,49" \
    --sweep_steps "5,10,15,20,30" \
    --sweep_g_scale "1.0,2.0,3.0,5.0,8.0" \
    "$@"
