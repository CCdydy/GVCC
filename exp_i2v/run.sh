#!/bin/bash
# I2V AR Compression — 720p full UVG
# Autoregressive: GT first frame → decoded last frame chains across GOPs
# Tail residual correction on last latent frame (default on)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

python ${SCRIPT_DIR}/run_uvg_chained_gop.py \
  --wan_ckpt ${SCRIPT_DIR}/Wan2.1-I2V-14B-720P \
  --data_dir ${PROJECT_DIR}/data/uvg \
  --output_dir ${SCRIPT_DIR}/results_720p \
  --num_frames_per_gop 33 \
  --num_gops 3 \
  --height 720 --width 1280 \
  --M 64 --M_tail 128 \
  --K 16384 \
  --steps 20 --ddim_tail 3 \
  --g_scale 3.0 \
  --seed 42 \
  "$@"
