#!/bin/bash
# FLF2V Compression — 1080p full UVG (all frames, same as DCVC-RT benchmark)
# Uses Wan2.1 FLF2V-14B with first+last frame conditioning

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "============================================================"
echo " FLF2V 1080p — Full UVG benchmark (all frames per sequence)"
echo " Started: $(date)"
echo "============================================================"

python ${SCRIPT_DIR}/run_flf2v_experiment.py \
  --wan_ckpt ${SCRIPT_DIR}/Wan2.1-FLF2V-14B-720P \
  --data_dir ${PROJECT_DIR}/data/uvg \
  --output_dir ${SCRIPT_DIR}/results_1080p \
  --num_frames_per_gop 33 \
  --num_gops 0 \
  --height 1080 --width 1920 \
  --M 80 \
  --K 16384 \
  --steps 20 --ddim_tail 3 \
  --g_scale 3.0 \
  --ref_codec compressai --ref_quality 4 \
  --seed 42 \
  "$@"

echo "============================================================"
echo " FLF2V 1080p finished: $(date)"
echo "============================================================"
