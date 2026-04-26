#!/bin/bash
# =============================================================
# run_residual.sh — Latent Residual Compression Experiment
#
# Usage:
#   bash run_residual.sh                               # defaults
#   bash run_residual.sh --sequences Beauty Jockey      # specific sequences
#   bash run_residual.sh --residual_bits 4 --residual_downsample 2
# =============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

WAN_CKPT="${SCRIPT_DIR}/Wan2.1-I2V-14B-720P"
DATA_DIR="${PROJECT_DIR}/data/uvg"
OUTPUT_DIR="${SCRIPT_DIR}/results_residual"

NUM_FRAMES=33
NUM_GOPS=1
HEIGHT=720
WIDTH=1280

STEPS=20
SEED=42

REF_CODEC="compressai"
REF_QUALITY=4

# Residual settings
RES_BITS=8          # quantization bits (4/8/16)
RES_DS=1            # spatial downsample (1/2/4)

# ---------------------------------------------------------

echo "============================================"
echo "  Latent Residual Experiment"
echo "  ref=$REF_CODEC q=$REF_QUALITY"
echo "  residual: ${RES_BITS}bit ds${RES_DS}x"
echo "  Resolution: ${WIDTH}x${HEIGHT}"
echo "============================================"

python ${SCRIPT_DIR}/run_residual_experiment.py \
  --wan_ckpt "$WAN_CKPT" \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --num_frames_per_gop $NUM_FRAMES \
  --num_gops $NUM_GOPS \
  --height $HEIGHT \
  --width $WIDTH \
  --steps $STEPS \
  --ref_codec $REF_CODEC \
  --ref_quality $REF_QUALITY \
  --residual_bits $RES_BITS \
  --residual_downsample $RES_DS \
  --seed $SEED \
  "$@"
