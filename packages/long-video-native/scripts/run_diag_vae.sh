#!/usr/bin/env bash
# Run the VAE decode OOM diag in the background, capture all output.
# Usage:
#   bash packages/long-video-native/scripts/run_diag_vae.sh           # defaults
#   SPT=64 SPO=32 TPT=16 TPO=8 MEM_EFF=1 bash .../run_diag_vae.sh
set -u
LOG=/tmp/diag_vae_decode.log
PIDF=/tmp/diag_vae_decode.pid
rm -f "$LOG" "$PIDF"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SPT="${SPT:-64}"
export SPO="${SPO:-32}"
export TPT="${TPT:-16}"
export TPO="${TPO:-8}"
export MEM_EFF="${MEM_EFF:-1}"
echo "=== diag start $(date +%T) SPT=$SPT SPO=$SPO TPT=$TPT TPO=$TPO MEM_EFF=$MEM_EFF ===" | tee -a "$LOG"
nohup python -u packages/long-video-native/scripts/diag_vae_decode_oom.py \
    >>"$LOG" 2>&1 &
echo $! >"$PIDF"
echo "pid=$(cat $PIDF) log=$LOG"
