#!/usr/bin/env bash
# LeNet-5 black-box recovery, committee + per-channel peel (the Sep-30 configuration).
#   8-member committee (mergedbest), leaky-ReLU LeNet teacher (MNIST, 25 ep, seed 0),
#   --fast-peel-partial : per-channel consensus -> kink refine -> pin in place, frontier advances when a layer is full
#   --loc-refine        : kink_solve conv refiner (forward-only, no prefix inversion)
#   --peelrefresh       : restart the iteration clock + flush the buffer when a layer completes
#   --retry             : reinit unsolved rows + deeper layers (5x) if a budget ends with the frontier unpeeled
# Usage: ./run_lenet_peel.sh [tag] [gpu]      e.g. ./run_lenet_peel.sh lenetpeel1 1
set -euo pipefail
cd "$(dirname "$0")"
TAG=${1:-lenetpeel_$(date +%m%d_%H%M)}
GPU=${2:-1}
mkdir -p pool_logs
LOG=pool_logs/${TAG}.log
echo "tag=$TAG gpu=$GPU log=$LOG"
CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
nohup python run.py --variant mergedbest --conv lenet --cnn-act leaky_relu --device cuda \
  --outer 60 --window 60 --q 20000 \
  --fast-peel-partial --peelrefresh --loc-refine --retry \
  --pop-save-every 5 --tag "$TAG" > "$LOG" 2>&1 &
echo "pid $!"
echo "follow:  tail -f $LOG | grep -E 'iter|cons|peel|solved|retry'"
