#!/usr/bin/env bash
# Modern LeNet-5 with tanh: committee recovery.
#   bash run_lenet_tanh.sh [GPU] [Q] [TAG]
# Architecture (--conv lenet --cnn-act tanh): 28x28 MNIST input (C1 pads by 2), tanh in every hidden layer, C1 6@5x5,
# 2x2 average pooling, C3 16@5x5 FULLY connected, avg pool, C5 120@5x5, F6 84, linear 10-way output.
# Gauges handled: free permutations of every hidden layer + odd sign flips of every unit (test_lenet5_isomorphisms.py tanh).
# No kink refiner for tanh: recovery = committee + quorum consensus (no pinning).
set -euo pipefail
GPU="${1:-2}"; Q="${2:-20000}"; TAG="${3:-lenet_tanh_q${Q}}"
cd "$(dirname "$0")"
python test_lenet5_isomorphisms.py tanh | tail -1                # gauge sanity check first (seconds)
CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup python run.py \
  --variant mergedbest --conv lenet --cnn-act tanh --device cuda \
  --outer 60 --window 20 --q "$Q" --qg-chunk-gb 4 \
  --cluster-eps 0.02 --masked-consensus --layer-consensus \
  --log-every 1 --pop-save-every 5 --tag "$TAG" > "pool_logs/$TAG.log" 2>&1 &
echo "launched $TAG on GPU $GPU (pid $!): tail -f pool_logs/$TAG.log"
