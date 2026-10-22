#!/usr/bin/env bash
# LeNet-5 (LeCun et al. 1998) tanh recovery with the committee method.
#   bash run_lenet5_tanh.sh [GPU] [Q] [TAG]
# Architecture (--conv lenet5): 32x32 input (MNIST padded), scaled tanh 1.7159 tanh(2a/3) in every layer, C1 6@5x5,
# S2/S4 plain 2x2 average pooling, C3 16@5x5 with LeCun's connection table, C5 120@5x5, F6 84, Euclidean RBF output
# units (teacher: fixed +-1 codes).  Gauges handled: C3-table automorphisms (12), free permutations of C5/F6, odd
# sign flips of every unit.
# No kink refiner for tanh: recovery = committee + quorum consensus (no pinning); exactness would need the jet solver
# (tanh_solve.py), which is MLP-only.
set -euo pipefail
GPU="${1:-1}"; Q="${2:-20000}"; TAG="${3:-lenet5_tanh_q${Q}}"
cd "$(dirname "$0")"
python test_lenet5_isomorphisms.py | tail -1                     # gauge sanity check first (seconds)
CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup python run.py \
  --variant mergedbest --conv lenet5 --device cuda \
  --outer 60 --window 20 --q "$Q" --qg-chunk-gb 4 \
  --cluster-eps 0.02 --masked-consensus --layer-consensus \
  --log-every 1 --pop-save-every 5 --tag "$TAG" > "pool_logs/$TAG.log" 2>&1 &
echo "launched $TAG on GPU $GPU (pid $!): tail -f pool_logs/$TAG.log"
