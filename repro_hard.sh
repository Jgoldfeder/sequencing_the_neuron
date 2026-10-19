#!/usr/bin/env bash
# Reproduce the hard-label run that worked (Sep 2026 benchmark):
#   results/mergedbest_hard_bench__784x16x10__s0.json
#   log: pool_logs/hard_disagree_bench_s0.log
# Teacher 784x16x10 leaky-ReLU trained 3 epochs on MNIST, argmax-only black box.
# Committee of 8, 1500 queries/iter x 40 iters = 60k queries, logit-disagreement sampler.
# Reference result: agree 0.9695, L1 eps max 1.15e-1 / mean 9.2e-3, wall ~700 s on a 3090.
# NOTE: raw L2 (output-layer) max_eps ~22 is almost entirely the unidentifiable
# label-preserving family (s*W2 + 1u^T); judge the run on agree + L1 eps, not raw max_eps.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-/opt/miniconda3/bin/python}
SEED=${SEED:-0}
DEVICE=${DEVICE:-cuda}
TAG=${TAG:-hard_bench}

mkdir -p pool_logs
$PY run.py --variant mergedbest --hard \
    --arch 784,16,10 --teacher-epochs 3 --teacher-seed 0 \
    --seed "$SEED" --outer 40 --q 1500 --p 8 \
    --pop-save-every 0 \
    --device "$DEVICE" --tag "$TAG" \
    2>&1 | tee "pool_logs/${TAG}_s${SEED}.log"
