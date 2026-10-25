#!/usr/bin/env bash
# Reproduce the modern tanh LeNet-5 black-box recovery end to end (2026-10-22 result: max eps 6.1e-8, mean 1.4e-9,
# 1.48M queries).  One GPU, ~12 h.  Usage:  bash reproduce_lenet_tanh.sh [GPU] [TAG]
# Network: 28x28 MNIST, tanh, avg-pool, conv1 6@5x5, conv2 16@5x5 (full C3), C5 120, F6 84, linear 10 head.
#   1 committee        run.py, q 20k x 60 iterations (1.2M queries)           -> best member ~2.5e-4
#   2 jet refiner, 1   K-FAC, 4000 points x 8 directions, 12 its (132k q)     -> ~2.9e-6
#   3 jet refiner, 2   resumed from stage 1, 15 its (132k q)                   -> ~6.1e-8
#   4 head LSQ         closed-form linear head on 20k outputs (20k q)          -> ~6.1e-8 (head 2e-8)
# GPU runs are not bit-reproducible: expect numbers close to, not equal to, the above.
set -euo pipefail
GPU="${1:-0}"; TAG="${2:-lenet_tanh_q20000}"
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p pool_logs recon
SNAP="recon/_pop__mergedbest_${TAG}_cnn__1x28x28__1-6-5-1-2-2_6-16-5-1-0-2_16-120-5-1-0-0__s0.pt"

echo "== 0 isomorphism check"
python test_lenet5_isomorphisms.py tanh | tail -1

echo "== 1 committee (log pool_logs/${TAG}.log)"
python -u run.py --variant mergedbest --conv lenet --cnn-act tanh --device cuda \
  --outer 60 --window 20 --q 20000 --qg-chunk-gb 4 \
  --cluster-eps 0.02 --masked-consensus --layer-consensus \
  --log-every 1 --pop-save-every 5 --tag "$TAG" 2>&1 | tee "pool_logs/${TAG}.log" | grep -E "^\s*it +[0-9]+ \|" || true

echo "== 2 jet refiner stage 1 (log pool_logs/${TAG}_refine1.log)"
python -u refine_cnn_snapshot.py "$SNAP" --points 4000 --dirs 8 --iters 12 --cg-iters 150 --cg-tol 1e-6 \
  --precond kfac --out "recon/_refined_${TAG}_stage1.pt" 2>&1 | tee "pool_logs/${TAG}_refine1.log" | grep -E "max_eps|\[refine\]"

echo "== 3 jet refiner stage 2 (log pool_logs/${TAG}_refine2.log)"
python -u refine_cnn_snapshot.py "recon/_refined_${TAG}_stage1.pt" --points 4000 --dirs 8 --iters 15 --cg-iters 150 \
  --cg-tol 1e-6 --precond kfac --out "recon/_refined_${TAG}_stage2.pt" 2>&1 | tee "pool_logs/${TAG}_refine2.log" | grep -E "max_eps|\[refine\]"

echo "== 4 head least squares"
python -u head_lsq_cnn.py "recon/_refined_${TAG}_stage2.pt" --out "recon/_refined_${TAG}_final.pt" 2>&1 | grep "head-lsq"

echo "== done: final net recon/_refined_${TAG}_final.pt"
