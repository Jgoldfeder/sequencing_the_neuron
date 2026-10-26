#!/bin/bash
# Sample complexity of black-box recovery vs the teacher's L1 penalty.
# Same committee protocol for every lambda; every iteration logged + every committee
# snapshot kept (POP_SAVE_ALL), so queries-to-threshold is computed offline by
#   python sample_complexity.py --arch 12288,1024,200 --runs l1_0 l11e-07 l11e-06 l11e-05
#
#   bash l1_sweep.sh <gpu> <lambda> [<lambda> ...]     e.g. bash l1_sweep.sh 2 0 1e-4 p0.03
# lambda 0 = the existing unregularized teacher (tagged l1_0 so it doesn't clobber old results).
# pX = proximal L1 (exact zeros, --teacher-l1-prox) with lambda X; plain X = L1 penalty.
set -e
cd "$(dirname "$0")"
GPU=$1; shift
ARCH=${ARCH:-12288,1024,200}
Q=${Q:-24000}; OUTER=${OUTER:-30}; WINDOW=${WINDOW:-0}; P=${P:-3}
mkdir -p runs
for L in "$@"; do
  if [ "$L" = 0 ]; then TAG="--tag l1_0"; else TAG=""; fi
  PROX=""; LAM=$L
  if [ "${L:0:1}" = p ]; then PROX="--teacher-l1-prox"; LAM=${L:1}; fi
  echo "=== lambda=$L gpu=$GPU q=$Q outer=$OUTER window=$WINDOW p=$P ==="
  POP_SAVE_ALL=1 python run.py --variant v18_lbfgs --arch $ARCH --teacher-l1 $LAM $PROX $TAG \
    --p $P --q $Q --outer $OUTER --window $WINDOW --log-every 1 --pop-save-every 1 \
    --device cuda:$GPU 2>&1 | tee runs/l1sweep_${ARCH//,/x}_l1$L.log
done
