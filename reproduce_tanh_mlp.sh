#!/usr/bin/env bash
# Reproduce the deep tanh MLP recoveries (results/RECORD_tanh_demo_campaign_2026-10-16.md), one depth per call.
#   bash reproduce_tanh_mlp.sh DEPTH [GPU] [TAG]          DEPTH in 5L 6L 8L 9L 10L
# Teacher seed 0, input 3072, hidden width 200, output 100 ("8L" = 8 layers incl. input and output = 6 hidden).
# Pipeline (solve_tanh.py): committee -> layer-1 isolation + tail-decay scales (deep nets only) -> K-FAC jet refine
# (layer 1 frozen) -> K-FAC polish (frozen) -> K-FAC final polish (all free).  5L/6L also get the extra 8-step
# all-free polish of the record.  Recorded results (max / mean eps, total queries, wall on a shared box):
#   5L  4.2e-11 / 2.8e-12   1.75M    ~1.5 h        8L   1.9e-6 / 8.9e-9   137.4M   ~7.7 h
#   6L  3.9e-8  / 1.4e-9    3.24M    ~1.5 h        9L   1.7e-6 / 3.4e-8   182.9M   ~14 h
#                                                  10L  2.9e-6 / 6.2e-8   181.5M   ~14 h
# 8L-10L use the --disk replay ring: free disk needed ~130 GB (8L), ~240 GB (9L), ~330 GB (10L) under rings/.
# GPU runs are not bit-reproducible: expect numbers close to, not equal to, the record.
set -euo pipefail
DEPTH="${1:?usage: reproduce_tanh_mlp.sh DEPTH [GPU] [TAG]}"; GPU="${2:-0}"; TAG="${3:-repro_$DEPTH}"
cd "$(dirname "$0")"
mkdir -p results/logs recon rings
S="--refine-cg 2000 --refine-iters 8 --polish-cg 2000 --polish-iters 6 --final-cg 2000 --final-iters 4"
DISK="--disk rings/$TAG --disk-ram-gb 50"
BIG="--window 20 --epochs 5 $DISK --qg-block-gb 1.61 --qg-chunk-gb 6.5"
case "$DEPTH" in
  5L)  ARCH=3072,200,200,200,100;                         OPTS="--q 48000 --window 30" ;;
  6L)  ARCH=3072,200,200,200,200,100;                     OPTS="--q 96000 --window 30" ;;
  8L)  ARCH=3072,200,200,200,200,200,200,100;             OPTS="--q 300000 --window 30 $DISK" ;;
  9L)  ARCH=3072,200,200,200,200,200,200,200,100;         OPTS="--q 1200000 $BIG" ;;
  10L) ARCH=3072,200,200,200,200,200,200,200,200,100;     OPTS="--q 1200000 $BIG" ;;
  *) echo "unknown depth $DEPTH (5L 6L 8L 9L 10L)"; exit 1 ;;
esac
LOG="results/logs/solve_tanh__$TAG.log"
echo "== $DEPTH: arch $ARCH, GPU $GPU, tag $TAG (log $LOG)"
python -u solve_tanh.py --arch "$ARCH" --tag "$TAG" --gpus "$GPU" $OPTS $S 2>&1 | tee "$LOG" | grep -E "^\[solve_tanh\] == |final net"

if [ "$DEPTH" = "5L" ] || [ "$DEPTH" = "6L" ]; then
  # extra 8-step all-free K-FAC polish (512 points x 64 directions, +131,584 queries), as in the record
  AT=$(echo "$ARCH" | tr ',' 'x')
  POP="recon/_pop__mergedbest512_tanh_${TAG}__${AT}__s0.pt"
  echo "== extra all-free polish -> recon/_final2__$TAG.pt (log results/logs/final2__$TAG.log)"
  PRECOND=kfac CUDA_VISIBLE_DEVICES="$GPU" python -u peel/polish_full.py "$POP" "recon/_final__$TAG.pt" \
    "recon/_final2__$TAG.pt" 512 64 8 2000 2>&1 | tee "results/logs/final2__$TAG.log" | grep -E "before:|after:"
fi
echo "== done ($DEPTH)"
