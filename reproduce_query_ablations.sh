#!/usr/bin/env bash
# Reproduce the query-sampler ablation tables (2026-10-26/27): disagreement (3 search starts), random (3 distributions)
# and one-shot random, on four nets. Leverage / kink designs are NOT included.
#   bash reproduce_query_ablations.sh [GPU_A] [GPU_B] [GPU_C]        (default 0 1 2; three lanes run in parallel)
#     lane A: MNIST 784x128x10 leaky ReLU + tanh, then the deep-tanh one-shot runs
#     lane B: deep tanh 3072x200x200x200x100, iterative samplers
#     lane C: deep leaky ReLU 3072x200x200x200x100: iterative samplers, the 6-seed disagreement check, one-shot
# Runs (names as in the scripts): disagree dgauss ddatacov = disagreement searched from U[-1,1] / N(0,.5^2) / data cov;
#   uniform gauss datacov = random; osuniform osgauss osdatacov = one-shot random (all queries drawn in iteration 1).
# Each run's end-of-run checkpoint (recon/<tag>.pt) holds the whole query buffer (up to 18 GB for the 3072-input nets);
# strip_buffers.py removes X/Y after every sweep (weights/config kept, verified), so a full run does not fill the disk.
# Tables afterwards:  ABL_NET=784|784tanh|tanh5L|relu5L ABL_Q=<q> [ABL_SEED=<s>] python ablation_table.py
# Runtime on the 3-GPU box: lane A ~10 h, lane B ~15 h, lane C ~15 h. Single-seed GPU runs are not bit-reproducible.
set -uo pipefail
A="${1:-0}"; B="${2:-1}"; C="${3:-2}"
cd "$(dirname "$0")"
mkdir -p results/logs
SIX="disagree uniform gauss datacov dgauss ddatacov"

strip() {   # drop the query buffers of finished ablation checkpoints (skips files still open by a running job)
  ls recon/*ablation*.pt 2>/dev/null | grep -v "_final.pt\|_consensus.pt\|/_pop__\|/_fast__" \
    | xargs -r python strip_buffers.py --apply >> results/logs/reproduce_strip.log 2>&1 || true
}

laneA() {
  bash run_query_ablation_784.sh     "$A" 750,1500,3000,4500 $SIX;  strip
  bash run_query_ablation_784tanh.sh "$A" 1500,3000,4500 $SIX;      strip
  bash run_oneshot_ablation_tanh5L.sh "$A" 16000,32000,48000;       strip
}
laneB() {
  bash run_query_ablation_tanh5L.sh "$B" 16000,32000,48000 $SIX;            strip
  bash run_query_ablation_tanh5L.sh "$B" 40000 disagree uniform gauss datacov; strip
}
laneC() {
  bash run_query_ablation_relu5L.sh "$C" 16000,32000,48000 $SIX;  strip
  bash run_query_ablation_relu5L.sh "$C" 8000 ddatacov;           strip
  for s in 1 2 3 4 5; do                                          # 6-seed check of disagreement-from-data-cov at 16k
    SEED=$s bash run_query_ablation_relu5L.sh "$C" 16000 ddatacov; strip
  done
  bash run_oneshot_ablation_relu5L.sh "$C" 16000,32000,48000;     strip
}

laneA > results/logs/reproduce_laneA.log 2>&1 &
laneB > results/logs/reproduce_laneB.log 2>&1 &
laneC > results/logs/reproduce_laneC.log 2>&1 &
echo "lanes running on GPUs $A $B $C (logs results/logs/reproduce_lane{A,B,C}.log)"
wait
strip
echo "== done. Tables:"
for q in 750 1500 3000 4500; do ABL_NET=784 ABL_Q=$q python ablation_table.py; done
for q in 1500 3000 4500; do ABL_NET=784tanh ABL_Q=$q python ablation_table.py; done
for q in 16000 32000 40000 48000; do ABL_NET=tanh5L ABL_Q=$q python ablation_table.py; done
for q in 8000 16000 32000 48000; do ABL_NET=relu5L ABL_Q=$q python ablation_table.py; done
for s in 1 2 3 4 5; do ABL_NET=relu5L ABL_Q=16000 ABL_SEED=$s python ablation_table.py; done
