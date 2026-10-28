#!/usr/bin/env bash
# Query-sampler ablation on the 5L deep tanh MLP 3072 -> 200x3 -> 100 (CIFAR-100 teacher, seed 0): the committee
# stage of the tanh demo (results/RECORD_tanh_demo_campaign_2026-10-16.md: q 48k, window 30, 30 iterations,
# 600/600 units certified, 2.4e-4 / 2.0e-5, 1.49M queries), repeated with six query sources:
#   disagree  disagreement sampling (the method)       uniform  random U[-1,1]
#   gauss     random N(0,0.5^2)                        datacov  random Gaussian, CIFAR-100 training mean + covariance
#   dgauss    disagreement, search started from N(0,0.5^2) (default start is U[-1,1])
#   ddatacov  disagreement, search started from the CIFAR-100 mean + covariance Gaussian
# Committee only (no refiner / polish). Same budget for all: Q queries x 30 iterations.
#   bash run_query_ablation_tanh5L.sh [GPU] [Q[,Q2,...]] [RUN ...]
#   bash run_query_ablation_tanh5L.sh 0 16000,32000,48000   # every run at q 16k, then all at 32k, then 48k
#   bash run_query_ablation_tanh5L.sh 0                     # all six, one after the other, q 48000
#   bash run_query_ablation_tanh5L.sh 1 48000 uniform gauss  # only these runs (e.g. split across GPUs)
set -euo pipefail
GPU="${1:-0}"; QLIST="${2:-48000}"; shift $(( $# > 2 ? 2 : $# )) || true
RUNS=("$@"); [ ${#RUNS[@]} -eq 0 ] && RUNS=(disagree uniform gauss datacov dgauss ddatacov)
cd "$(dirname "$0")"
mkdir -p results/logs
IFS=',' read -ra QS <<< "$QLIST"

for Q in "${QS[@]}"; do
COMMON="--variant mergedbest512 --tanh --device cuda --arch 3072,200,200,200,100 --q $Q --window 30 --outer 30
        --teacher-seed 0 --seed 0 --layer-consensus --align-inverse --log-every 1"

for r in "${RUNS[@]}"; do
  case "$r" in
    disagree) EXTRA="" ;;
    uniform|gauss|datacov) EXTRA="--random-queries $r" ;;
    dgauss)   EXTRA="--qg-init gauss" ;;
    ddatacov) EXTRA="--qg-init datacov" ;;
    *) echo "unknown run $r (disagree uniform gauss datacov dgauss ddatacov)"; exit 1 ;;
  esac
  LOG="results/logs/ablation_tanh5L_q${Q}_$r.log"
  echo "== $r q=$Q (log $LOG)"
  CUDA_VISIBLE_DEVICES="$GPU" python -u run.py $COMMON --tag "ablation_tanh5L_q${Q}_$r" $EXTRA 2>&1 \
    | tee "$LOG" | grep -E "^\s+it +[0-9]+ \||Traceback|Error" || true
done

ABL_NET=tanh5L ABL_Q="$Q" python ablation_table.py
done

if [ ${#QS[@]} -gt 1 ]; then                      # all budgets again, together, at the end
  echo; echo "== all budgets"
  for Q in "${QS[@]}"; do ABL_NET=tanh5L ABL_Q="$Q" python ablation_table.py; done
fi
