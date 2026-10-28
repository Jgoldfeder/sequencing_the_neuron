#!/usr/bin/env bash
# Query-sampler ablation on the 784x128x10 leaky-ReLU MNIST net (budget/window of the v18_min run that worked,
# 90k queries -> 7.2e-5), as a mergedbest512 --fast --combine run, repeated with six query sources:
#   disagree  disagreement sampling (the method; search starts from U[-1,1])
#   uniform   random U[-1,1]          gauss    random N(0,0.5^2)
#   datacov   random Gaussian with the MNIST training inputs' mean and covariance
#   dgauss    disagreement, search started from N(0,0.5^2)
#   ddatacov  disagreement, search started from the MNIST mean + covariance Gaussian
# Same budget each: up to Q queries x 60 iterations (Q default 1500 -> 90k). --fast stops at whole-net consensus
# (= solved) and runs its MSE->MAE solve; NO --gn-polish (the long Gauss-Newton polish).
#   bash run_query_ablation_784.sh [GPU] [Q[,Q2,...]] [RUN ...]
#   bash run_query_ablation_784.sh 0 1500,3000,4500        # every run at q 1500, then all at 3000, then 4500
#   bash run_query_ablation_784.sh 0 4500                  # all six, one after the other
#   bash run_query_ablation_784.sh 0 4500 dgauss ddatacov  # only these runs
set -euo pipefail
GPU="${1:-0}"; QLIST="${2:-1500}"; shift $(( $# > 2 ? 2 : $# )) || true
RUNS=("$@"); [ ${#RUNS[@]} -eq 0 ] && RUNS=(disagree uniform gauss datacov dgauss ddatacov)
cd "$(dirname "$0")"
mkdir -p results/logs
IFS=',' read -ra QS <<< "$QLIST"

for Q in "${QS[@]}"; do
COMMON="--variant mergedbest512 --arch 784,128,10 --outer 60 --q $Q --window 60 --device cuda --combine --fast"

for r in "${RUNS[@]}"; do
  case "$r" in
    disagree) EXTRA="" ;;
    uniform|gauss|datacov) EXTRA="--random-queries $r" ;;
    dgauss)   EXTRA="--qg-init gauss" ;;
    ddatacov) EXTRA="--qg-init datacov" ;;
    *) echo "unknown run $r (disagree uniform gauss datacov dgauss ddatacov)"; exit 1 ;;
  esac
  LOG="results/logs/ablation784_q${Q}_$r.log"
  echo "== $r q=$Q $EXTRA (log $LOG)"
  CUDA_VISIBLE_DEVICES="$GPU" python -u run.py $COMMON --tag "ablation_q${Q}_$r" $EXTRA 2>&1 \
    | tee "$LOG" | grep -E "^\s+it +[0-9]+ \||^\[fast\]|Traceback|Error" || true
done

ABL_NET=784 ABL_Q="$Q" python ablation_table.py
done

if [ ${#QS[@]} -gt 1 ]; then                      # all budgets again, together, at the end
  echo; echo "== all budgets"
  for Q in "${QS[@]}"; do ABL_NET=784 ABL_Q="$Q" python ablation_table.py; done
fi
