#!/usr/bin/env bash
# ONE-SHOT (non-adaptive) baseline for run_query_ablation_tanh5L.sh on the 3072x200x200x200x100 tanh (CIFAR-100) net: all Q x iterations random
# queries are drawn in iteration 1 (--random-queries <kind> --oneshot, replay window off) and the committee
# trains on that fixed set for the same iterations x epochs as the iterative runs. No disagreement search.
#   osuniform  U[-1,1]     osgauss  N(0,0.5^2)     osdatacov  Gaussian with the training inputs' mean + covariance
# Settings, budget list and table: as run_query_ablation_tanh5L.sh; results print in the same table as the iterative runs.
#   bash run_oneshot_ablation_tanh5L.sh [GPU] [Q[,Q2,...]] [RUN ...]
set -euo pipefail
GPU="${1:-0}"; QLIST="${2:-48000}"; shift $(( $# > 2 ? 2 : $# )) || true
RUNS=("$@"); [ ${#RUNS[@]} -eq 0 ] && RUNS=(osuniform osgauss osdatacov)
cd "$(dirname "$0")"
mkdir -p results/logs
IFS=',' read -ra QS <<< "$QLIST"

for Q in "${QS[@]}"; do
COMMON="--variant mergedbest512 --tanh --device cuda --arch 3072,200,200,200,100 --q $Q --window 30 --outer 30
        --teacher-seed 0 --seed 0 --layer-consensus --align-inverse --log-every 1"

for r in "${RUNS[@]}"; do
  case "$r" in
    osuniform|osgauss|osdatacov) EXTRA="--random-queries ${r#os} --oneshot" ;;
    *) echo "unknown run $r (osuniform osgauss osdatacov)"; exit 1 ;;
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
