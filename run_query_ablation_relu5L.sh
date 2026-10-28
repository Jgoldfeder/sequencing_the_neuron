#!/usr/bin/env bash
# Query-sampler ablation on the leaky-ReLU twin of the tanh ablation net: 3072 -> 200x3 -> 100 (CIFAR-100 teacher).
# Same committee stage as run_query_ablation_tanh5L.sh (mergedbest512, window 30, 30 iterations, --layer-consensus),
# minus the tanh-only flags (--tanh, --align-inverse). Earlier disagreement runs at q 48k: seed 0 max eps 6.9
# (L1/L2 exact, L3 + head failed), seed 1 max eps 2.0e-3 / mean 2.6e-5 -> borderline, seed-dependent.
#   disagree  disagreement sampling (the method; search starts from U[-1,1])
#   uniform   random U[-1,1]          gauss    random N(0,0.5^2)
#   datacov   random Gaussian with the CIFAR-100 training inputs' mean and covariance
#   dgauss    disagreement, search started from N(0,0.5^2)
#   ddatacov  disagreement, search started from the CIFAR-100 mean + covariance Gaussian
#   leverage  informed design, NO disagreement: best member's layer-input leverage (query_design.py); not in the default set
#   levseq    leverage + batch design (chunks of 2048 see the earlier chunks); levm3seq: + score averaged over 3 best members
# Committee only (no refiner / polish). Same budget for all: Q queries x 30 iterations.
#   bash run_query_ablation_relu5L.sh [GPU] [Q[,Q2,...]] [RUN ...]   (SEED=1 bash ... for another committee seed)
#   bash run_query_ablation_relu5L.sh 0 16000,32000,48000   # every run at q 16k, then all at 32k, then 48k
#   bash run_query_ablation_relu5L.sh 0                     # all six, one after the other, q 48000
#   bash run_query_ablation_relu5L.sh 1 16000 disagree gauss
set -euo pipefail
GPU="${1:-0}"; QLIST="${2:-48000}"; shift $(( $# > 2 ? 2 : $# )) || true
SEED="${SEED:-0}"
RUNS=("$@"); [ ${#RUNS[@]} -eq 0 ] && RUNS=(disagree uniform gauss datacov dgauss ddatacov)
cd "$(dirname "$0")"
mkdir -p results/logs
IFS=',' read -ra QS <<< "$QLIST"

for Q in "${QS[@]}"; do
COMMON="--variant mergedbest512 --device cuda --arch 3072,200,200,200,100 --q $Q --window 30 --outer 30
        --teacher-seed 0 --seed $SEED --layer-consensus --log-every 1"

for r in "${RUNS[@]}"; do
  case "$r" in
    disagree) EXTRA="" ;;
    uniform|gauss|datacov) EXTRA="--random-queries $r" ;;
    dgauss)   EXTRA="--qg-init gauss" ;;
    ddatacov) EXTRA="--qg-init datacov" ;;
    leverage) EXTRA="--query-design leverage" ;;
    levseq)   EXTRA="--query-design leverage --qd-seq --qd-chunk 2048" ;;
    levm3seq) EXTRA="--query-design leverage --qd-seq --qd-chunk 2048 --qd-members 3" ;;
    levseqdc) EXTRA="--query-design leverage --qd-seq --qd-chunk 2048 --qg-init datacov" ;;
    levseqnox) EXTRA="--query-design leverage --qd-seq --qd-chunk 2048 --qd-skip-input" ;;
    levdcr57) EXTRA="--query-design leverage --qd-seq --qd-chunk 2048 --qg-init datacov --qd-radius 57" ;;
    levdcr80) EXTRA="--query-design leverage --qd-seq --qd-chunk 2048 --qg-init datacov --qd-radius 80" ;;
    levtop4)  EXTRA="--query-design leverage --qd-seq --qd-chunk 2048 --qg-init datacov --qd-radius 80 --qd-topk 4" ;;
    levtop16) EXTRA="--query-design leverage --qd-seq --qd-chunk 2048 --qg-init datacov --qd-radius 80 --qd-topk 16" ;;
    levres50)  EXTRA="--query-design leverage --qd-seq --qd-chunk 2048 --qg-init datacov --qd-radius 80 --qd-resid-frac 0.5" ;;
    levres100) EXTRA="--query-design leverage --qd-seq --qd-chunk 2048 --qg-init datacov --qd-radius 80 --qd-resid-frac 1.0" ;;
    kinka)    EXTRA="--query-design kink --qd-chunk 2048 --qd-tau 0.05 --qd-lr 0.1" ;;
    kinkb)    EXTRA="--query-design kink --qd-chunk 2048 --qd-tau 0.05 --qd-lr 0.03" ;;
    kinkc)    EXTRA="--query-design kink --qd-chunk 2048 --qd-tau 0.2 --qd-lr 0.03" ;;
    kinkm3)   EXTRA="--query-design kink --qd-chunk 2048 --qd-tau 0.05 --qd-lr 0.1 --qd-members 3" ;;
    kinkmix)  EXTRA="--query-design kink --qd-chunk 2048 --qd-tau 0.05 --qd-lr 0.1 --qd-mix 0.3" ;;
    kinkwide) EXTRA="--query-design kink --qd-chunk 2048 --qd-tau 0.1 --qd-lr 0.1" ;;
    kinkm8)   EXTRA="--query-design kink --qd-chunk 2048 --qd-tau 0.05 --qd-lr 0.1 --qd-members 8" ;;
    kinkres10) EXTRA="--query-design kink --qd-chunk 2048 --qd-tau 0.05 --qd-lr 0.1 --qd-resid-frac 0.1" ;;
    kinkres20) EXTRA="--query-design kink --qd-chunk 2048 --qd-tau 0.05 --qd-lr 0.1 --qd-resid-frac 0.2" ;;
    kinkres30) EXTRA="--query-design kink --qd-chunk 2048 --qd-tau 0.05 --qd-lr 0.1 --qd-resid-frac 0.3" ;;
    *) echo "unknown run $r (disagree uniform gauss datacov dgauss ddatacov leverage levseq levm3seq levseqdc levseqnox levdcr57 levdcr80 levtop4 levtop16 levres50 levres100 kinka kinkb kinkc kinkm3 kinkmix kinkwide kinkm8 kinkres10 kinkres20 kinkres30)"; exit 1 ;;
  esac
  LOG="results/logs/ablation_relu5L_q${Q}_s${SEED}_$r.log"
  echo "== $r q=$Q seed=$SEED $EXTRA (log $LOG)"
  CUDA_VISIBLE_DEVICES="$GPU" python -u run.py $COMMON --tag "ablation_relu5L_q${Q}_$r" $EXTRA 2>&1 \
    | tee "$LOG" | grep -E "^\s+it +[0-9]+ \||Traceback|Error" || true
done

ABL_NET=relu5L ABL_Q="$Q" ABL_SEED="$SEED" python ablation_table.py
done

if [ ${#QS[@]} -gt 1 ]; then                      # all budgets again, together, at the end
  echo; echo "== all budgets"
  for Q in "${QS[@]}"; do ABL_NET=relu5L ABL_Q="$Q" ABL_SEED="$SEED" python ablation_table.py; done
fi
