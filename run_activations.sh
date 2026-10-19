#!/usr/bin/env bash
# Run the committee extraction once per hidden activation (teacher + committee share it).
# Usage:  ./run_activations.sh [Q] [extra run.py args...]
#   ./run_activations.sh                  # q = 1500 (default), all activations
#   ./run_activations.sh 5000             # q = 5000
#   ./run_activations.sh 5000 --outer 60  # extra args pass through to run.py
#   ACTS="tanh gelu swiglu" ./run_activations.sh      # subset (space-separated)
#   DEVICE=cuda:1 ./run_activations.sh                # device (default cuda)
# Results: results/mergedbest_<act>_act_<act>_q<Q>__784x128x10__s0.json (tag = act_<act>_q<Q>);
# full stdout/stderr of every run in logs/act_<act>_q<Q>.log; a one-line summary per
# activation (mean eps, max eps, agree) is printed and appended to logs/summary_q<Q>.txt.
set -u
cd "$(dirname "$0")"
Q=1500
if [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; then Q="$1"; shift; fi
DEVICE="${DEVICE:-cuda}"
ALL_ACTS="leaky_relu sigmoid tanh square elu gelu silu mish softplus softsign celu selu hardswish squareplus abs pow3 pow5 sine softmax prelu glu swiglu"
ACTS="${ACTS:-$ALL_ACTS}"
ARCH="784,128,10"
mkdir -p logs
SUMMARY="logs/summary_q$Q.txt"
printf "%-11s %12s %12s %8s  %s\n" "act" "mean_eps" "max_eps" "agree" "result" | tee -a "$SUMMARY"
for act in $ACTS; do
    case "$act" in                     # the first four have their own flags; the rest go through --act
        leaky_relu) flag="" ;;
        sigmoid)    flag="--sigmoid" ;;
        tanh)       flag="--tanh" ;;
        square)     flag="--square" ;;
        *)          flag="--act $act" ;;
    esac
    tag="act_${act}_q$Q"
    echo "=== $act  q=$Q  ($(date '+%F %T')) ==="
    python run.py --variant mergedbest $flag --tag "$tag" \
        --arch "$ARCH" --teacher-epochs 25 --outer 40 --q "$Q" --p 8 \
        --pop-save-every 0 --device "$DEVICE" "$@" 2>&1 | tee "logs/$tag.log" \
        | grep -E "^\[setup\]|^\[done\]|^\[act\]|^\[phases\]|Error|Traceback"
    # one-line summary from the result json (mean AND max eps)
    res=$(ls -t results/mergedbest_*"${tag}"__*__s*.json 2>/dev/null | head -1)
    if [[ -n "$res" ]]; then
        python - "$act" "$res" <<'PY' | tee -a "$SUMMARY"
import json, sys
act, path = sys.argv[1], sys.argv[2]
d = json.load(open(path))
print(f"{act:<11s} {d['final_mean_eps']:12.3e} {d['final_max_eps']:12.3e} {d['final_agree']:8.4f}  {path}")
PY
    else
        printf "%-11s %12s %12s %8s  %s\n" "$act" "n/a" "n/a" "n/a" "(no result file: see logs/$tag.log)" | tee -a "$SUMMARY"
    fi
done
echo "summary: $SUMMARY"
