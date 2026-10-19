#!/usr/bin/env bash
# Run the committee extraction once per hidden activation (teacher + committee share it).
# Usage:  ./run_activations.sh [extra run.py args...]
#   ACTS="tanh gelu swiglu" ./run_activations.sh      # subset (space-separated)
#   DEVICE=cuda:1 ./run_activations.sh                # device (default cuda)
# Results land in results/mergedbest_<act>_act_<act>__784x128x10__s0.json (tag = act_<act>);
# stdout/stderr of every run goes to logs/act_<act>.log.
set -u
cd "$(dirname "$0")"
DEVICE="${DEVICE:-cuda}"
ALL_ACTS="leaky_relu sigmoid tanh square elu gelu silu mish softplus softsign celu selu hardswish squareplus abs pow3 pow5 sine softmax prelu glu swiglu"
ACTS="${ACTS:-$ALL_ACTS}"
mkdir -p logs
for act in $ACTS; do
    case "$act" in                     # the first four have their own flags; the rest go through --act
        leaky_relu) flag="" ;;
        sigmoid)    flag="--sigmoid" ;;
        tanh)       flag="--tanh" ;;
        square)     flag="--square" ;;
        *)          flag="--act $act" ;;
    esac
    echo "=== $act ($(date '+%F %T')) ==="
    python run.py --variant mergedbest $flag --tag "act_$act" \
        --arch 784,128,10 --teacher-epochs 3 --outer 40 --q 1500 --p 8 \
        --pop-save-every 0 --device "$DEVICE" "$@" 2>&1 | tee "logs/act_$act.log" | grep -E "^\[setup\]|^\[done\]|^\[act\]|Error|Traceback"
done
