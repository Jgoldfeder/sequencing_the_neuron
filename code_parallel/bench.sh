#!/bin/zsh
# Sequential benchmark matrix (wall-clock fairness). Usage: bench.sh <arch> <seeds...>
cd "$(dirname "$0")"
PY=../.venv/bin/python
ARCH=${1:-784,64,10}
shift
SEEDS=${@:-"0 1"}
VARIANTS=(v0_baseline v0a_underfit v1a_minpair v1b_medianpair v1c_variance v2_box v3_div v4_window v5_maint v6_polish v8_warmstart v9_fitdelta v10_earlystop v7_combo)
for v in $VARIANTS; do
  for s in $=SEEDS; do
    echo "=== $v seed $s arch $ARCH $(date +%H:%M:%S)"
    $PY run.py --variant $v --seed $s --arch $ARCH --outer 30 --q 1500 --p 8
  done
done
echo "ALL DONE"
