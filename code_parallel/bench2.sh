#!/bin/zsh
# Batch-2 benchmark matrix. Usage: bench2.sh <arch> <seeds...>
cd "$(dirname "$0")"
PY=../.venv/bin/python
ARCH=${1:-784,32,10}
shift
SEEDS=${@:-"0 1"}
VARIANTS=(v6_polish v11_mse v12_lastlayer v13_popavg v14_lbfgs v15_gate v16_combo2 v10b_earlystop v17_full)
for v in $VARIANTS; do
  for s in $=SEEDS; do
    echo "=== $v seed $s arch $ARCH $(date +%H:%M:%S)"
    $PY run.py --variant $v --seed $s --arch $ARCH --outer 30 --q 1500 --p 8
  done
done
echo "ALL DONE"
