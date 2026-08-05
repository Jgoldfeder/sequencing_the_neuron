#!/bin/zsh
# Deep-architecture stress test: 4-layer net, same 45k-query budget.
cd "$(dirname "$0")"
PY=../.venv/bin/python
ARCH=784,64,32,16,10
for v in v0_baseline v13_popavg v14_lbfgs v17_full v17d_noll v18_min; do
  for s in 0 1; do
    echo "=== $v seed $s arch $ARCH $(date +%H:%M:%S)"
    $PY run.py --variant $v --seed $s --arch $ARCH --outer 30 --q 1500 --p 8
  done
done
echo "ALL DONE"
