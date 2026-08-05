#!/bin/zsh
# Re-run of crashed batch-2 variants after float64 fixes.
cd "$(dirname "$0")"
PY=../.venv/bin/python
ARCH=${1:-784,32,10}
for v in v6_polish v12_lastlayer v17_full; do
  for s in 0 1; do
    echo "=== $v seed $s arch $ARCH $(date +%H:%M:%S)"
    $PY run.py --variant $v --seed $s --arch $ARCH --outer 30 --q 1500 --p 8
  done
done
echo "ALL DONE"
