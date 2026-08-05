#!/bin/zsh
# v17 attribution ablations on 784x32x10.
cd "$(dirname "$0")"
PY=../.venv/bin/python
for v in v17a_nolf v17b_nopop v17c_nomse v17d_noll; do
  for s in 0 1; do
    echo "=== $v seed $s $(date +%H:%M:%S)"
    $PY run.py --variant $v --seed $s --arch 784,32,10 --outer 30 --q 1500 --p 8
  done
done
echo "ALL DONE"
