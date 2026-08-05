#!/bin/zsh
cd "$(dirname "$0")"
PY=../.venv/bin/python

echo "Starting Scale Experiments: Wide & Deep"
# Run wide_128 first
$PY bench_scale.py wide_128
# Run wide_256 next
$PY bench_scale.py wide_256
# Run deep_3layer
$PY bench_scale.py deep_3layer
# Run deep_4layer
$PY bench_scale.py deep_4layer
# Run wide_3072
$PY bench_scale.py wide_3072

echo "ALL SCALE EXPERIMENTS COMPLETE"
