#!/usr/bin/env bash
# Reproduce the solved-activation table (784x128x10 MNIST teacher, 25 teacher epochs,
# 40 outer iterations, 8 members): every activation at the smallest query budget q
# that solved it (total queries = 40 q), then print the table from the result files.
#   ./reproduce_activation_table.sh            # run everything (~15 h on one GPU), then print
#   ./reproduce_activation_table.sh --table    # only print the table from existing results
#   DEVICE=cuda:1 ./reproduce_activation_table.sh
set -u
cd "$(dirname "$0")"
if [[ "${1:-}" != "--table" ]]; then
    ACTS="leaky_relu tanh square silu softsign abs pow3 prelu" ./run_activations.sh 5000
    ACTS="softplus hardswish sigmoid sine squareplus"          ./run_activations.sh 20000
    ACTS="elu pow5"                                           ./run_activations.sh 50000
fi
python make_activation_table.py
