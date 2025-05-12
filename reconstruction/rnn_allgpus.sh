#!/bin/bash

trap 'kill 0' SIGINT

# hidden size 1, hidden size 2, hidden size 3, seed, comment
CUDA_VISIBLE_DEVICES=0 python reconstruction/reconstruct.py $4 RNNx$1 55 10000 5 mnist adam relu $5 28 &
# CUDA_VISIBLE_DEVICES=1 python reconstruction/reconstruct.py $4 RNNx$2 55 10000 5 mnist adam relu $5 28 &
CUDA_VISIBLE_DEVICES=2 python reconstruction/reconstruct.py $4 RNNx$3 55 10000 5 mnist adam relu $5 28 &
wait