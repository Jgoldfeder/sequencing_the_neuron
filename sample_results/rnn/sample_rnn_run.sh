#!/bin/bash
#gpu_num, comment
CUDA_VISIBLE_DEVICES=$1 python reconstruction/reconstruct.py 31 RNNx28 55 10000 5 mnist adam relu $2 28