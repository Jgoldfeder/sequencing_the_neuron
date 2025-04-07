#!/bin/bash

#graphics card num, seed, num_samples, comment
CUDA_VISIBLE_DEVICES=$1 python reconstruction/reconstruct.py $2 CNNx1-40-3-1x40-20-3-1x20-10-3-2x10-3-3-2 55 $3 5 mnist adam relu $4 1x28x28