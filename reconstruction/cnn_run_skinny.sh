#!/bin/bash

#graphics card num, seed, num_samples, comment
CUDA_VISIBLE_DEVICES=$1 python reconstruction/reconstruct.py $2 CNNx1-16-3-1x16-8-3-1x8-4-3-2x4-3-3-2 55 $3 5 mnist adam relu $4 1x28x28