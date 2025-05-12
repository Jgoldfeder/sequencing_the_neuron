#!/bin/bash

# device, seed, samples, comment
CUDA_VISIBLE_DEVICES=$1 python reconstruction/reconstruct.py $2 3072x1024x100 55 $3 25 cifar100 adam relu $4 3072