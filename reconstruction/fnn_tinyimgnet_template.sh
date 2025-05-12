#!/bin/bash

# device, seed, samples, comment
CUDA_VISIBLE_DEVICES=$1 python reconstruction/reconstruct.py $2 12288x512x200 55 $3 25 tinyimagenet adam relu $4 12288