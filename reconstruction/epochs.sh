#!/bin/bash

# device, seed
CUDA_VISIBLE_DEVICES=$1 python reconstruct.py $2 784x128x10 55 10000 5 mnist adam relu

CUDA_VISIBLE_DEVICES=$1 python reconstruct.py $2 784x128x10 55 10000 25 mnist adam relu

CUDA_VISIBLE_DEVICES=$1 python reconstruct.py $2 784x128x10 55 10000 50 mnist adam relu

CUDA_VISIBLE_DEVICES=$1 python reconstruct.py $2 784x128x10 55 10000 100 mnist adam relu

CUDA_VISIBLE_DEVICES=$1 python reconstruct.py $2 784x128x10 55 10000 500 mnist adam relu


