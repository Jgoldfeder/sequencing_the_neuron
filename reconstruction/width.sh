#!/bin/bash

# device, seed

CUDA_VISIBLE_DEVICES=$1 python reconstruct.py $2 784x128x10 55 10000 25 mnist adam relu

CUDA_VISIBLE_DEVICES=$1 python reconstruct.py $2 784x128x10 55 15000 25 mnist adam tanh


CUDA_VISIBLE_DEVICES=$1 python reconstruct.py $2 784x256x10 55 30000 25 mnist adam relu

CUDA_VISIBLE_DEVICES=$1 python reconstruct.py $2 784x256x10 55 30000 25 mnist adam tanh


