#!/bin/bash

# device, seed
CUDA_VISIBLE_DEVICES=$1 python reconstruct.py $2 784x128x10 55 10000 25 mnist sgd relu

CUDA_VISIBLE_DEVICES=$1 python reconstruct.py $2 784x128x10 55 10000 25 mnist rmsprop relu

CUDA_VISIBLE_DEVICES=$1 python reconstruct.py $2 784x128x10 55 10000 25 mnist adam relu

