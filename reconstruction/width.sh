#!/bin/bash

# device, seed

# FNN: 
CUDA_VISIBLE_DEVICES=$1 python reconstruct.py $2 fnn 784x128x10 55 10000 25 mnist adam relu

CUDA_VISIBLE_DEVICES=$1 python reconstruct.py $2 fnn 784x128x10 55 15000 25 mnist adam tanh


CUDA_VISIBLE_DEVICES=$1 python reconstruct.py $2 fnn 784x256x10 55 30000 25 mnist adam relu

CUDA_VISIBLE_DEVICES=$1 python reconstruct.py $2 fnn 784x256x10 55 30000 25 mnist adam tanh


# CNN: (changing kernel size or num kernels.)
# Format of model: kernel_widthxkernel_heightxkernel_depthxnum_kernels 
CUDA_VISIBLE_DEVICES=$1 python reconstruct.py $2 cnn 2x2x1x3 55 15000 25 mnist adam relu

CUDA_VISIBLE_DEVICES=$1 python reconstruct.py $2 cnn 5x5x1x3 55 15000 25 mnist adam relu

CUDA_VISIBLE_DEVICES=$1 python reconstruct.py $2 cnn  5x5x3x3 55 15000 25 cifar100 adam relu

CUDA_VISIBLE_DEVICES=$1 python reconstruct.py $2 cnn  5x5x3x6 55 15000 25 cifar100 adam relu
