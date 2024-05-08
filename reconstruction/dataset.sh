#!/bin/bash

# device, seed
 
# FNN: 
CUDA_VISIBLE_DEVICES=$1 python reconstruct.py $2 fnn 784x128x10 55 10000 25 mnist adam relu

CUDA_VISIBLE_DEVICES=$1 python reconstruct.py $2 fnn 784x128x10 55 10000 25 kmnist adam relu

CUDA_VISIBLE_DEVICES=$1 python reconstruct.py $2 fnn 784x128x10 55 10000 25 fmnist adam relu

CUDA_VISIBLE_DEVICES=$1 python reconstruct.py $2 fnn 3072x128x10 55 50000 25 cifar10 adam relu

CUDA_VISIBLE_DEVICES=$1 python reconstruct.py $2 fnn 3072x128x10 55 50000 25 cifar100 adam relu


# CNN: 
# Format of model: kernel_widthxkernel_heightxkernel_depthxnum_kernels 
CUDA_VISIBLE_DEVICES=$1 python reconstruct.py $2 cnn 5x5x1x3 55 15000 25 mnist adam relu

CUDA_VISIBLE_DEVICES=$1 python reconstruct.py $2 cnn  5x5x3x3 55 15000 25 cifar100 adam relu

CUDA_VISIBLE_DEVICES=$1 python reconstruct.py $2 cnn  5x5x3x3 55 15000 25 imagenet adam relu

