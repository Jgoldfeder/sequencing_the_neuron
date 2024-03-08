#!/bin/bash

# device, seed
CUDA_VISIBLE_DEVICES=$1 python reconstruct.py $2 784x128x10 55 10000 25 mnist adam relu

CUDA_VISIBLE_DEVICES=$1 python reconstruct.py $2 784x128x10 55 10000 25 kmnist adam relu

CUDA_VISIBLE_DEVICES=$1 python reconstruct.py $2 784x128x10 55 10000 25 fmnist adam relu

CUDA_VISIBLE_DEVICES=$1 python reconstruct.py $2 3072x128x10 55 50000 25 cifar10 adam relu

CUDA_VISIBLE_DEVICES=$1 python reconstruct.py $2 3072x128x10 55 50000 25 cifar100 adam relu
