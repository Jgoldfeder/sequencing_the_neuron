#!/bin/bash

# device, seed

CUDA_VISIBLE_DEVICES=$1 python reconstruct.py $2 784x128x10 55 60000 25 mnist adam relu

CUDA_VISIBLE_DEVICES=$1 python reconstruct.py $2 784x128x10 55 60000 25 mnist adam tanh


CUDA_VISIBLE_DEVICES=$1 python reconstruct.py $2 784x128x64x10 55 60000 25 mnist adam relu

CUDA_VISIBLE_DEVICES=$1 python reconstruct.py $2 784x128x64x10 55 60000 25 mnist adam tanh


CUDA_VISIBLE_DEVICES=$1 python reconstruct.py $2 784x128x64x32x10 55 60000 25 mnist adam relu

CUDA_VISIBLE_DEVICES=$1 python reconstruct.py $2 784x128x64x32x10 55 60000 25 mnist adam tanh


CUDA_VISIBLE_DEVICES=$1 python reconstruct.py $2 784x128x64x32x16x10 55 60000 25 mnist adam relu

CUDA_VISIBLE_DEVICES=$1 python reconstruct.py $2 784x128x64x32x16x10 55 60000 25 mnist adam tanh


CUDA_VISIBLE_DEVICES=$1 python reconstruct.py $2  784x128x80x40x32x16x10 55 60000 25 mnist adam relu

CUDA_VISIBLE_DEVICES=$1 python reconstruct.py $2 784x128x80x40x32x16x10 55 60000 25 mnist adam tanh

