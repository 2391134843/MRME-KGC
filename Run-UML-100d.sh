#!/bin/bash


cd model/

# Specify the GPU device to use
export CUDA_VISIBLE_DEVICES=0

# Run the Python script and pass in the corresponding parameters
python3 learn.py \
    --dataset uml \
    --model MRME_KGC \
    --rank 100 \
    --optimizer Adagrad \
    --learning_rate 1e-1 \
    --batch_size 2048 \
    --regularizer N3 \
    --reg 5e-2 \
    --max_epochs 110 \
    --valid 500 \
    -train \
    -id 0 \
    -save
