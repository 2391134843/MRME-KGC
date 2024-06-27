#!/bin/bash


cd model/

# Specify the GPU device to use
export CUDA_VISIBLE_DEVICES=8

# Run the Python script and pass in the corresponding parameters
python3 learn.py \
    --dataset WN18RR \
    --model MRME_KGC \
    --rank 100 \
    --learning_rate 0.15 \
    --optimizer Adagrad \
    --batch_size 3000 \
    --regularizer N3 \
    --reg 1.5e-1 \
    --max_epochs 500 \
    --valid 5 \
    -train \
    -id 0 \
    -save
