#!/bin/bash


cd model/

# Specify the GPU device to use
export CUDA_VISIBLE_DEVICES=0

# Run the Python script and pass in the corresponding parameters
python3 learn.py \
    --dataset kinships \
    --model MRME-KGC \
    --rank 100 \
    --learning_rate 0.2 \
    --optimizer Adagrad \
    --batch_size 2048 \
    --regularizer N3 \
    --reg 5e-2 \
    --max_epochs 110 \
    --valid 5 \
    -train \
    -id 0 \
    -save
