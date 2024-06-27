#!/bin/bash


cd model/

# Specify the GPU device to use
export CUDA_VISIBLE_DEVICES=6

# Run the Python script and pass in the corresponding parameters
python3 learn.py \
    --dataset YAGO3-10 \
    --model MRME_KGC \
    --rank 50 \
    --learning_rate 0.01 \
    --optimizer Adagrad \
    --batch_size 2048 \
    --regularizer N3 \
    --reg 5e-2 \
    --max_epochs 150 \
    --valid 5 \
    -train \
    -id 0 \
    -save
