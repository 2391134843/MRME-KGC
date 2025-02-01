#!/bin/bash

cd model/
#!/bin/bash


export CUDA_VISIBLE_DEVICES=0


python3 learn.py \
    --dataset FB237 \
    --model MRME_KGC \
    --rank 100 \
    --learning_rate 0.005 \
    --optimizer Adagrad \
    --batch_size 2048 \
    --regularizer N3 \
    --reg 9e-2 \
    --max_epochs 110 \
    --valid 5 \
    -train \
    -id 0 \
    -save