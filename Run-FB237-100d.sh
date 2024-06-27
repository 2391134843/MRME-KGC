#!/bin/bash

cd model/
#!/bin/bash

# 指定使用的GPU设备
export CUDA_VISIBLE_DEVICES=0

# 运行Python脚本，并传入相应的参数
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