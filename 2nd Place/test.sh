#!/bin/bash

GPU=0

TTA=1
FOLD=0
ENCODER=resnet101

WORK_DIR=model_4_cnn3d_bce_160_pretrain_ft
for EPOCH in 515 705 915;
do
    CUDA_VISIBLE_DEVICES=${GPU} python ./src/predict.py \
        --data ./data \
        --load ./${WORK_DIR}/${ENCODER}_b4_adam_lr0.0001_f${FOLD}_fps1_s160/chkp_${EPOCH}.pth
done

WORK_DIR=model_4_cnn3d_bce_160_pretrain_ft2_t6
for EPOCH in 798 1115;
do
    CUDA_VISIBLE_DEVICES=${GPU} python ./src/predict.py \
        --data ./data \
        --load ./${WORK_DIR}/${ENCODER}_b4_adam_lr0.0001_f${FOLD}_fps1_s160/chkp_${EPOCH}.pth
done

CUDA_VISIBLE_DEVICES=${GPU} python ./src/submit.py \
    --data ./data \
    --csv test_metadata.csv \
    --lmdb test_croped2.lmdb \
    --exp ./ \
    --submit-path ./sub_ft_515_705_915_ft2_t6_798_1115_tta${TTA}.csv \
    --thresh 0.6 \
    --tta ${TTA} \
    --batch-size 8

