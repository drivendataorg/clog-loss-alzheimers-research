#!/bin/bash


GPU=0


TTA=1

FOLD=0
ENCODER=resnet101

CUDA_VISIBLE_DEVICES=${GPU} python ./src/submit.py \
    --data ./data \
    --csv test_metadata.csv \
    --lmdb test_croped2.lmdb \
    --exp ./chkps \
    --submit-path ./chkps/sub_ft_515_705_915_ft2_t6_798_1115_tta${TTA}.csv \
    --thresh 0.5 \
    --tta ${TTA} \
    --batch-size 8
