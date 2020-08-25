#!/bin/bash


GPU=0

MODEL=cnn3d
BS=4
LR=0.0001
LOSS=bce
SIZE=160
WORK_DIR=model_${BS}_${MODEL}_${LOSS}_${SIZE}_pretrain_ft
ENCODER=resnet101
FOLD=0

CUDA_VISIBLE_DEVICES=${GPU} python ./src/train.py \
  --model ${MODEL} \
  --encoder ${ENCODER} \
  --fps 1 \
  --size ${SIZE} \
  --work-dir ${WORK_DIR} \
  --batch-size ${BS} \
  --lr ${LR} \
  --epochs 1120 \
  --csv train_metadata.csv \
  --lmdb micro_croped2.lmdb \
  --data ./data \
  --n-folds 10 \
  --fold ${FOLD} \
  --n-classes 1 \
  --loss ${LOSS} \
  --scheduler cos \
  --pretrain tier1_croped_tier1.lmdb \
  --resume ${WORK_DIR}/${ENCODER}_b${BS}_adam_lr${LR}_f${FOLD}_fps1_s${SIZE}/last.pth \
  --ft \


WORK_DIR=model_${BS}_${MODEL}_${LOSS}_${SIZE}_pretrain_ft2_t6
CUDA_VISIBLE_DEVICES=${GPU} python ./src/train.py \
  --model ${MODEL} \
  --encoder ${ENCODER} \
  --fps 1 \
  --size ${SIZE} \
  --work-dir ${WORK_DIR} \
  --batch-size ${BS} \
  --lr ${LR} \
  --epochs 1120 \
  --csv train_metadata.csv \
  --lmdb micro_croped2.lmdb \
  --data ./data \
  --n-folds 10 \
  --fold ${FOLD} \
  --n-classes 1 \
  --loss ${LOSS} \
  --scheduler cos \
  --pretrain tier1_croped_tier1.lmdb \
  --pretrain2 tier2_croped_tier2.lmdb \
  --load model_${BS}_${MODEL}_${LOSS}_${SIZE}_pretrain_ft/${ENCODER}_b${BS}_adam_lr${LR}_f${FOLD}_fps1_s${SIZE}/chkp_515.pth \
  --ft \
#  --resume ${WORK_DIR}/${ENCODER}_b${BS}_adam_lr${LR}_f${FOLD}_fps1_s${SIZE}/last.pth \

