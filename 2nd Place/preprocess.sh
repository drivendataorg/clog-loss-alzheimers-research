#!/bin/bash


DATA_ROOT=/home/data/clog-loss
SAVE_DIR=./data

mkdir $SAVE_DIR
cp $DATA_ROOT/*.csv $SAVE_DIR

python ./src/preprocess.py micro --data $DATA_ROOT --save $SAVE_DIR
python ./src/preprocess.py test --data $DATA_ROOT --save $SAVE_DIR

# 200GB RAM
python ./src/preprocess_dl.py tier1 --data $DATA_ROOT --save $SAVE_DIR

python ./src/preprocess_dl2.py tier2 --data $DATA_ROOT --save $SAVE_DIR

