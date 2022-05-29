#!/usr/bin/env bash
MMDET_ROOT='/nas/home/fangc/mmdetection/'

CONFIG=$1

PYTHONPATH="$(dirname $0)/":"$MMDET_ROOT":$PYTHONPATH \
python $MMDET_ROOT/tools/train.py $CONFIG