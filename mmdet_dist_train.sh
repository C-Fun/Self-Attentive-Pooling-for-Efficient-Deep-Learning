#!/usr/bin/env bash
MMDET_ROOT='/nas/home/fangc/mmdetection/'

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/":"$MMDET_ROOT"$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $MMDET_ROOT/tools/train.py $CONFIG --launcher pytorch ${@:3}