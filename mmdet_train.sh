#!/usr/bin/env bash

CONFIG=$1

PYTHONPATH="$(dirname $0)/":"$(dirname $0)/mmdetection/":$PYTHONPATH \
python $(dirname $0)/mmdetection/tools/train.py $CONFIG