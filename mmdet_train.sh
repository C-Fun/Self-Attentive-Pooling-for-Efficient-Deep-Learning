#!/usr/bin/env bash

CONFIG=$1

PYTHONPATH="$(dirname $0)/":$PYTHONPATH \
python ./mmdetection/tools/train.py $CONFIG