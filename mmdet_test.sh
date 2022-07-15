#!/usr/bin/env bash
MMDET_ROOT='/nas/home/fangc/mmdetection/'

#CONFIG='mmdet_repo/configs/network/ssd/resnet18/ssd_resnet18-nlp-1222.py'
#CHECKPOINT='work_dirs/ssd-resnet18/ssd_resnet18-nlp-1222/epoch_20.pth'
#PKLDIR='work_dirs/ssd-resnet18/ssd_resnet18-nlp-1222/test.pkl'
#SHOW_DIR='work_dirs/ssd-resnet18/ssd_resnet18-nlp-1222/results/'

CONFIG='mmdet_repo/configs/network/ssd/resnet18/ssd_resnet18-lip-1222.py'
CHECKPOINT='work_dirs/ssd-resnet18/ssd_resnet18-lip-1222/epoch_20.pth'
PKLDIR='work_dirs/ssd-resnet18/ssd_resnet18-lip-1222/test.pkl'
SHOW_DIR='work_dirs/ssd-resnet18/ssd_resnet18-lip-1222/results/'

#CONFIG='mmdet_repo/configs/network/ssd/resnet18/ssd_resnet18-gaussian_pool-1222.py'
#CHECKPOINT='work_dirs/ssd-resnet18/ssd_resnet18-gaussian_pool-1222/epoch_20.pth'
#PKLDIR='work_dirs/ssd-resnet18/ssd_resnet18-gaussian_pool-1222/test.pkl'

#PYTHONPATH="$(dirname $0)/":"$MMDET_ROOT":$PYTHONPATH \
#python $MMDET_ROOT/tools/test.py $CONFIG $CHECKPOINT --eval bbox --out $PKLDIR

PYTHONPATH="$(dirname $0)/":"$MMDET_ROOT":$PYTHONPATH \
python $MMDET_ROOT/tools/analysis_tools/analyze_results.py $CONFIG $PKLDIR $SHOW_DIR --show