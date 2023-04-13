MMDET_ROOT='/nas/home/fangc/mmdetection/'
ROOT='/nas/home/fangc/Non-Local-Pooling/'

METHOD=lip
CFG_DIR=$ROOT/mmdet_repo/configs/network/ssd/resnet18/ssd_resnet18-$METHOD-1222.py
WORK_DIR=$ROOT/work_dirs/ssd-resnet18/ssd_resnet18-$METHOD-1222/

#000000000885_0.272.jpg, 000000004134_0.276.jpg, 000000005193_0.534.jpg, 000000014473_0.609.jpg, 000000017178_0.9.jpg, 000000017207_0.324.jpg, 000000023272_0.7.jpg, 000000025181_0.458.jpg

PYTHONPATH="$(dirname $0)/":"$MMDET_ROOT":"$ROOT":$PYTHONPATH \
python visualize.py $CFG_DIR \
                    $WORK_DIR/epoch_20.pth \
                    --img_list $WORK_DIR/img_list.txt \
                    --show-dir $WORK_DIR/visualize_results/