from mmdet_repo.configs.common.coco_detection import root as coco_root
root = coco_root + '/Non-Local-Pooling/' # linux

_base_ = [root + '/mmdet_repo/configs/network/yolov3/resnet18/skip-2222.py']

model = dict(
    backbone=dict(
        name='resnet18-lip-2222',
        ),
    )