from mmdet_repo.configs.common.coco_detection import root as coco_root
root = coco_root + '/Non-Local-Pooling/' # linux

_base_ = [root + '/mmdet_repo/configs/network/yolov3/resnet18/skip-1222.py']

model = dict(
    backbone=dict(
        name='resnet18-skip-4222',
        ),
    box_head=dict(
        anchor_generator=dict(
            strides=[128, 64, 32]),
        featmap_strides=[128, 64, 32],
        ),
    )