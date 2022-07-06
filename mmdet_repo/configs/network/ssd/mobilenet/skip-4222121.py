from mmdet_repo.configs.common.coco_detection import root as coco_root
root = coco_root + '/Non-Local-Pooling/' # linux

_base_ = [root + '/mmdet_repo/configs/network/yolov3/mobilenet/skip-1222121.py']

model = dict(
    backbone=dict(
        name='mobilenet-skip-4222121',
        ),
    bbox_head=dict(
        anchor_generator=dict(
            strides=[k*4 for k in [16, 32, 64, 107, 160, 320]]),
        ),
    )