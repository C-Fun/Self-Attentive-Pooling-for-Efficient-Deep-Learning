from mmdet_repo.configs.common.coco_detection import root as coco_root
root = coco_root + '/Non-Local-Pooling/' # linux

_base_ = [root + '/mmdet_repo/configs/network/ssd/mobilenet/skip-1222121.py']

model = dict(
    backbone=dict(
        name='mobilenet-skip-2222121',
        ),
    box_head=dict(
        anchor_generator=dict(
            strides=[k*2 for k in [16, 32, 64, 107, 160, 320]]),
        ),
    )