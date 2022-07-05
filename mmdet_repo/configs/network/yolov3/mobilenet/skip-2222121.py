from mmdet_repo.configs.common.coco_detection import root as coco_root
root = coco_root + '/Non-Local-Pooling/' # linux

_base_ = [root + '/mmdet_repo/configs/network/yolov3/mobilenet/skip-1222121.py']

model = dict(
    backbone=dict(
        name='mobilenet-skip-2222121',
        ),
    box_head=dict(
        anchor_generator=dict(
            strides=[64, 32, 16]),
        featmap_strides=[64, 32, 16],
        ),
    )