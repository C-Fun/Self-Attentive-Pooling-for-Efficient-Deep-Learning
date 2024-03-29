from mmdet_repo.configs.common.coco_detection import root as coco_root
root = coco_root + '/Non-Local-Pooling/' # linux

_base_ = [root + '/mmdet_repo/configs/network/ssd/mobilenet/skip-1222121.py']

model = dict(
    backbone=dict(
        name='mobilenet_v2-lip-1222121',
        ),
    )