from mmdet_repo.configs.common.coco_detection import root as coco_root
root = coco_root + '/Non-Local-Pooling/' # linux

_base_ = [root + '/mmdet_repo/configs/network/yolov3/mobilenet/skip-1222121.py']

model = dict(
    backbone=dict(
        name='mobilenet_v2-gaussian_pool-1222121',
        ),
    )