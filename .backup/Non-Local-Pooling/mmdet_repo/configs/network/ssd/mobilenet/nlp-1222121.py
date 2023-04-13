from mmdet_repo.configs.common.coco_detection import root as coco_root
root = coco_root + '/Non-Local-Pooling/' # linux

_base_ = [root + '/mmdet_repo/configs/network/ssd/mobilenet/skip-1222121.py']

model = dict(
    backbone=dict(
        name='mobilenet_v2-nlp_headfix2_nowin_reduced-1222121',
        ),
    )

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
)