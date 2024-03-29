from mmdet_repo.configs.common.coco_detection import root as coco_root
root = coco_root + '/Non-Local-Pooling/' # linux

_base_ = [root + '/mmdet_repo/configs/network/faster_rcnn/_base_.py']

model = dict(
	backbone=dict(
		name='dyresnet50-skip-1222',
		),
	)