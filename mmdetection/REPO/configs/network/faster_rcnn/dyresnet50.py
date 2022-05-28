from mmdetection.REPO.configs.common.coco_detection import root as coco_root
root = coco_root + '/Non-Local-Pooling/' # linux

_base_ = [root + '/mmdetection/REPO/configs/network/faster_rcnn/_base_.py']

model = dict(
	backbone=dict(
		name='dyresnet50',
		),
	)