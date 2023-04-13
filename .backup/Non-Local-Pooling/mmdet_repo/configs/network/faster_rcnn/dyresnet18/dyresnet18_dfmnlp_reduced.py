from mmdet_repo.configs.common.coco_detection import root as coco_root
root = coco_root + '/Non-Local-Pooling/' # linux

_base_ = [root + '/mmdet_repo/configs/network/faster_rcnn/dyresnet18/dyresnet18.py']

model = dict(
	backbone=dict(
		name='dyresnet18-dfmnlp_reduced-1222',
		),
	)