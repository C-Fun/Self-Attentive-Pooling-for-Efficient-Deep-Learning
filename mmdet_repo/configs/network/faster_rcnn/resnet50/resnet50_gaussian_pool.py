from mmdet_repo.configs.common.coco_detection import root as coco_root
root = coco_root + '/Non-Local-Pooling/' # linux

_base_ = [root + '/mmdet_repo/configs/network/faster_rcnn/resnet50/resnet50.py']

model = dict(
	backbone=dict(
		name='resnet50_gaussian_pool',
		)
	)