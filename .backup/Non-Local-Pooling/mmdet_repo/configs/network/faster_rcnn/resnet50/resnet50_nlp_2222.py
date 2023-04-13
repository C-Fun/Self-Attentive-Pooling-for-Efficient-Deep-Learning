from mmdet_repo.configs.common.coco_detection import root as coco_root
root = coco_root + '/Non-Local-Pooling/' # linux

_base_ = [root + '/mmdet_repo/configs/network/faster_rcnn/resnet50/resnet50.py']

model = dict(
	backbone=dict(
		name='resnet50-nlp-2222',
		)
	)

data = dict(
	    samples_per_gpu=1,
	    workers_per_gpu=1,
	    )