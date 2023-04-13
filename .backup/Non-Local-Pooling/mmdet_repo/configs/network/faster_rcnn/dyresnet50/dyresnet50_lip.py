from mmdet_repo.configs.common.coco_detection import root as coco_root
root = coco_root + '/Non-Local-Pooling/' # linux

_base_ = [root + '/mmdet_repo/configs/network/faster_rcnn/dyresnet50/dyresnet50.py']

model = dict(
	backbone=dict(
		name='dyresnet50-lip-1222',
		)
	)

data = dict(
	    samples_per_gpu=1,
	    workers_per_gpu=1,
	    )