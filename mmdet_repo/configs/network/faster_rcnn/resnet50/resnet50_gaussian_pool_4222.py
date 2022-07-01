from mmdet_repo.configs.common.coco_detection import root as coco_root
root = coco_root + '/Non-Local-Pooling/' # linux

_base_ = [root + '/mmdet_repo/configs/network/faster_rcnn/resnet50/resnet50.py']

model = dict(
	backbone=dict(
		name='resnet50_v2-gaussian_pool-4222',
		),
	rpn_head=dict(
		anchor_generator=dict(
			strides=[16, 32, 64, 128, 256]),
		),
	roi_head=dict(
		bbox_roi_extractor=dict(
			featmap_strides=[16, 32, 64, 128]
			),
		),
	)