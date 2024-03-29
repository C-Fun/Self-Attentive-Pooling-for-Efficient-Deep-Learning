from mmdet_repo.configs.common.coco_detection import root as coco_root
root = coco_root + '/Non-Local-Pooling/' # linux

_base_ = [root + '/mmdet_repo/configs/network/faster_rcnn/resnet50/resnet50.py']

model = dict(
	backbone=dict(
		name='resnet50-lip-2222',
		),
	rpn_head=dict(
		anchor_generator=dict(
			strides=[8, 16, 32, 64, 128]),
		),
	roi_head=dict(
		bbox_roi_extractor=dict(
			featmap_strides=[8, 16, 32, 64]
			),
		),
	)