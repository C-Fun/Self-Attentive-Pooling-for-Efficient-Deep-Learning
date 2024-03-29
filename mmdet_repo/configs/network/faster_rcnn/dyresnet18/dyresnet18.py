from mmdet_repo.configs.common.coco_detection import root as coco_root
root = coco_root + '/Non-Local-Pooling/' # linux

_base_ = [root + '/mmdet_repo/configs/network/faster_rcnn/_base_.py']

model = dict(
	backbone=dict(
		name='dyresnet18-skip-1222',
		pth_file=root+'/check_points/resnet18.pth',
		),
	neck=dict(
		in_channels=[64, 128, 256, 512], # resnet18
		)
	)