from mmdet_repo.configs.common.coco_detection import root as coco_root
root = coco_root + '/Non-Local-Pooling/' # linux

_base_ = [root + '/mmdet_repo/configs/network/ssd/_base_.py']

runner = dict(type='EpochBasedRunner', max_epochs=20)

model = dict(
	backbone=dict(
		type='MyBackBone',
		name='resnet18-skip-1222',
		pth_file=root+'/check_points/resnet18.pth',
        out_indices=[2, 3],
		use_fc_layer=False
	    ),
	neck=dict(
        type='SSDNeck',
        in_channels=(256, 512),
        out_channels=(256, 512, 512, 256, 256, 128),
	),
    bbox_head=dict(
		in_channels=(256, 512, 512, 256, 256, 128),
        anchor_generator=dict(
            strides=[16, 32, 64, 107, 160, 320]),
        )
	)