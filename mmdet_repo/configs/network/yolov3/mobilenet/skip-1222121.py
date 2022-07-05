from mmdet_repo.configs.common.coco_detection import root as coco_root
root = coco_root + '/Non-Local-Pooling/' # linux

_base_ = [root + '/mmdet_repo/configs/network/yolov3/_base_.py']

model = dict(
	backbone=dict(
		type='MyBackBone',
		name='mobilenet-skip-1222121',
		pth_file=None,
        out_indices=[2, 4, 6],
		use_fc_layer=False
	    ),
	neck=dict(
        num_scales=3,
        in_channels=[320, 96, 32],
        out_channels=[96, 96, 96]
        ),
    bbox_head=dict(
        in_channels=[96, 96, 96],
        out_channels=[96, 96, 96],
        anchor_generator=dict(
            strides=[32, 16, 8]),
        featmap_strides=[32, 16, 8],
        )
	)