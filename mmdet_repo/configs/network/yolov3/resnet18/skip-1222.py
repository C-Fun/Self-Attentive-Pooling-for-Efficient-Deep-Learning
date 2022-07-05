from mmdet_repo.configs.common.coco_detection import root as coco_root
root = coco_root + '/Non-Local-Pooling/' # linux

_base_ = [root + '/mmdet_repo/configs/network/yolov3/_base_.py']

model = dict(
	backbone=dict(
		type='MyBackBone',
		name='resnet18-skip-1222',
		pth_file=root+'/check_points/resnet18.pth',
        out_indices=[1, 2, 3],
		use_fc_layer=False
	    ),
	neck=dict(
        num_scales=3,
        in_channels=[512, 256, 128],
        out_channels=[256, 128, 64]
        ),
    bbox_head=dict(
        in_channels=[256, 128, 64],
        out_channels=[512, 256, 128],
        anchor_generator=dict(
            strides=[32, 16, 8]),
        featmap_strides=[32, 16, 8],
        )
	)