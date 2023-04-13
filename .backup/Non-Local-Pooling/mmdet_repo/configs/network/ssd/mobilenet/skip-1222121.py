from mmdet_repo.configs.common.coco_detection import root as coco_root
root = coco_root + '/Non-Local-Pooling/' # linux

_base_ = [root + '/mmdet_repo/configs/network/ssd/_base_.py']

model = dict(
	backbone=dict(
		type='MyBackBone',
		name='mobilenet-skip-1222121',
		pth_file=None,
        out_indices=[4, 7],
		use_fc_layer=False
	    ),
    bbox_head=dict(
        anchor_generator=dict(
            strides=[16, 32, 64, 107, 160, 320]),
        )
	)