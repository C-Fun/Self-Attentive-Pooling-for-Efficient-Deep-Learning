from mmdet_repo.configs.common.coco_detection import root as coco_root
root = coco_root + '/Non-Local-Pooling/' # linux

try:
    from mmdet_repo.configs.common.coco_detection import classes
    num_classes = len(classes)
except:
    print('====================================================')
    print('** Failed to Setting Classes => Use Default Classes')
    print('====================================================')
    num_classes = 80

# The new config inherits a base config to highlight the necessary modification
_base_ = [
	root + '/mmdet_repo/configs/baseline/yolov3.py',
	root + '/mmdet_repo/configs/common/coco_detection.py',
	root + '/mmdet_repo/configs/common/runtime.py',
]


# import my net
custom_imports = dict(
	imports=['mmdet_repo.mmdet_network'],
	allow_failed_imports=False
	)

# model settings
model = dict(
    type='YOLOV3',
    backbone=dict(
		type='MyBackBone',
		name='resnet50-skip-1222',
		pth_file=root+'/check_points/resnet50.pth',
		use_fc_layer=False
		),
    neck=dict(
        type='YOLOV3Neck',
        num_scales=3,
        in_channels=[1024, 512, 256],
        out_channels=[512, 256, 128]),
    bbox_head=dict(
        type='YOLOV3Head',
        num_classes=num_classes,
        in_channels=[512, 256, 128],
        out_channels=[1024, 512, 256],
        anchor_generator=dict(
            type='YOLOAnchorGenerator',
            base_sizes=[[(116, 90), (156, 198), (373, 326)],
                        [(30, 61), (62, 45), (59, 119)],
                        [(10, 13), (16, 30), (33, 23)]],
            strides=[32, 16, 8]),
        bbox_coder=dict(type='YOLOBBoxCoder'),
        featmap_strides=[32, 16, 8],
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            reduction='sum'),
        loss_conf=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            reduction='sum'),
        loss_xy=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=2.0,
            reduction='sum'),
        loss_wh=dict(type='MSELoss', loss_weight=2.0, reduction='sum')),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='GridAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0)),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        conf_thr=0.005,
        nms=dict(type='nms', iou_threshold=0.45),
        max_per_img=100))

try:
	data = dict(
	    samples_per_gpu=2,
	    workers_per_gpu=2,
	    train=dict(
	        classes = classes
	        ),
	    val=dict(
	        classes = classes
	        ),
	    test=dict(
	        classes = classes
	        )
	    )
except:
	data = dict(
	    samples_per_gpu=2,
	    workers_per_gpu=2,
	    )
