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

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=20)

# The new config inherits a base config to highlight the necessary modification
_base_ = [
	root + '/mmdet_repo/configs/common/coco_detection.py',
	root + '/mmdet_repo/configs/common/runtime.py',
]

# import my net
custom_imports = dict(
	imports=['mmdet_repo.mmdet_network'],
	allow_failed_imports=False
	)

model = dict(
    type='SingleStageDetector',
    backbone=dict(
        type='MyBackBone',
		name='mobilenet-skip-1222121',
		pth_file=root+'/check_points/mobilenet_v2.pth',
        out_indices=(4, 7),
		use_fc_layer=False
    ),
    neck=dict(
        type='SSDNeck',
        in_channels=(96, 1280),
        out_channels=(96, 1280, 512, 256, 256, 128),
        level_strides=(2, 2, 2, 2),
        level_paddings=(1, 1, 1, 1),
        l2_norm_scale=None,
        use_depthwise=True,
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.03),
        act_cfg=dict(type='ReLU6'),
        init_cfg=dict(type='TruncNormal', layer='Conv2d', std=0.03)),
    bbox_head=dict(
        type='SSDHead',
        in_channels=(96, 1280, 512, 256, 256, 128),
        num_classes=num_classes,
        use_depthwise=True,
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.03),
        act_cfg=dict(type='ReLU6'),
        init_cfg=dict(type='Normal', layer='Conv2d', std=0.001),

        # set anchor size manually instead of using the predefined
        # SSD300 setting.
        anchor_generator=dict(
            type='SSDAnchorGenerator',
            scale_major=False,
            strides=[16, 32, 64, 107, 160, 320],
            ratios=[[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
            min_sizes=[48, 100, 150, 202, 253, 304],
            max_sizes=[100, 150, 202, 253, 304, 320]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2])),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.,
            ignore_iof_thr=-1,
            gt_max_assign_all=False),
        smoothl1_beta=1.,
        allowed_border=-1,
        pos_weight=-1,
        neg_pos_ratio=3,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        nms=dict(type='nms', iou_threshold=0.45),
        min_bbox_size=0,
        score_thr=0.02,
        max_per_img=200))
cudnn_benchmark = True

# dataset settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 4)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=(320, 320), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=320),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(320, 320),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=320),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
try:
    data = dict(
        samples_per_gpu=16,
        workers_per_gpu=4,
        train=dict(
            classes = classes,
            pipeline = train_pipeline,
            ),
        val=dict(
            classes = classes,
            pipeline=test_pipeline
            ),
        test=dict(
            classes = classes,
            pipeline=test_pipeline
            )
        )
except:
    data = dict(
        samples_per_gpu=16,
        workers_per_gpu=4,
        train=dict(pipeline=train_pipeline),
        val=dict(pipeline=test_pipeline),
        test=dict(pipeline=test_pipeline))

# optimizer
optimizer = dict(type='SGD', lr=0.015, momentum=0.9, weight_decay=4.0e-5)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    min_lr=0)

# Avoid evaluation and saving weights too frequently
evaluation = dict(interval=5, metric='bbox')
checkpoint_config = dict(interval=5)
custom_hooks = [
    dict(type='NumClassCheckHook'),
    dict(type='CheckInvalidLossHook', interval=50, priority='VERY_LOW')
]