root = 'E:/留学相关/研究/' # windows
# root = '/nas/home/fangc/' # linux
im_size = (224, 224)
dataset_type = 'CocoDataset'
data_root = root + '/data/coco/'
# data_root = os.path.join(root, '/mmdetection/data/bdd100k/')

# The new config inherits a base config to highlight the necessary modification
_base_ = [
    root + '/mmdetection/configs/_base_/models/faster_rcnn_r50_fpn.py',
    root + '/mmdetection/configs/_base_/datasets/coco_detection.py',
    root + '/mmdetection/configs/_base_/schedules/schedule_1x.py', 
    root + '/mmdetection/configs/_base_/default_runtime.py',
]

# import my net
custom_imports = dict(
    imports=['RPIXEL.models.deform_patch'],
    allow_failed_imports=False
    )

# model settings
model = dict(
    type='FasterRCNN',
    backbone=dict(
        type='DeformPatchNet',
        imsize = im_size,
        in_channels = 3,
        mid_channels = 256,
        out_channels = 256,
        feature_num = 512,
        head_conv_blocknum = 0,
        head_offset_blocknum = 1,
        head_offset_size = 3,
        patch_size = 32,
        patch_stride = 32,
        resnet_depth = 50,
        resnet_stagesnum = 4),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=10,  # original: 80
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ))
# Modify dataset related settings

# train_pipeline = [
#     dict(type=‘LoadImageFromFile’),
#     dict(type=‘LoadAnnotations’, with_bbox=True),
#     dict(
#         type=‘Resize’,
#         img_scale = (1280, 720),
#         keep_ratio=False),
#     dict(type=‘RandomFlip’, flip_ratio=0.5),
#     dict(
#         type=‘Normalize’,
#         mean=[103.53, 116.28, 123.675],
#         std=[1.0, 1.0, 1.0],
#         to_rgb=False),
#     #dict(type=‘Pad’, size_divisor=32),
#     dict(type=‘DefaultFormatBundle’),
#     dict(type=‘Collect’, keys=[‘img’, ‘gt_bboxes’, ‘gt_labels’])
# ]
# test_pipeline = [
#     dict(type=‘LoadImageFromFile’),
#     dict(
#         type=‘MultiScaleFlipAug’,
#         img_scale=(1333, 800),
#         flip=False,
#         transforms=[
#             dict(type=‘Resize’, keep_ratio=True),
#             dict(type=‘RandomFlip’),
#             dict(
#                 type=‘Normalize’,
#                 mean=[103.53, 116.28, 123.675],
#                 std=[1.0, 1.0, 1.0],
#                 to_rgb=False),
#             dict(type=‘Pad’, size_divisor=32),
#             dict(type=‘ImageToTensor’, keys=[‘img’]),
#             dict(type=‘Collect’, keys=[‘img’])
#         ])
# ]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=im_size, keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=im_size,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
classes = ("person", "rider", "car", "bus", "truck", "bike",
               "motor", "traffic light", "traffic sign", "train")
#classes = ('pedestrian', 'car', 'rider', 'truck', 'bus', 'train', 'motorcycle','bicycle','traffic light','traffic sign')
runner = dict(type='EpochBasedRunner', max_epochs=50)
data = dict(
    train=dict(
        img_prefix=data_root + '/train2017/',
        classes=classes,
        ann_file=data_root + '/annotations/instances_train2017.json',
        #pipeline=train_pipeline
        ),
    val=dict(
        img_prefix=data_root + '/val2017/',
        classes=classes,
        ann_file=data_root + '/annotations/instances_val2017.json',
        #pipeline=test_pipeline
        ),
    test=dict(
        img_prefix=data_root + '/val2017/',
        classes=classes,
        ann_file=data_root + '/annotations/instances_val2017.json',
        #pipeline=test_pipeline
        )
    )
# optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
# optimizer_config = dict(_delete_=True,
#                         grad_clip=dict(max_norm=35, norm_type=2))
# optimizer = dict(
#     _delete_=True,
#     type='Adam',
#     lr=0.0025)


# We can use the pre-trained Mask RCNN model to obtain higher performance
#load_from = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'


# log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])