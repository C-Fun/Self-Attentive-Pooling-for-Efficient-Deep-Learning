root = 'E:/留学相关/研究/' # windows
# root = '/nas/home/fangc/' # linux

# The new config inherits a base config to highlight the necessary modification
_base_ = root + '/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'

# import my net
custom_imports = dict(
    imports=['RPIXEL.models.mybackbone'],
    allow_failed_imports=False,
    )

# We also need to change the num_classes in head to match the dataset’s annotation
model = dict(
    backbone=dict(
        type='MyBackBone',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        # init_cfg=dict(
        #     type='Pretrained',
        #     checkpoint='/nas/home/mtian/mmdetection/pretrained_model/90_epoch_best_models/best_resnet50DP.pth')
        ),
    roi_head=dict(
        bbox_head=dict(num_classes=10),
        )
    )

# Modify dataset related settings
dataset_type = 'CocoDataset'
data_root = root + '/mmdetection/data/coco/'
# data_root = os.path.join(root, '/mmdetection/data/bdd100k/')

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
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
#load_from = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'