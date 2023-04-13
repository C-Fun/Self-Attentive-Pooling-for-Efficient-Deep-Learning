# root = 'E:/留学相关/研究/' # windows
root = '/nas/home/fangc/' # linux


im_size = (1333, 800)
keep_ratio = True
dataset_type = 'CocoDataset'


# Coco settings
classes = ('person', 'bicycle', 'car', 'motorcycle',
		   'bus', 'train', 'truck', 'traffic light')
data_root = root + '/data/coco/'
train_img_path = data_root + '/train2017/'
train_ann_path = data_root + '/annotations/instances_train2017.json'
val_img_path = data_root + '/val2017/'
val_ann_path = data_root + '/annotations/instances_val2017.json'
test_img_path = data_root + '/val2017/'
test_ann_path = data_root + '/annotations/instances_val2017.json'

# # BDD 100K settings
# classes = ('pedestrian', 'rider', 'car', 'truck',
# 		   'bus', 'train', 'truck', 'motorcycle',
# 		   'bicycle', 'traffic light', 'traffic sign')
# data_root = root + '/data/bdd100k/'
# train_img_path = data_root + '/images/100k/train/'
# train_ann_path = data_root + '/annotations/det_train_coco.json'
# val_img_path = data_root + '/images/100k/val/'
# val_ann_path = data_root + '/annotations/det_val_coco.json'
# test_img_path = data_root + '/images/100k/val/'
# test_ann_path = data_root + '/annotations/det_val_coco.json'

# Modify dataset related settings
img_norm_cfg = dict(
	mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
	dict(type='LoadImageFromFile'),
	dict(type='LoadAnnotations', with_bbox=True),
	dict(type='Resize', img_scale=im_size, keep_ratio=keep_ratio),
	dict(type='RandomFlip', flip_ratio=0.5),
	dict(type='Normalize', **img_norm_cfg),
	dict(type='Pad', size_divisor=32),
	dict(type='DefaultFormatBundle'),
	dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
# val_pipeline = [
# 	dict(type='LoadImageFromFile'),
# 	dict(type='LoadAnnotations', with_bbox=True),
# 	dict(type='Resize', img_scale=im_size, keep_ratio=keep_ratio),
# 	dict(type='RandomFlip', flip_ratio=0.0),
# 	dict(type='Normalize', **img_norm_cfg),
# 	dict(type='Pad', size_divisor=32),
# 	dict(type='DefaultFormatBundle'),
# 	dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
# ]
test_pipeline = [
	dict(type='LoadImageFromFile'),
	dict(
		type='MultiScaleFlipAug',
		img_scale=im_size,
		flip=False,
		transforms=[
			dict(type='Resize', keep_ratio=keep_ratio),
			dict(type='RandomFlip'),
			dict(type='Normalize', **img_norm_cfg),
			dict(type='Pad', size_divisor=32),
			dict(type='ImageToTensor', keys=['img']),
			dict(type='Collect', keys=['img']),
		])
]

data = dict(
	samples_per_gpu=2,
    workers_per_gpu=2,
	train=dict(
		type=dataset_type,
		img_prefix=train_img_path,
		ann_file=train_ann_path,
		pipeline=train_pipeline
		),
	val=dict(
		type=dataset_type,
		img_prefix=val_img_path,
		ann_file=val_ann_path,
		pipeline=test_pipeline
		),
	test=dict(
		type=dataset_type,
		img_prefix=test_img_path,
		ann_file=test_ann_path,
		pipeline=test_pipeline
		)
	)
evaluation = dict(interval=1, metric='bbox')