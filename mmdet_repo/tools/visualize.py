# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import time
import warnings

import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
						 wrap_fp16_model)
from mmcv.image import tensor2imgs

from mmdet.core import encode_mask_results
from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset, get_loading_pipeline, 
							replace_ImageToTensor)
from mmdet.models import build_detector


def visualize(model, data_loader, classes, img_list, show_dir, gpu_ids=[0], visual_type=['directly']):
	with open(img_list, 'r') as f:
		img_list = f.read()

	img_list = list(map(lambda x: x.split('_')[0].strip(), img_list.split(',')))

	print(img_list)

	model = MMDataParallel(model, gpu_ids)
	model.eval()
	num_classes = len(classes)

	def minmax(x):
		return (x-np.min(x))/(1e-10+np.max(x)-np.min(x))

	activation = {}
	def get_activation(name):
		def hook(model, input, output):
			activation[name] = output
		return hook

	name_list = []
	module_list = []
	for (name, module) in model.named_modules():
		if name.endswith('pool_weight'):
			module.register_forward_hook(get_activation(name))
			name_list.append(name)
			module_list.append(module)


	for i, data_dict in enumerate(tqdm(data_loader)):
		# print(data_dict)
		img_metas = data_dict['img_metas'][0].data[0]
		img_name = img_metas[0]['filename'].split('/')[-1].split('.')[0]
		# gt_bboxes = data_dict['gt_bboxes'].data[0][0]
		# gt_labels = data_dict['gt_labels'].data[0][0]
		# print('info:', img_name, gt_labels, gt_bboxes)

		if img_name not in img_list:
			continue

		forward_dict = {'img': [data_dict['img'][0].data], 'img_metas':[data_dict['img_metas'][0].data[0]]}
		with torch.no_grad():
			result = model(return_loss=False, rescale=True, **forward_dict)

		img = data_dict['img'][0].data[0,:,:,:].detach().cpu().numpy().transpose([1,2,0])
		img = np.uint8(255 * minmax(img))
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		(h,w,c) = img.shape


		# Directly
		if 'directly' in visual_type:
			direct_res = np.zeros((h, w*(len(name_list)+1), c))
			direct_res[:,:w,:]=img
			for (i, name) in enumerate(name_list):
				weight = activation[name]
				restore_weight = F.interpolate(weight, size=(h,w), mode='bilinear')
				avg_weight = torch.mean(restore_weight, axis=1)
				heatmap = avg_weight.detach().cpu().numpy().transpose([1,2,0])
				heatmap = np.uint8(255 * minmax(heatmap))
				heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
				heatimg = np.uint8(255 * minmax(heatmap*0.9+img))

				direct_res[:,(i+1)*w:(i+2)*w] = heatimg
			print(os.path.join(show_dir, img_name+'_directly.jpg'))
			cv2.imwrite(os.path.join(show_dir, img_name+'_directly.jpg'), direct_res)



def parse_args():
	parser = argparse.ArgumentParser(
		description='MMDet test (and eval) a model')
	parser.add_argument('config', help='test config file path')
	parser.add_argument('checkpoint', help='checkpoint file')
	parser.add_argument(
		'--work-dir',
		help='the directory to save the file containing evaluation metrics')
	parser.add_argument('--out', help='output result file in pickle format')
	parser.add_argument('--gpu_ids', type=str, default='0', help='output result file in pickle format')
	parser.add_argument(
		'--fuse-conv-bn',
		action='store_true',
		help='Whether to fuse conv and bn, this will slightly increase'
		'the inference speed')
	parser.add_argument(
		'--format-only',
		action='store_true',
		help='Format the output results without perform evaluation. It is'
		'useful when you want to format the result to a specific format and '
		'submit it to the test server')
	parser.add_argument(
		'--eval',
		type=str,
		nargs='+',
		help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
		' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
	parser.add_argument('--show', action='store_true', help='show results')
	parser.add_argument(
		'--show-dir', help='directory where painted images will be saved')
	parser.add_argument(
		'--show-score-thr',
		type=float,
		default=0.3,
		help='score threshold (default: 0.3)')
	parser.add_argument(
		'--gpu-collect',
		action='store_true',
		help='whether to use gpu to collect results.')
	parser.add_argument(
		'--tmpdir',
		help='tmp directory used for collecting results from multiple '
		'workers, available when gpu-collect is not specified')
	parser.add_argument(
		'--cfg-options',
		nargs='+',
		action=DictAction,
		help='override some settings in the used config, the key-value pair '
		'in xxx=yyy format will be merged into config file. If the value to '
		'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
		'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
		'Note that the quotation marks are necessary and that no white space '
		'is allowed.')
	parser.add_argument(
		'--options',
		nargs='+',
		action=DictAction,
		help='custom options for evaluation, the key-value pair in xxx=yyy '
		'format will be kwargs for dataset.evaluate() function (deprecate), '
		'change to --eval-options instead.')
	parser.add_argument(
		'--eval-options',
		nargs='+',
		action=DictAction,
		help='custom options for evaluation, the key-value pair in xxx=yyy '
		'format will be kwargs for dataset.evaluate() function')
	parser.add_argument(
		'--launcher',
		choices=['none', 'pytorch', 'slurm', 'mpi'],
		default='none',
		help='job launcher')
	parser.add_argument('--local_rank', type=int, default=0)

	parser.add_argument(
		'--img_list',
		type=str)

	args = parser.parse_args()
	if 'LOCAL_RANK' not in os.environ:
		os.environ['LOCAL_RANK'] = str(args.local_rank)

	if args.options and args.eval_options:
		raise ValueError(
			'--options and --eval-options cannot be both '
			'specified, --options is deprecated in favor of --eval-options')
	if args.options:
		warnings.warn('--options is deprecated in favor of --eval-options')
		args.eval_options = args.options

	return args


def main():
	args = parse_args()

	assert args.out or args.eval or args.format_only or args.show \
		or args.show_dir, \
		('Please specify at least one operation (save/eval/format/show the '
		 'results / save the results) with the argument "--out", "--eval"'
		 ', "--format-only", "--show" or "--show-dir"')

	if args.eval and args.format_only:
		raise ValueError('--eval and --format_only cannot be both specified')

	if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
		raise ValueError('The output file must be a pkl file.')

	cfg = Config.fromfile(args.config)

	# init distributed env first, since logger depends on the dist info.
	if args.launcher == 'none':
		distributed = False
	else:
		distributed = True
		init_dist(args.launcher, **cfg.dist_params)

	if not os.path.isdir(args.show_dir):
		os.mkdir(args.show_dir)

	# build the dataloader
	dataset = build_dataset(cfg.data.val)
	data_loader = build_dataloader(
		dataset,
		samples_per_gpu=1,
		workers_per_gpu=1,
		dist=distributed,
		shuffle=False)

	# print(cfg.data.val, dataset, data_loader)

	# print(help(build_dataset))
	# print(dataset, len(data_loader), data_loader.batch_size)

	# build the model and load checkpoint
	model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
	checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

	# old versions did not save class info in checkpoints, this walkaround is
	# for backward compatibility
	if 'CLASSES' in checkpoint.get('meta', {}):
		model.CLASSES = checkpoint['meta']['CLASSES']
	else:
		model.CLASSES = dataset.CLASSES

	gpu_ids = list(map(int, args.gpu_ids.split(',')))
	visualize(model, data_loader, dataset.CLASSES, args.img_list, args.show_dir, gpu_ids)


if __name__ == '__main__':
	main()
