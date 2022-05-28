import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
from mmcv.runner import BaseModule

from mmdet.models.builder import BACKBONES

from RPIXEL.models.backbones.resnet import resnet18, resnet50
# from RPIXEL.models.utils.dynamic_convs.implementation1 import dynamic_convolution_generator
from RPIXEL.models.utils.dynamic_convs.implementation2 import Dynamic_conv2d
from RPIXEL.models.utils.pool_models import PosEncodeNLP, LIP2d, MixedPool

def cfg_parse(key):
	if key=='resnet18':
		from RPIXEL.configs.backbone._base import resnet18 as cfg
	if key=='resnet18_nlp':
		from RPIXEL.configs.backbone import resnet18_nlp as cfg
	if key=='resnet18_lip':
		from RPIXEL.configs.backbone import resnet18_lip as cfg

	if key=='dyresnet18':
		from RPIXEL.configs.backbone._base import dyresnet18 as cfg
	if key=='dyresnet18_nlp':
		from RPIXEL.configs.backbone import dyresnet18_nlp as cfg
	if key=='dyresnet18_lip':
		from RPIXEL.configs.backbone import dyresnet18_lip as cfg
	if key=='dyresnet18_mixp':
		from RPIXEL.configs.backbone import dyresnet18_mixp as cfg




	if key=='resnet50':
		from RPIXEL.configs.backbone._base import resnet50 as cfg
	if key=='resnet50_2222':
		from RPIXEL.configs.backbone.workshop_exp import resnet50_2222 as cfg
	if key=='resnet50_4222':
		from RPIXEL.configs.backbone.workshop_exp import resnet50_4222 as cfg

	if key=='dyresnet50':
		from RPIXEL.configs.backbone._base import dyresnet50 as cfg
	if key=='dyresnet50_2222':
		from RPIXEL.configs.backbone.workshop_exp import dyresnet50_2222 as cfg
	if key=='dyresnet50_4222':
		from RPIXEL.configs.backbone.workshop_exp import dyresnet50_4222 as cfg

	return cfg

@BACKBONES.register_module()
class MyBackBone(BaseModule):
	def __init__(self, key, pretrained=False, pth_file=None, **kwargs):
		super(MyBackBone, self).__init__()
		cfg = cfg_parse(key)

		# ====== conv1 args parse =========
		conv1_type = cfg.conv1[0]
		if conv1_type == 'normal':
			cfg.conv1[0] = nn.Conv2d
		elif conv1_type == 'dynamic':
			# cfg.conv1[0] = dynamic_convolution_generator(4, 4)
			cfg.conv1[0] = Dynamic_conv2d
		else:
			raise Exception("Undefined Conv1 Type!")

		# ====== conv2d args parse =========
		conv2d_type = cfg._convtype
		if conv2d_type == 'normal':
			cfg.conv2d = nn.Conv2d
		elif conv2d_type == 'dynamic':
			# cfg.conv2d = dynamic_convolution_generator(4, 4)
			cfg.conv2d = Dynamic_conv2d
		else:
			raise Exception("Undefined Conv2d Type!")

		# ====== pool1 args parse =========
		try:
			print(cfg.pool1)
		except:
			cfg.pool1 = None

		if cfg.pool1 != None:
			pool1_type = cfg.pool1[0]
			if pool1_type == 'maxpool':
				cfg.pool1[0] = nn.MaxPool2d
			elif pool1_type == 'penlp':
				cfg.pool1[0] = PosEncodeNLP
			elif pool1_type == 'lip':
				cfg.pool1[0] = LIP2d
			elif pool1_type == 'mixp':
				cfg.pool1[0] = MixedPool
			else:
				raise Exception("Undefined Pool1 Type!")

		# ====== pool params args parse =========
		add_keys = ['ksize', 'psize', 'dim_reduced_ratio', 'num_heads']
		if cfg._poolparams != None:
			for i in range(len(cfg._poolparams)):
				pool_param = cfg._poolparams[i]
				if 'type' not in pool_param.keys():
					raise Exception("Need to define the pool type!")
				if 'stride' not in pool_param.keys():
					raise Exception("Need to define the stride!")

				pool_type = pool_param['type']
				if pool_type == 'none':
					pool2d = None
				elif pool_type == 'penlp':
					pool2d = PosEncodeNLP
				elif pool_type == 'lip':
					pool2d = LIP2d
				elif pool_type == 'mixp':
					pool2d = MixedPool
				else:
					raise Exception("Undefined Pool2d Type!")

				cfg._poolparams[i]['pool2d'] = pool2d
				for key in add_keys:
					if key not in pool_param.keys():
						cfg._poolparams[i][key] = None
		if cfg._backbone == 'resnet50':
			self.resnet = resnet50(cfg, pretrained=pretrained, pth_file=pth_file, **kwargs)
		elif cfg._backbone == 'resnet18':
			self.resnet = resnet18(cfg, pretrained=pretrained, pth_file=pth_file, **kwargs)
		else:
			raise Exception("Undefined BackBone!")
	def forward(self,x):
		outs = self.resnet(x)
		# for i,out in enumerate(outs):
		# 	print(i, out.shape)
		# print()
		return outs