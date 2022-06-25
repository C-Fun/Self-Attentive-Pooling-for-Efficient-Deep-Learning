import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.runner import BaseModule

from mmdet.models.builder import BACKBONES

from base.models.network import Network

def name_parse(name):
	cfg = None
	if name=='resnet18':
		from base.models.configs.resnet18.base import cfg
	if name=='resnet18_lip':
		from base.models.configs.resnet18.lip import cfg
	if name=='resnet18_gaussian_pool':
		from base.models.configs.resnet18.gaussian_pool import cfg

	if name=='resnet18_nlp':
		from base.models.configs.resnet18.nlp import cfg
	if name=='resnet18_nlp_2222':
		from base.models.configs.resnet18_2222.nlp import cfg
	if name=='resnet18_nlp_4222':
		from base.models.configs.resnet18_4222.nlp import cfg
	if name=='resnet18_dfmnlp':
		from base.models.configs.resnet18.dfmnlp import cfg
	if name=='resnet18_dfmnlp_2222':
		from base.models.configs.resnet18_2222.dfmnlp import cfg
	if name=='resnet18_dfmnlp_4222':
		from base.models.configs.resnet18_4222.dfmnlp import cfg

	if name=='resnet18_dfmnlp_reduced':
		from base.models.configs.resnet18.dfmnlp_reduced import cfg
	if name=='resnet18_dfmixp_reduced':
		from base.models.configs.resnet18.dfmixp_reduced import cfg
	if name=='resnet18_mixp_reduced':
		from base.models.configs.resnet18.mixp_reduced import cfg
	if name=='resnet18_nlp_reduced':
		from base.models.configs.resnet18.nlp_reduced import cfg


	if name=='dyresnet18':
		from base.models.configs.dyresnet18.base import cfg
	if name=='dyresnet18_lip':
		from base.models.configs.dyresnet18.lip import cfg
	if name=='dyresnet18_gaussian_pool':
		from base.models.configs.dyresnet18.gaussian_pool import cfg

	if name=='dyresnet18_nlp':
		from base.models.configs.dyresnet18.nlp import cfg
	if name=='dyresnet18_nlp_2222':
		from base.models.configs.dyresnet18_2222.nlp import cfg
	if name=='dyresnet18_nlp_4222':
		from base.models.configs.dyresnet18_4222.nlp import cfg
	if name=='dyresnet18_dfmnlp':
		from base.models.configs.dyresnet18.dfmnlp import cfg
	if name=='dyresnet18_dfmnlp_2222':
		from base.models.configs.dyresnet18_2222.dfmnlp import cfg
	if name=='dyresnet18_dfmnlp_4222':
		from base.models.configs.dyresnet18_4222.dfmnlp import cfg

	if name=='dyresnet18_dfmnlp_reduced':
		from base.models.configs.dyresnet18.dfmnlp_reduced import cfg
	if name=='dyresnet18_dfmixp_reduced':
		from base.models.configs.dyresnet18.dfmixp_reduced import cfg
	if name=='dyresnet18_mixp_reduced':
		from base.models.configs.dyresnet18.mixp_reduced import cfg
	if name=='dyresnet18_nlp_reduced':
		from base.models.configs.dyresnet18.nlp_reduced import cfg


	if name=='resnet50':
		from base.models.configs.resnet50.base import cfg
	if name=='resnet50_2222':
		from base.models.configs.resnet50_2222.base import cfg
	if name=='resnet50_4222':
		from base.models.configs.resnet50_4222.base import cfg
	if name=='resnet50_lip':
		from base.models.configs.resnet50.lip import cfg
	if name=='resnet50_lip_2222':
		from base.models.configs.resnet50_2222.lip import cfg
	if name=='resnet50_lip_4222':
		from base.models.configs.resnet50_4222.lip import cfg
	if name=='resnet50_gaussian_pool':
		from base.models.configs.resnet50.gaussian_pool import cfg
	if name=='resnet50_gaussian_pool_2222':
		from base.models.configs.resnet50_2222.gaussian_pool import cfg
	if name=='resnet50_gaussian_pool_4222':
		from base.models.configs.resnet50_4222.gaussian_pool import cfg

	if name=='resnet50_nlp':
		from base.models.configs.resnet50.nlp import cfg
	if name=='resnet50_nlp_2222':
		from base.models.configs.resnet50_2222.nlp import cfg
	if name=='resnet50_nlp_4222':
		from base.models.configs.resnet50_4222.nlp import cfg
	if name=='resnet50_dfmnlp':
		from base.models.configs.resnet50.dfmnlp import cfg
	if name=='resnet50_dfmnlp_2222':
		from base.models.configs.resnet50_2222.dfmnlp import cfg
	if name=='resnet50_dfmnlp_4222':
		from base.models.configs.resnet50_4222.dfmnlp import cfg


	if name=='dyresnet50':
		from base.models.configs.dyresnet50.base import cfg
	if name=='dyresnet50_2222':
		from base.models.configs.dyresnet50_2222.base import cfg
	if name=='dyresnet50_4222':
		from base.models.configs.dyresnet50_4222.base import cfg
	if name=='dyresnet50_lip':
		from base.models.configs.dyresnet50.lip import cfg
	if name=='dyresnet50_lip_2222':
		from base.models.configs.dyresnet50_2222.lip import cfg
	if name=='dyresnet50_lip_4222':
		from base.models.configs.dyresnet50_4222.lip import cfg
	if name=='dyresnet50_gaussian_pool':
		from base.models.configs.dyresnet50.gaussian_pool import cfg
	if name=='dyresnet50_gaussian_pool_2222':
		from base.models.configs.dyresnet50_2222.gaussian_pool import cfg
	if name=='dyresnet50_gaussian_pool_4222':
		from base.models.configs.dyresnet50_4222.gaussian_pool import cfg

	if name=='dyresnet50_nlp':
		from base.models.configs.dyresnet50.nlp import cfg
	if name=='dyresnet50_nlp_2222':
		from base.models.configs.dyresnet50_2222.nlp import cfg
	if name=='dyresnet50_nlp_4222':
		from base.models.configs.dyresnet50_4222.nlp import cfg
	if name=='dyresnet50_dfmnlp':
		from base.models.configs.dyresnet50.dfmnlp import cfg
	if name=='dyresnet50_dfmnlp_2222':
		from base.models.configs.dyresnet50_2222.dfmnlp import cfg
	if name=='dyresnet50_dfmnlp_4222':
		from base.models.configs.dyresnet50_4222.dfmnlp import cfg


	if name=='resnet50_nlp_reduced':
		from base.models.configs.resnet50.nlp_reduced import cfg
	if name=='dyresnet50_nlp_reduced':
		from base.models.configs.dyresnet50.nlp_reduced import cfg



	if cfg==None:
		raise Exception("Undefined Network Type!")

	return cfg


@BACKBONES.register_module()
class MyBackBone(Network, BaseModule):
	def __init__(self, name, pth_file=None, **kwargs):
		cfg = name_parse(name)
		super(MyBackBone, self).__init__(cfg, pth_file=pth_file, **kwargs)

	def forward(self, x):
		outs = self.net(x)
		# for i,out in enumerate(outs):
		# 	print(i, out.shape)
		# print()
		return outs