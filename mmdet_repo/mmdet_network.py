import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.runner import BaseModule

from mmdet.models.builder import BACKBONES

from base.models.network import Network

def name_parse(name):
	cfg = None

	if name=='resnet50':
		from base.models.configs.resnet50.base import cfg
	if name=='resnet50_lip':
		from base.models.configs.resnet50.lip import cfg
	if name=='resnet50_gaussian_pool':
		from base.models.configs.resnet50.gaussian_pool import cfg
	if name=='resnet50_2222':
		from base.models.configs.resnet50_2222.base import cfg
	if name=='resnet50_4222':
		from base.models.configs.resnet50_4222.base import cfg


	if name=='dyresnet50':
		from base.models.configs.dyresnet50.base import cfg
	if name=='dyresnet50_lip':
		from base.models.configs.dyresnet50.lip import cfg
	if name=='dyresnet50_gaussian_pool':
		from base.models.configs.dyresnet50.gaussian_pool import cfg
	if name=='dyresnet50_2222':
		from base.models.configs.dyresnet50_2222.base import cfg
	if name=='dyresnet50_4222':
		from base.models.configs.dyresnet50_4222.base import cfg

	

	if name=='resnet50_nlp':
		from mmdet_repo.configs._backbone.resnet50.nlp import cfg
	if name=='dyresnet50_nlp':
		from mmdet_repo.configs._backbone.dyresnet50.nlp import cfg


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