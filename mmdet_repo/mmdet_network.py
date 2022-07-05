import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.runner import BaseModule

from mmdet.models.builder import BACKBONES
from base.models.network import Network
from base.models.network import name_parse as base_name_parse

def name_parse(name):
	cfg = base_name_parse(name)
	return cfg

@BACKBONES.register_module()
class MyBackBone(Network, BaseModule):
	def __init__(self, name, pth_file=None, out_indices=[0,1,2,3], **kwargs):
		cfg = name_parse(name)
		self.out_indices = out_indices
		super(MyBackBone, self).__init__(cfg, pth_file=pth_file, **kwargs)

	def forward(self, x):
		outs = self.net(x)
		final_outs = []
		for i in self.out_indices:
			final_outs.append(outs[i])
		# 	print(i, outs[i].shape)
		# print()
		return final_outs