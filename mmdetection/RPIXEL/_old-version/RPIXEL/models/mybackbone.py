import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
from mmcv.runner import BaseModule

from mmdet.models.builder import BACKBONES

from RPIXEL.models.my_resnet import resnet50

@BACKBONES.register_module()
class MyBackBone(BaseModule):
	def __init__(self, pretrained=False, pth_file=None):
		super(MyBackBone, self).__init__()
		self.resnet = resnet50(pretrained=pretrained, pth_file=pth_file)
	def forward(self,x):
		outs = self.resnet(x)
		# for i,out in enumerate(outs):
		# 	print(i, out.shape)
		# print()
		return outs
