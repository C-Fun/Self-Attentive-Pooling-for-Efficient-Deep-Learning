import warnings

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
from mmcv.runner import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.models.builder import BACKBONES
from mmdet.models.utils import ResLayer
from mmdet.models import ResNet

from RPIXEL.models.deform_conv import DeformConv2D

class Deconv(BaseModule):
	def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
		super(Deconv, self).__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.deconv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=1, output_padding=0)
		self.bn = nn.BatchNorm2d(num_features=out_channels)

	def forward(self,x):
		return F.relu(self.bn(self.deconv(x)))

class ResBlock(BaseModule):
	def __init__(self, in_channels, out_channels):
		super(ResBlock, self).__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels

		self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
		self.bn1 = nn.BatchNorm2d(num_features=out_channels)

		self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
		self.bn2 = nn.BatchNorm2d(num_features=out_channels)

		self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
		self.bn3 = nn.BatchNorm2d(num_features=out_channels)

		if in_channels == out_channels:
			pass
		else:
			self.identity = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
			self.bn = nn.BatchNorm2d(num_features=out_channels)

	def forward(self,x):
		res = F.relu(self.bn1(self.conv1(x)))
		res = F.relu(self.bn2(self.conv2(res)))
		res = self.bn3(self.conv3(res))
		if self.in_channels == self.out_channels:
			y = res + x
		else:
			y = res + self.bn(self.identity(x))
		y = F.relu(y)
		return y

class DeformHead(BaseModule):
	def __init__(self, 
				 in_channels = 3,
				 mid_channels = 128,
				 out_channels = 64,
				 conv_blocknum = 0,
				 offset_blocknum = 1,
				 offset_size = 3,
				 feature_num = 16):
		super(DeformHead, self).__init__()

		# conv layers
		self.convs = nn.Sequential()
		self.convs.add_module("x_in", nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0))
		for i in range(conv_blocknum):
			self.convs.add_module("x_block_"+str(i+1), ResBlock(mid_channels, mid_channels))
		self.convs.add_module("x_out", nn.Conv2d(mid_channels, feature_num, kernel_size=1, padding=0))
		self.convs_bn = nn.BatchNorm2d(feature_num)

		# offset layers
		self.offsets = nn.Sequential()
		self.offsets.add_module("offset_in", nn.Conv2d(feature_num, mid_channels, kernel_size=1, padding=0))
		for i in range(offset_blocknum):
			self.offsets.add_module("offset_block_"+str(i+1), ResBlock(mid_channels, mid_channels))
		self.offsets.add_module("offset_out", nn.Conv2d(mid_channels, 2*offset_size**2, kernel_size=1, padding=0))

		# self.offsets = nn.Conv2d(feature_num, 2*offset_size**2, kernel_size=1, padding=0)

		# deformable conv
		self.deform_conv = DeformConv2D(feature_num, out_channels, kernel_size=offset_size, padding=1)
		self.deform_bn = nn.BatchNorm2d(out_channels)

	def forward(self,x):
		# convs
		x = F.relu(self.convs(x))
		x = self.convs_bn(x)
		# deformable convolution
		offsets = self.offsets(x)
		x, x_offset = self.deform_conv(x, offsets)
		x = F.relu(x)
		x = self.deform_bn(x)
		return x

class PatchNet(BaseModule):
	def __init__(self,
				 imsize,
				 in_channels = 3,
				 out_channels = 128,
				 patch_size = 16,
				 patch_stride = 16,
				 feature_num = 128,
				 depth = 50,
				 num_stages = 4):
		super(PatchNet, self).__init__()
		self.patch_size = patch_size
		patch_num = (1 + (imsize[0]-patch_size)//patch_stride) * (1 + (imsize[1]-patch_size)//patch_stride)
		
		self.patch_splitting = nn.Conv2d(in_channels,patch_size**2,kernel_size=patch_size,stride=patch_stride,padding=0)
		self.patch_net = ResNet(depth=depth,
						        num_stages=num_stages,
						        out_indices=(num_stages-1,))
		self.restore = Deconv(patch_size**2, feature_num, kernel_size=patch_size, stride=patch_stride, padding=0)
		self.conv = nn.Conv2d(feature_num,out_channels,kernel_size=1,stride=1,padding=0)
		self.bn = nn.BatchNorm2d(num_features=out_channels)

		self.patch_num = patch_num

	def forward(self,x):
		# x [b,c,h,w]
		# print('x_shape:', x.shape)
		patch = self.patch_splitting(x) # patches: [b,ps*ps,ph,pw]
		(b,pc,ph,pw) = patch.shape
		# print('patch_shape:', patch.shape)
		patch = patch.reshape((b, pc, ph*pw)).permute(0, 2, 1).reshape((b, ph*pw, self.patch_size, self.patch_size)) # patches: [b, ph*pw, ps, ps]
		# print('patch_shape:', patch.shape)
		y_patch = self.patch_net(patch)[-1] # [b, ph*pw, ps, ps]
		# print('ypatch_shape:', y_patch.shape)
		y_patch = y_patch.reshape((b,ph*pw,pc)).permute(0,2,1).reshape((b,pc,ph,pw)) # [b,ps*ps,ph,pw]
		y_restore = self.restore(y_patch) # [b,c,h,w]
		# print('restore_shape:', y_restore.shape)
		y = F.relu(self.bn(self.conv(y_restore))) # [b,c,h,w]
		# print('y_shape:', y.shape)
		return y


@BACKBONES.register_module()
class DeformPatchNet(BaseModule):
	def __init__(self,
				 imsize,
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
				 resnet_stagesnum = 4):
		super(DeformPatchNet, self).__init__()
		self.head = DeformHead(in_channels = in_channels,
							   mid_channels = mid_channels,
							   out_channels = mid_channels,
							   conv_blocknum = head_conv_blocknum,
							   offset_blocknum = head_offset_blocknum,
							   offset_size = head_offset_size,
							   feature_num = feature_num)
		self.patch = PatchNet(imsize = imsize, 
							  in_channels = mid_channels, 
							  out_channels = out_channels,
							  patch_size = patch_size,
							  patch_stride = patch_stride,
							  feature_num = feature_num,
							  depth = resnet_depth,
							  num_stages = resnet_stagesnum)

	def forward(self,x):
		deform_x = self.head(x)
		y = self.patch(deform_x)
		return y




