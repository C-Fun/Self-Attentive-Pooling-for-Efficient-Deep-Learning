import warnings

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
from mmcv.runner import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.models.builder import BACKBONES
from mmdet.models.utils import ResLayer
from mmdet.models import ResNet

from RPIXEL.models.deformable_models.deform_conv import DeformConv2D

class ResBlock(nn.Module):
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

class DeformHead(nn.Module):
	def __init__(self, 
				 in_channels,
				 feat_channels,
				 out_channels,
				 conv_blocknum,
				 offset_blocknum,
				 offset_size):
		super(DeformHead, self).__init__()

		# conv layers
		self.convs = nn.Sequential()
		self.convs.add_module("x_in", nn.Conv2d(in_channels, feat_channels, kernel_size=1, padding=0))
		for i in range(conv_blocknum):
			self.convs.add_module("x_block_"+str(i+1), ResBlock(feat_channels, feat_channels))
		self.convs.add_module("x_out", nn.Conv2d(feat_channels, feat_channels, kernel_size=1, padding=0))
		self.convs_bn = nn.BatchNorm2d(feat_channels)

		# offset layers
		self.offsets = nn.Sequential()
		self.offsets.add_module("offset_in", nn.Conv2d(feat_channels, feat_channels, kernel_size=1, padding=0))
		for i in range(offset_blocknum):
			self.offsets.add_module("offset_block_"+str(i+1), ResBlock(feat_channels, feat_channels))
		self.offsets.add_module("offset_out", nn.Conv2d(feat_channels, 2*offset_size**2, kernel_size=1, padding=0))

		# deformable conv
		self.deform_conv = DeformConv2D(feat_channels, out_channels, kernel_size=offset_size, padding=1)
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

class kconv(nn.Module):
	def __init__(self,in_channels,out_channels,kernel_size,padding):
		super(kconv,self).__init__()
		self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding)
		self.bn = nn.BatchNorm2d(num_features=out_channels)
		self.relu = nn.ReLU(inplace=True)
	def forward(self,x):
		k = self.relu(self.bn(self.conv(x)))
		return k

class LFE(nn.Module):
	def __init__(self, in_channels, kconv_channels, kconv_size, resnet_stages_num, resnet_out_indices):
		super(LFE,self).__init__()
		self.kconvs = nn.Sequential()
		for i, ksize in enumerate(kconv_size):
			self.kconvs.add_module('kconv'+str(i+1), kconv(in_channels, kconv_channels, ksize, ksize//2))
		self.resnet = ResNet(depth=50,
							in_channels=kconv_channels*len(kconv_size),
							num_stages=resnet_stages_num,
							out_indices=resnet_out_indices,
							frozen_stages=-1,
							norm_cfg=dict(type='BN', requires_grad=True),
							norm_eval=True,
							style='pytorch',
							init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
							)

	def forward(self,x):
		kconvs = []
		for k in range(len(self.kconvs)):
			kconvs.append(self.kconvs[k](x))
		kconvs = torch.cat(kconvs, dim=1)
		outs = self.resnet(kconvs)
		return outs

class Deconv(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1):
		super(Deconv, self).__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.deconv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, output_padding=output_padding)
		self.bn = nn.BatchNorm2d(num_features=out_channels)

	def forward(self,x):
		return F.relu(self.bn(self.deconv(x)))

class PatchNet(nn.Module):
	def __init__(self,
				 imsize,
				 in_channels,
				 kconv_channels,
				 patch_size,
				 patch_stride,
				 kconv_size,
				 resnet_stages_num, 
				 resnet_out_indices):
		super(PatchNet, self).__init__()
		self.patch_size = patch_size
		patch_num = (1 + (imsize[0]-patch_size)//patch_stride) * (1 + (imsize[1]-patch_size)//patch_stride)

		self.patch_splitting = nn.Conv2d(in_channels,patch_size**2,kernel_size=patch_size,stride=patch_stride,padding=0)

		self.sub_net = LFE(1, kconv_channels, kconv_size, resnet_stages_num, resnet_out_indices)

		self.restore = nn.Sequential()
		for i, order in enumerate(resnet_out_indices):
			feat_size = int(patch_size / (2**(order+2)))
			feat_stride = int(patch_stride * (feat_size/patch_size))
			self.restore.add_module('restore_'+str(i+1), Deconv(feat_size**2, 1, kernel_size=feat_size, stride=feat_stride, padding=0, dilation=1, output_padding=0))
		
		self.patch_num = patch_num

	def forward(self,x):
		# x: [b,c,h,w]
		(b,c,h,w) = x.shape

		# patches: [b,ps*ps,ph,pw]
		patch = self.patch_splitting(x)
		(pb,pc,ph,pw) = patch.shape

		# patch: [b*ph*pw, 1, ps, ps]
		patch = patch.reshape((b, pc, ph*pw)).permute(0, 2, 1).reshape((b*ph*pw, 1, self.patch_size, self.patch_size))
		# print('patch_shape:', patch.shape)
		
		y_outs = self.sub_net(patch)
		outs = []
		for i in range(len(y_outs)):
			# y_out: [b*ph*pw, yc, ys, ys]
			y_out = y_outs[i]
			(_, yc, yh, yw) = y_out.shape
			
			y_out = y_out.reshape((b, ph*pw, yc, yh*yw)).permute(0, 2, 3, 1).reshape((b*yc, yh*yw, ph, pw))
			# print('yout_'+str(i+1)+':', y_out.shape) # [b*yc, ys*ys, ph, pw]
			
			out = self.restore[i](y_out) # [b*yc, 1, fh, fw]
			(_, _, oh, ow) = out.shape
			
			out = out.reshape((b, yc, oh, ow))
			# print('out_'+str(i+1)+':', out.shape) # [b, yc, ps*ps, ph*pw]
			
			outs.append(out)

		return outs

@BACKBONES.register_module()
class DeformPatchNet(BaseModule):
	def __init__(self,
				 imsize,
				 head_in_channels = 3,
				 head_feat_channels = 8,
				 head_out_channels = 1,
				 head_conv_blocknum = 0,
				 head_offset_blocknum = 1,
				 head_offset_size = 3,
				 patch_in_channels = 1,
				 patch_kconv_channels = 8,
				 patch_size = 32,
				 patch_stride = 32,
				 patch_kconv_size = [1, 3, 5],
				 resnet_stages_num = 4,
				 resnet_out_indices = (0, 1, 2, 3)
				 ):
		super(DeformPatchNet, self).__init__()
		self.head = DeformHead(in_channels = head_in_channels,
							   feat_channels = head_feat_channels,
							   out_channels = head_out_channels,
							   conv_blocknum = head_conv_blocknum,
							   offset_blocknum = head_offset_blocknum,
							   offset_size = head_offset_size)
		self.patch = PatchNet(imsize = imsize,
							  in_channels = patch_in_channels,
							  kconv_channels = patch_kconv_channels,
							  patch_size = patch_size,
							  patch_stride = patch_stride,
							  kconv_size = patch_kconv_size,
							  resnet_stages_num = resnet_stages_num,
							  resnet_out_indices = resnet_out_indices)

	def forward(self,x):
		deform_x = self.head(x)
		y = self.patch(deform_x)
		return y




