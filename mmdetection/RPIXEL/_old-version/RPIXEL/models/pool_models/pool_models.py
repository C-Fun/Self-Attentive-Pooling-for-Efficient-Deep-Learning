import warnings


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn import MultiheadAttention

from einops import rearrange, reduce, repeat

from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
from mmcv.runner import BaseModule
from mmcv.ops import DeformConv2dPack as DCN

from mmdet.models.builder import BACKBONES


class EntropyPool2d(BaseModule):
	def __init__(self, in_channels):
		super(EntropyPool2d, self).__init__()
		self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
		self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
	def forward(self, x):
		b,c,h,w = x.shape
		attn = self.conv(x)
		px = F.softmax(attn, dim=1)
		entropy = -torch.sum(px*torch.log(px), dim=1)
		entropy_max, _ = torch.max(entropy.reshape(b, h*w), dim=1)
		_lambda = 1-entropy/entropy_max.reshape(b, 1, 1).repeat(1, h, w)
		_lambda = _lambda.unsqueeze(1).repeat(1,c,1,1)
		downsampled = self.pool(x*attn)
		return downsampled


class GlobalPool2d(BaseModule):
	def __init__(self, in_channels):
		super(GlobalPool2d, self).__init__()
		self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
		self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
	def forward(self, x):
		b,c,h,w = x.shape
		attn = self.conv(x) # b,c,h,w
		_max, _ = torch.max(attn.reshape(b,c,h*w), dim=2)
		attn = _max * attn
		downsampled = self.pool(x * attn)
		return downsampled


class LIP2d(BaseModule):
	def __init__(self, in_channels):
		super(LIP2d, self).__init__()
		self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
		self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
	def forward(self, x):
		b,c,h,w = x.shape
		logit = self.conv(x)
		weight = torch.exp(logit)
		y = self.pool(x*weight)/self.pool(weight)
		return y


class MultiheadAttnPool(BaseModule):
	def __init__(self, in_channels, kernel_size=2, stride=2, patch_size=2, embed_dim=None, num_heads=2):
		super(MultiheadAttnPool, self).__init__()

		self.patch_size = patch_size;
		
		# if embed_dim == None or embed_dim == in_channels*self.patch_size**2:
		# 	embed_dim = in_channels*self.patch_size**2
		# 	self.dim_reduction = None
		# else:
		# 	self.dim_reduction = nn.Linear(in_channels*self.patch_size**2, embed_dim)

		if embed_dim == None:
			embed_dim = in_channels
		self.downsample = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0)
		
		self.multihead_attn = MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
		# self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))

		self.conv = nn.Conv2d(embed_dim, in_channels, kernel_size=1, stride=1, padding=0)

		self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride)

	def forward(self, x):
		b,c,h,w = x.shape
		p = self.patch_size

		# embed = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
		# if self.dim_reduction != None:
		# 	embed = self.dim_reduction(embed)

		downsampled = self.downsample(x)
		embed = rearrange(downsampled, 'b c h w -> b (h w) c')
		
		attn_seq, attn_weights = self.multihead_attn(embed, embed, embed)
		attn = rearrange(attn_seq, 'b (h w) c -> b c h w', h=h//p, w=w//p)
		attn = self.conv(F.interpolate(attn, size=(h,w), mode='nearest'))

		y = self.pool(x * attn)/self.pool(attn)

		# print('x shape:', x.shape)
		# print('downsampled shape:', downsampled.shape)
		# print('embed shape:', embed.shape)
		# print('attn_seq shape:', attn_seq.shape)
		# print('attn shape:', attn.shape)
		return y


if __name__ == '__main__':
	x = torch.randn(2, 512, 123, 233).cuda()
	model = MultiheadAttnPool(512, kernel_size=2, stride=2, patch_size=16, embed_dim=16, num_heads=2).cuda()

	# x = x.to('cuda:0')
	# model.to('cuda')

	model(x)