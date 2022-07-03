import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, reduce, repeat

from ..deform_convs.dcnv2 import DeformToolkit, OffsetConv2d, DeformConv2d, reshape_x_offset
from .nlp import NLP_BASE
from .position_encoding import NestedTensor

class DFMNLP_BASE(NLP_BASE):
	def __init__(self, in_channels, patch_size=2, dim_reduced_ratio=1.0, num_heads=2, position_embedding='learned', conv2d=nn.Conv2d):
		super(DFMNLP_BASE, self).__init__(in_channels, patch_size, dim_reduced_ratio, num_heads, position_embedding, conv2d)
		self.offset_conv = OffsetConv2d(inc=self.in_channels, kernel_size=self.patch_size, stride=self.patch_size, conv2d=conv2d)
		self.downsample = nn.Sequential(
							DeformConv2d(self.in_channels, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size, padding=0, modulation=True, conv2d=conv2d),
							nn.BatchNorm2d(self.embed_dim),
							nn.ReLU(inplace=True)
						)
		self.dfm_toolkit = DeformToolkit(kernel_size=self.patch_size, padding=0, stride=self.patch_size)
	def forward(self, x):
		b,c,h,w = x.shape
		p = self.patch_size

		offset = self.offset_conv(x)

		downsampled = self.downsample((x, offset))

		nested_downsampled = NestedTensor(downsampled, None)
		pos_embed = self.pos_embed(nested_downsampled)
		pos_embed = rearrange(pos_embed, 'b c h w -> b (h w) c')

		embed = rearrange(downsampled, 'b c h w -> b (h w) c') + pos_embed

		attn_seq, attn_weights = self.multihead_attn(embed, embed, embed)
		attn = rearrange(attn_seq, 'b (h w) c -> b c h w', h=h//p, w=w//p)
		attn = self.restore(F.interpolate(attn, size=(h,w), mode='nearest'))

		weight = torch.exp(attn)
		weight = reshape_x_offset(self.dfm_toolkit(weight, -offset), self.patch_size)

		return weight
