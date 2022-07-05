import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from einops import rearrange, reduce, repeat

from .position_encoding import build_position_encoding, NestedTensor

class NLP_BASE(nn.Module):
	def __init__(self, in_channels, patch_size=1, dim_reduced_ratio=1.0, num_heads=2, position_embedding='learned', conv2d=nn.Conv2d):
		super(NLP_BASE, self).__init__()

		self.in_channels = in_channels
		self.patch_size = patch_size

		reduced_dim = round(dim_reduced_ratio * in_channels / num_heads)
		if reduced_dim == 0:
			reduced_dim = 1
		embed_dim =  reduced_dim * num_heads
		self.embed_dim = embed_dim

		self.num_heads = num_heads

		self.downsample = nn.Sequential(
						  conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0),
						  nn.BatchNorm2d(embed_dim),
						  nn.ReLU(inplace=True)
						)

		self.multihead_attn = MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
		
		self.restore = nn.Sequential(
					conv2d(embed_dim, in_channels, kernel_size=1, stride=1, padding=0),
					nn.BatchNorm2d(in_channels),
					nn.Sigmoid()
				)

		self.pos_embed = build_position_encoding(hidden_dim=embed_dim, position_embedding=position_embedding)
	
	def forward(self, x):
		b,c,h,w = x.shape
		p = self.patch_size

		downsampled = self.downsample(x)

		nested_downsampled = NestedTensor(downsampled, None)
		pos_embed = self.pos_embed(nested_downsampled)
		pos_embed = rearrange(pos_embed, 'b c h w -> b (h w) c')

		embed = rearrange(downsampled, 'b c h w -> b (h w) c') + pos_embed

		attn_seq, attn_weights = self.multihead_attn(embed, embed, embed)
		attn = rearrange(attn_seq, 'b (h w) c -> b c h w', h=h//p, w=w//p)
		attn = self.restore(F.interpolate(attn, size=(h,w), mode='nearest')) 

		weight = torch.exp(attn)
		return weight