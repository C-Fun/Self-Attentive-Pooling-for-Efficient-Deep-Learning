import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from einops import rearrange, reduce, repeat

from .position_encoding import build_position_encoding, NestedTensor

class NLP_PER_CHANNEL_BASE(nn.Module):
	def __init__(self, in_channels, patch_size=1, num_heads=2, position_embedding='learned', conv2d=nn.Conv2d, **kwargs):
		super(NLP_PER_CHANNEL_BASE, self).__init__()

		self.in_channels = in_channels
		self.patch_size = patch_size

		self.num_heads = num_heads

		embed_dim = max(round(patch_size ** 2 / num_heads), 1) * num_heads

		self.downsample = nn.Sequential(
						  conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0),
						  nn.BatchNorm2d(embed_dim),
						  nn.ReLU(inplace=True)
						)

		self.multihead_attn = MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

		self.restore = nn.Sequential(
					conv2d(embed_dim, 1, kernel_size=1, stride=1, padding=0),
					nn.BatchNorm2d(1),
					nn.Sigmoid()
				)

		self.pos_embed = build_position_encoding(hidden_dim=embed_dim, position_embedding=position_embedding)

	def forward(self, x):
		b,c,h,w = x.shape
		p = self.patch_size

		print('x:',x.shape)

		attn = []
		for ch in range(c):
			x_res = x[:,ch,:,:].reshape((b,1,h,w))
			print('x_res:', x_res.shape)

			downsampled = self.downsample(x_res)

			print('downsample:', downsampled.shape)

			nested_downsampled = NestedTensor(downsampled, None)
			pos_embed = self.pos_embed(nested_downsampled)
			pos_embed = rearrange(pos_embed, 'b c h w -> b (h w) c')

			embed = rearrange(downsampled, 'b c h w -> b (h w) c') + pos_embed

			print('embed:', embed.shape)

			attn_seq, attn_weights = self.multihead_attn(embed, embed, embed)
			print('attn seq:', attn_seq.shape)
			attn_ch = rearrange(attn_seq, 'b (h w) c -> b c h w', h=h//p, w=w//p)
			attn_ch = self.restore(F.interpolate(attn_ch, size=(h,w), mode='nearest'))

			print('attn ch:', attn_ch.shape)

			attn.append(attn_ch)

			print(attn_ch.shape)
		attn = torch.cat(attn, axis=1)
		print(attn.shape)
		weight = torch.exp(attn)
		return weight

	# def forward(self, x):
	# 	b,c,h,w = x.shape
	# 	p = self.patch_size
	#
	# 	print('x:', x.shape)
	#
	# 	x_res = rearrange(x, 'b c h w -> (b c) 1 h w')
	#
	# 	print('x_res:', x_res.shape)
	#
	# 	downsampled = self.downsample(x_res) # (b c) p^2 h//p w//p
	#
	# 	print('downsample:', downsampled.shape)
	#
	# 	# nested_downsampled = NestedTensor(downsampled, None)
	# 	# pos_embed = self.pos_embed(nested_downsampled)
	# 	# pos_embed = rearrange(pos_embed, 'b c h w -> b (h w) c')
	# 	#
	# 	# embed = rearrange(downsampled, 'b c h w -> b (h w) c') + pos_embed
	#
	# 	embed = rearrange(downsampled, 'b c h w -> b (h w) c')
	# 	print('embed:', embed.shape)
	#
	# 	attn_seq, attn_weights = self.multihead_attn(embed, embed, embed)
	# 	print('attn seq:', attn_seq.shape)
	# 	attn = rearrange(attn_seq, 'b (h w) c -> b c h w', h=h//p, w=w//p)
	# 	attn = self.restore(F.interpolate(attn, size=(h,w), mode='nearest'))
	#
	# 	print('attn:',attn.shape)
	# 	attn = rearrange(attn, '(b c) 1 h w -> b c h w', b=b, c=c)
	#
	# 	weight = torch.exp(attn)
	# 	return weight