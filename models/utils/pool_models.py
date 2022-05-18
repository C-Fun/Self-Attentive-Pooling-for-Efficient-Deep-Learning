import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn import MultiheadAttention

from einops import rearrange, reduce, repeat
from .deform_conv import DeformConv2d as DCN
from .position_encoding import build_position_encoding, NestedTensor

__all__ = ['LIP2d', 'NLP2d', 'MixedPool', 'DeformNLP', 'PosEncodeNLP', 'GaussianP2d', 'PeNLPChLoc']

class LIP_BASE(nn.Module):
	def __init__(self, in_channels, **kwargs):
		super(LIP_BASE, self).__init__()

		self.logit = nn.Sequential(
					 nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
					 nn.BatchNorm2d(in_channels),
					 nn.Sigmoid()
				)
	def forward(self, x):
		b,c,h,w = x.shape
		logit = self.logit(x)
		weight = torch.exp(logit)
		return weight

class NLP_BASE(nn.Module):
	def __init__(self, in_channels, patch_size=1, embed_dim=None, num_heads=2):
		super(NLP_BASE, self).__init__()

		self.in_channels = in_channels
		self.patch_size = patch_size
		self.embed_dim = embed_dim
		self.num_heads = num_heads

		# if embed_dim == None or embed_dim == in_channels*self.patch_size**2:
		# 	embed_dim = in_channels*self.patch_size**2
		# 	self.dim_reduction = None
		# else:
		# 	self.dim_reduction = nn.Linear(in_channels*self.patch_size**2, embed_dim)

		if embed_dim == None:
			self.embed_dim = in_channels

		self.downsample = nn.Sequential(
						  nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0),
						  nn.BatchNorm2d(embed_dim),
						  nn.ReLU(inplace=True)
						)

		self.multihead_attn = MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
		
		self.restore = nn.Sequential(
					nn.Conv2d(embed_dim, in_channels, kernel_size=1, stride=1, padding=0),
					nn.BatchNorm2d(in_channels),
					nn.Sigmoid()
				)
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
		attn = self.restore(F.interpolate(attn, size=(h,w), mode='nearest'))

		weight = torch.exp(attn)
		return weight

class MIXP_BASE(nn.Module):
	def __init__(self, in_channels, patch_size=2, embed_dim=None, num_heads=2):
		super(MIXP_BASE, self).__init__()
		self.local = LIP_BASE(in_channels)
		self.non_local = NLP_BASE(in_channels, patch_size=patch_size, embed_dim=embed_dim, num_heads=num_heads)
	def forward(self, x):
		local_weight = self.local(x)
		nonlocal_weight = self.non_local(x)
		mixed_weight = local_weight * nonlocal_weight
		return mixed_weight

class DeformNLP_BASE(NLP_BASE):
	def __init__(self, in_channels, patch_size=2, embed_dim=None, num_heads=2):
		super(DeformNLP_BASE, self).__init__(in_channels, patch_size, embed_dim, num_heads)
		self.downsample = None
		self.dcn = DCN(self.in_channels, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size, padding=0)
		self.bn = nn.BatchNorm2d(self.embed_dim)
		self.relu = nn.ReLU(inplace=True)
	def forward(self, x):
		b,c,h,w = x.shape
		p = self.patch_size

		# embed = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
		# if self.dim_reduction != None:
		# 	embed = self.dim_reduction(embed)

		downsampled, offset = self.dcn(x)
		downsampled = self.relu(self.bn(downsampled))
		embed = rearrange(downsampled, 'b c h w -> b (h w) c')

		attn_seq, attn_weights = self.multihead_attn(embed, embed, embed)
		attn = rearrange(attn_seq, 'b (h w) c -> b c h w', h=h//p, w=w//p)
		attn = self.restore(F.interpolate(attn, size=(h,w), mode='nearest'))

		weight = torch.exp(attn)
		return weight

class PosEncodeNLP_BASE(NLP_BASE):
	def __init__(self, in_channels, patch_size=2, embed_dim=None, num_heads=2, position_embedding='learned'):
		super(PosEncodeNLP_BASE, self).__init__(in_channels, patch_size, embed_dim, num_heads)
		self.pos_embed = build_position_encoding(hidden_dim=self.embed_dim, position_embedding=position_embedding)
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


class PeNLPChLoc_BASE(nn.Module):
	def __init__(self, in_channels, patch_size=1, embed_dim=None, num_heads=2, position_embedding='learned'):
		super(PeNLPChLoc_BASE, self).__init__()

		self.in_channels = in_channels
		self.patch_size = patch_size
		self.embed_dim = embed_dim
		self.num_heads = num_heads

		if embed_dim == None:
			self.embed_dim = in_channels

		self.downsample = nn.Sequential(
						  nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0),
						  nn.BatchNorm2d(embed_dim),
						  nn.ReLU(inplace=True)
						)

		self.spatial_attn = MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
		self.global_avgpool = nn.AdaptiveAvgPool2d((1,1))
		self.channel_attn = MultiheadAttention(embed_dim=1, num_heads=1, batch_first=True)

		self.restore = nn.Sequential(
					nn.Conv2d(embed_dim, in_channels, kernel_size=1, stride=1, padding=0),
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

		spatial_embed = rearrange(downsampled + pos_embed, 'b c h w -> b (h w) c')
		spatial_seq, spatial_weights = self.spatial_attn(spatial_embed, spatial_embed, spatial_embed)
		spatial_attn = rearrange(spatial_seq, 'b (h w) c -> b c h w', h=h//p, w=w//p)

		channel_embed = rearrange(self.global_avgpool(downsampled), 'b c h w -> b c (h w)')
		channel_seq, channel_weights = self.channel_attn(channel_embed, channel_embed, channel_embed)
		channel_attn = rearrange(channel_seq, 'b c (h w) -> b c h w', h=1, w=1)

		attn = spatial_attn * channel_attn
		attn = self.restore(F.interpolate(attn, size=(h,w), mode='nearest'))

		weight = torch.exp(attn)
		return weight


class POOL_BASE(nn.Module):
	def __init__(self, BASE, in_channels, kernel_size=2, stride=2, padding=0, **kwargs):
		super(POOL_BASE, self).__init__()
		self.logit = BASE(in_channels, **kwargs)
		self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
	def forward(self, x):
		weight = self.logit(x)
		y = self.pool(x * weight) / self.pool(weight)
		return y


def LIP2d(in_channels, kernel_size=2, stride=2, padding=0, **kwargs):
	return POOL_BASE(LIP_BASE, in_channels, kernel_size, stride, padding, **kwargs)
def NLP2d(in_channels, kernel_size=2, stride=2, padding=0, **kwargs):
	return POOL_BASE(NLP_BASE, in_channels, kernel_size, stride, padding, **kwargs)
def MixedPool(in_channels, kernel_size=2, stride=2, padding=0, **kwargs):
	return POOL_BASE(MIXP_BASE, in_channels, kernel_size, stride, padding, **kwargs)
def DeformNLP(in_channels, kernel_size=2, stride=2, padding=0, **kwargs):
	return POOL_BASE(DeformNLP_BASE, in_channels, kernel_size, stride, padding, **kwargs)
def PosEncodeNLP(in_channels, kernel_size=2, stride=2, padding=0, **kwargs):
	return POOL_BASE(PosEncodeNLP_BASE, in_channels, kernel_size, stride, padding, **kwargs)
def PeNLPChLoc(in_channels, kernel_size=2, stride=2, padding=0, **kwargs):
	return POOL_BASE(PeNLPChLoc_BASE, in_channels, kernel_size, stride, padding, **kwargs)


class GaussianPooling2d(nn.AvgPool2d):
    def __init__(self, num_features, kernel_size, stride=None, padding=0, ceil_mode=False,
                 count_include_pad=True, hidden_node=None, stochasticity='HWCN', eps=1e-6):
        if stochasticity != 'HWCN' and stochasticity != 'CN' and stochasticity is not None:
            raise ValueError("gaussian pooling stochasticity has to be 'HWCN'/'CN' or None, "
                         "but got {}".format(stochasticity))
        if hidden_node is None:
            hidden_node = num_features // 2

        super(GaussianPooling2d, self).__init__(kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode,
                    count_include_pad=count_include_pad)
        self.eps = eps
        self.stochasticity = stochasticity

        self.ToHidden = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(num_features, hidden_node, kernel_size=1,  padding=0, bias=True),
            nn.BatchNorm2d(hidden_node),
            nn.ReLU(False),
        )
        self.ToMean = nn.Sequential(
            nn.Conv2d(hidden_node, num_features, kernel_size=1,  padding=0, bias=True),
            nn.BatchNorm2d(num_features),
        )
        self.ToSigma = nn.Sequential(
            nn.Conv2d(hidden_node, num_features, kernel_size=1,  padding=0, bias=True),
            nn.BatchNorm2d(num_features),
            nn.Sigmoid()
        )
        self.activation = nn.Softplus()
        
    def forward(self, input):
        mu0 = F.avg_pool2d(input, self.kernel_size, self.stride, self.padding, self.ceil_mode, self.count_include_pad)
        sig0= F.avg_pool2d(input**2, self.kernel_size, self.stride, self.padding, self.ceil_mode, self.count_include_pad)
        sig0= torch.sqrt(torch.clamp(sig0 - mu0**2, self.eps))

        Z = self.ToHidden(input)
        MU = self.ToMean(Z)

        if self.training and self.stochasticity is not None:
            SIGMA = self.ToSigma(Z)
            if self.stochasticity == 'HWCN':
                size = sig0.size()
            else:
                size = [sig0.size(0), sig0.size(1), 1, 1]
            W = self.activation(MU + SIGMA * 
                torch.randn(size, dtype=sig0.dtype, layout=sig0.layout, device=sig0.device))
        else:
            W = self.activation(MU)

        return mu0 + W*sig0

def GaussianP2d(in_channels, kernel_size=2, stride=2, padding=0, **kwargs):
	return GaussianPooling2d(in_channels, kernel_size, stride, padding)


if __name__ == '__main__':
	x = torch.randn(2, 512, 160, 320).cuda()
	model = PeNLPChLoc(512, kernel_size=2, stride=2, padding=0, patch_size=16, embed_dim=16, num_heads=2).cuda()

	# x = x.to('cuda:0')
	# model.to('cuda')

	y = model(x)
	print(y.shape)
