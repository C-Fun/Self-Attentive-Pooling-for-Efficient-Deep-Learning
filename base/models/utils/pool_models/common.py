import torch
import torch.nn as nn
from collections import OrderedDict

from .lip import LIP_BASE
from .nlp import NLP_BASE
from .dfm_nlp import DFMNLP_BASE
from .dfmixp import DFMIXP_BASE
from .mixp import MIXP_BASE
from .gaussian_pool import GaussianPooling2d

class Pool2d(nn.Module):
	def __init__(self, pool_module, in_channels, kernel_size=2, stride=2, padding=0, win_norm=True, **kwargs):
		super(Pool2d, self).__init__()
		self.win_norm = win_norm
		self.logit = nn.Sequential(OrderedDict([
			('pool_weight', pool_module(in_channels, **kwargs))
			]))
		self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
	def forward(self, x):
		weight = self.logit(x)

		if self.win_norm==True:
			y = self.pool(x * weight) / self.pool(weight)
		else:
			y = self.pool(x * weight)
		return y


def lip2d(inc, kernel_size, stride, padding, **kwargs):
	return Pool2d(LIP_BASE, inc, kernel_size, stride, padding)

def nlp2d(inc, kernel_size, stride, padding, win_norm=True, **kwargs):
	return Pool2d(NLP_BASE, inc, kernel_size, stride, padding, win_norm, **kwargs)

def dfm_nlp2d(inc, kernel_size, stride, padding, win_norm=True, **kwargs):
	return Pool2d(DFMNLP_BASE, inc, kernel_size, stride, padding, win_norm, **kwargs)

def dfmixp2d(inc, kernel_size, stride, padding, win_norm=True, **kwargs):
	return Pool2d(DFMIXP_BASE, inc, kernel_size, stride, padding, win_norm, **kwargs)

def mixp2d(inc, kernel_size, stride, padding, win_norm=True, **kwargs):
	return Pool2d(MIXP_BASE, inc, kernel_size, stride, padding, win_norm, **kwargs)



# def skip_pool2d(**kwargs):
# 	return None

def max_pool2d(kernel_size, stride, padding, **kwargs):
	return nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)

def avg_pool2d(kernel_size, stride, padding, **kwargs):
	return nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)

def gaussian_pool2d(inc, kernel_size, stride, padding, **kwargs):
	return GaussianPooling2d(inc, kernel_size, stride, padding, stochasticity='HWCN')