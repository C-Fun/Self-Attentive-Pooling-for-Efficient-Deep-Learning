import torch
import torch.nn as nn

class LIP_BASE(nn.Module):
	def __init__(self, in_channels):
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