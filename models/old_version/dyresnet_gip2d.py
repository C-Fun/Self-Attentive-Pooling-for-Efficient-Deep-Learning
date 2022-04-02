import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch.nn.modules.batchnorm import _BatchNorm

from mmcv.ops import DeformConv2dPack as DCN

from .dynamic_models.dynamic_conv import Dynamic_conv2d
from .dynamic_models.dy_resnet import conv1x1, BasicBlock, Bottleneck

from .pool_models.pool_models import GIP2d

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
		   'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
		   'wide_resnet50_2', 'wide_resnet101_2']

# def deform_pool(in_channels, out_channels, kernel_size=1, stride=2, padding=0):
# 	return DCN(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

class ResNet(nn.Module):

	def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
				 groups=1, width_per_group=64, replace_stride_with_dilation=None,
				 norm_layer=None):
		super(ResNet, self).__init__()
		if norm_layer is None:
			norm_layer = nn.BatchNorm2d
		self._norm_layer = norm_layer

		self.inplanes = 64
		self.dilation = 1
		if replace_stride_with_dilation is None:
			# each element in the tuple indicates if we should replace
			# the 2x2 stride with a dilated convolution instead
			replace_stride_with_dilation = [False, False, False]
		if len(replace_stride_with_dilation) != 3:
			raise ValueError("replace_stride_with_dilation should be None "
							 "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
		self.groups = groups
		self.base_width = width_per_group
		self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
							   bias=False)
		self.bn1 = norm_layer(self.inplanes)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

		self.layer1 = self._make_layer(block, 64, layers[0], stride=1)

		self.layer2 = self._make_layer(block, 128, layers[1], stride=1,
									   dilate=replace_stride_with_dilation[0])
		self.pool2 = GIP2d(128*block.expansion, kernel_size=2, stride=2)

		self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
									   dilate=replace_stride_with_dilation[1])
		self.pool3 = GIP2d(256*block.expansion, kernel_size=2, stride=2, padding=0)

		self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
									   dilate=replace_stride_with_dilation[2])
		self.pool4 = GIP2d(512*block.expansion, kernel_size=2, stride=2, padding=0)

		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
		self.fc = nn.Linear(512 * block.expansion, num_classes)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

		# Zero-initialize the last BN in each residual branch,
		# so that the residual branch starts with zeros, and each residual block behaves like an identity.
		# This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
		if zero_init_residual:
			for m in self.modules():
				if isinstance(m, Bottleneck):
					nn.init.constant_(m.bn3.weight, 0)
				elif isinstance(m, BasicBlock):
					nn.init.constant_(m.bn2.weight, 0)

	def update_temperature(self):
		for m in self.modules():
			if isinstance(m, Dynamic_conv2d):
				m.update_temperature()


	def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
		norm_layer = self._norm_layer
		downsample = None
		previous_dilation = self.dilation
		if dilate:
			self.dilation *= stride
			stride = 1
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				conv1x1(self.inplanes, planes * block.expansion, stride),
				norm_layer(planes * block.expansion),
			)
		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
							self.base_width, previous_dilation, norm_layer))
		self.inplanes = planes * block.expansion
		for _ in range(1, blocks):
			layers.append(block(self.inplanes, planes, groups=self.groups,
								base_width=self.base_width, dilation=self.dilation,
								norm_layer=norm_layer))

		return nn.Sequential(*layers)

	def _forward_impl(self, x):
		x = self.maxpool(self.relu(self.bn1(self.conv1(x))))

		x1 = self.layer1(x)
		x2 = self.pool2(self.layer2(x1))
		x3 = self.pool3(self.layer3(x2))
		x4 = self.pool4(self.layer4(x3))

		y = self.avgpool(x4)
		y = torch.flatten(y, 1)
		y = self.fc(y)
		return y

	def forward(self, x):
		return self._forward_impl(x)


# def _resnet(arch, block, layers, pretrained, progress, **kwargs):
# 	model = ResNet(block, layers, **kwargs)
# 	if pretrained:
# 		state_dict = load_state_dict_from_url(model_urls[arch],
# 											  progress=progress)
# 		model.load_state_dict(state_dict)
# 	return model


def _resnet(arch, block, layers, pretrained, pth_file, **kwargs):
	model = ResNet(block, layers, **kwargs)
	if pretrained:
		pretrained_resnet = torch.load(pth_file)
		model_dict = model.state_dict()
		common_keys = [k for k in pretrained_resnet.keys() if(k in model_dict.keys() and 'layer' in k)]
		state_dict = {}
		for k in common_keys:
			pv = pretrained_resnet[k]
			mv = model_dict[k]
			# print(k,':', pv.shape, mv.shape)
			if mv.shape!=pv.shape:
				# print('Shape inconsist:', k)
				# print(mv.shape, pv.shape)
				v = pv.expand(mv.shape)
				# print(v.shape)
				# for i in range(4):
				# 	for j in range(4):
				# 		print(torch.all(v[i,:,:,:,:]==v[j,:,:,:,:]), end=';')
				# print()
				state_dict[k] = v
		model_dict.update(state_dict)
		model.load_state_dict(model_dict)
	return model


def resnet18(pretrained=False, pth_file=None, **kwargs):
	r"""ResNet-18 model from
	`"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
		pth_file (string): .pth file path
	"""
	return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, pth_file,
				   **kwargs)


def resnet34(pretrained=False, pth_file=None, **kwargs):
	r"""ResNet-34 model from
	`"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
		pth_file (string): .pth file path
	"""
	return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, pth_file,
				   **kwargs)


def resnet50(pretrained=False, pth_file=None, **kwargs):
	r"""ResNet-50 model from
	`"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
		pth_file (string): .pth file path
	"""
	return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, pth_file,
				   **kwargs)


def resnet101(pretrained=False, pth_file=None, **kwargs):
	r"""ResNet-101 model from
	`"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
		pth_file (string): .pth file path
	"""
	return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, pth_file,
				   **kwargs)


def resnet152(pretrained=False, pth_file=None, **kwargs):
	r"""ResNet-152 model from
	`"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
		pth_file (string): .pth file path
	"""
	return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, pth_file,
				   **kwargs)


def resnext50_32x4d(pretrained=False, pth_file=None, **kwargs):
	r"""ResNeXt-50 32x4d model from
	`"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
		pth_file (string): .pth file path
	"""
	kwargs['groups'] = 32
	kwargs['width_per_group'] = 4
	return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
				   pretrained, pth_file, **kwargs)


def resnext101_32x8d(pretrained=False, pth_file=None, **kwargs):
	r"""ResNeXt-101 32x8d model from
	`"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
		pth_file (string): .pth file path
	"""
	kwargs['groups'] = 32
	kwargs['width_per_group'] = 8
	return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
				   pretrained, pth_file, **kwargs)


def wide_resnet50_2(pretrained=False, pth_file=None, **kwargs):
	r"""Wide ResNet-50-2 model from
	`"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

	The model is the same as ResNet except for the bottleneck number of channels
	which is twice larger in every block. The number of channels in outer 1x1
	convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
	channels, and in Wide ResNet-50-2 has 2048-1024-2048.

	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
		pth_file (string): .pth file path
	"""
	kwargs['width_per_group'] = 64 * 2
	return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
				   pretrained, pth_file, **kwargs)


def wide_resnet101_2(pretrained=False, pth_file=None, **kwargs):
	r"""Wide ResNet-101-2 model from
	`"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

	The model is the same as ResNet except for the bottleneck number of channels
	which is twice larger in every block. The number of channels in outer 1x1
	convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
	channels, and in Wide ResNet-50-2 has 2048-1024-2048.

	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
		pth_file (string): .pth file path
	"""
	kwargs['width_per_group'] = 64 * 2
	return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
				   pretrained, pth_file, **kwargs)


class DyResNet50_GIP(nn.Module):
	def __init__(self, pretrained=False, pth_file=None, **kwargs):
		super(DyResNet50_GIP, self).__init__()
		self.resnet = resnet50(pretrained=pretrained, pth_file=pth_file, **kwargs)
	def forward(self, x):
		output = self.resnet(x)
		# for i,out in enumerate(outs):
		# 	print(i, out.shape)
		# print()
		return output


if __name__ == '__main__':
	x = torch.randn(2, 3, 1920, 1080).cuda()
	pth_file = '/nas/home/fangc/mmdetection/RPIXEL/checkpoints/resnet50.pth'
	model = resnet50(pretrained=True, pth_file=pth_file)

	# x = x.to('cuda:0')
	# model.to('cuda')

	outs = model(x)
	for out in outs:
		print(out.shape)