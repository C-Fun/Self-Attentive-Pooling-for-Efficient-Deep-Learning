import torch
import torch.nn as nn

from .backbones.resnet import resnet50
from .backbones.mobile_net import mobilenet_v2

from .dynamic_models.dynamic_conv import Dynamic_conv2d
from .pool_models.pool_models import LIP2d, NLP2d, MixedPool

class Network(nn.Module):
    def __init__(self, backbone_type, conv_type, pool_type, pretrained=False, pth_file=None, **kwargs):
        super(Network, self).__init__()
        if conv_type == 'normal':
            conv2d = nn.Conv2d
        elif conv_type == 'dynamic':
            conv2d = Dynamic_conv2d

        if pool_type == 'none':
            pool2d = None
        elif pool_type == 'lip':
            pool2d = LIP2d
        elif pool_type == 'nlp':
            pool2d = NLP2d
        elif pool_type == 'mixp':
            pool2d = MixedPool

        if backbone_type == 'resnet50':
            self.net = resnet50(conv2d, pool2d, pretrained=pretrained, pth_file=pth_file, **kwargs)
        elif backbone_type == 'mobilenet':
            self.net = mobilenet_v2(conv2d, pool2d, pretrained=pretrained, pth_file=pth_file, **kwargs)
    def forward(self, x):
        outs = self.net(x)
        return outs