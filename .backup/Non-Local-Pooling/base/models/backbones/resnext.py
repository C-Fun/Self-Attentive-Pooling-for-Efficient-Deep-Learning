'''
New for ResNeXt:
1. Wider bottleneck
2. Add group for conv2
'''

import torch.nn as nn
import math

__all__ = ['ResNeXt', 'resnext18', 'resnext34', 'resnext50', 'resnext101',
           'resnext152']

def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, groups=groups)

def pooling(inc, pool_cfg):
    return pool_cfg._pool2d(inc=inc, \
                            kernel_size=pool_cfg._ksize, \
                            stride=pool_cfg._stride, \
                            padding=pool_cfg._padding, \
                            patch_size=pool_cfg._psize, \
                            dim_reduced_ratio=pool_cfg._dim_reduced_ratio, \
                            num_heads=pool_cfg._num_heads, \
                            conv2d=pool_cfg._conv2d, \
                            win_norm=pool_cfg._win_norm)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, num_group=32, pool_cfg=None):
        super(BasicBlock, self).__init__()

        if stride > 1:
            if pool_cfg._ptype == 'skip': # skip-pooling
                s = stride
                self.pooling = None
            else:
                s = 1
                self.pooling = pooling(planes, pool_cfg) # other pooling
        else:
            s = 1 # no pooling
            self.pooling = None

        self.conv1 = conv3x3(inplanes, planes, s)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, groups=num_group)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        if self.pooling is not None:
            out = self.pooling(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, num_group=32, pool_cfg=None):
        super(Bottleneck, self).__init__()

        if stride > 1:
            if pool_cfg._ptype == 'skip': # skip-pooling
                s = stride
                self.pooling = None
            else:
                s = 1
                self.pooling = pooling(planes * 4, pool_cfg) # other pooling
        else:
            s = 1 # no pooling
            self.pooling = None

        self.conv1 = nn.Conv2d(inplanes, planes*2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes*2)
        self.conv2 = nn.Conv2d(planes*2, planes*2, kernel_size=3, stride=s,
                               padding=1, bias=False, groups=num_group)
        self.bn2 = nn.BatchNorm2d(planes*2)
        self.conv3 = nn.Conv2d(planes*2, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        if self.pooling is not None:
            out = self.pooling(out)

        return out


class ResNeXt(nn.Module):

    def __init__(self, cfg, block, layers, num_classes=1000, num_group=32):
        self.inplanes = 64
        super(ResNeXt, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], num_group, cfg['layer1']) # 1
        self.layer2 = self._make_layer(block, 128, layers[1], num_group, cfg['layer2']) # 2
        self.layer3 = self._make_layer(block, 256, layers[2], num_group, cfg['layer3']) # 2
        self.layer4 = self._make_layer(block, 512, layers[3], num_group, cfg['layer4']) # 2
        # self.avgpool = nn.AvgPool2d(4, stride=1) # 7
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, num_group, layer_cfg):
        downsample = None

        pool_cfg = layer_cfg.pool_cfg
        stride = pool_cfg._stride

        if stride != 1 or self.inplanes != planes * block.expansion:
            if pool_cfg._ptype == 'skip':
                stride_ = stride
            else:
                stride_ = 1
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride_, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, num_group=num_group, pool_cfg=pool_cfg))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, num_group=num_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnext18(cfg, **kwargs):
    """Constructs a ResNeXt-18 model.
    """
    model = ResNeXt(cfg, BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnext34(cfg, **kwargs):
    """Constructs a ResNeXt-34 model.
    """
    model = ResNeXt(cfg, BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnext50(cfg, **kwargs):
    """Constructs a ResNeXt-50 model.
    """
    model = ResNeXt(cfg, Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnext101(cfg, **kwargs):
    """Constructs a ResNeXt-101 model.
    """
    model = ResNeXt(cfg, Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnext152(cfg, **kwargs):
    """Constructs a ResNeXt-152 model.
    """
    model = ResNeXt(cfg, Bottleneck, [3, 8, 36, 3], **kwargs)
    return model