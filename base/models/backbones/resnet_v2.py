import torch
import torch.nn as nn

__all__ = ['ResNet_v2', 'resnet18_v2', 'resnet34_v2', 'resnet50_v2', 'resnet101_v2',
           'resnet152_v2', 'resnext50_32x4d_v2', 'resnext101_32x8d_v2']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, conv2d=nn.Conv2d):
    """3x3 convolution with padding"""
    return conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1, conv2d=nn.Conv2d):
    """1x1 convolution"""
    return conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


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

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, conv2d=nn.Conv2d, pool_cfg=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

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

        self.conv1 = conv3x3(inplanes, planes, s, conv2d=conv2d)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, conv2d=conv2d)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        if self.pooling is not None:
            out = self.pooling(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, conv2d=nn.Conv2d, pool_cfg=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width, conv2d=conv2d)
        self.bn1 = norm_layer(width)

        if stride > 1:
            if pool_cfg._ptype == 'skip': # skip-pooling
                s = stride
                self.pooling = None
            else:
                s = 1
                self.pooling = pooling(planes * self.expansion, pool_cfg) # other pooling
        else:
            s = 1 # no pooling
            self.pooling = None

        self.conv2 = conv3x3(width, width, s, groups, dilation, conv2d=conv2d)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion, conv2d=conv2d)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        if self.pooling is not None:
            out = self.pooling(out)

        return out


class ResNet_v2(nn.Module):

    def __init__(self, cfg, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, use_fc_layer=True):
        super(ResNet_v2, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self._use_fc_layer = use_fc_layer

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

        _conv1 = cfg['conv1']
        if _conv1.pool_cfg._ptype=='skip':
            self.conv1 = _conv1._conv2d(3, self.inplanes, kernel_size=7, stride=_conv1.pool_cfg._stride, padding=3, bias=False)
            self.conv1_pool = None
        else:
            self.conv1 = _conv1._conv2d(3, self.inplanes, kernel_size=7, stride=1, padding=3, bias=False)
            self.conv1_pool = pooling(self.inplanes, _conv1.pool_cfg)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        _pool = cfg['pool']
        if _pool.pool_cfg._ptype=='skip':
            self.pool = None
        else:
            self.pool = pooling(self.inplanes, _pool.pool_cfg)


        _layer1 = cfg['layer1']
        self.layer1 = self._make_layer(block, 64, layers[0], \
                                       stride=_layer1.pool_cfg._stride, \
                                       conv2d=_layer1._conv2d, \
                                       pool_cfg=_layer1.pool_cfg)

        _layer2 = cfg['layer2']
        self.layer2 = self._make_layer(block, 128, layers[1],  \
                                       stride=_layer2.pool_cfg._stride, \
                                       conv2d=_layer2._conv2d, \
                                       pool_cfg=_layer2.pool_cfg)

        _layer3 = cfg['layer3']
        self.layer3 = self._make_layer(block, 256, layers[2],  \
                                       stride=_layer3.pool_cfg._stride, \
                                       conv2d=_layer3._conv2d, \
                                       pool_cfg=_layer3.pool_cfg)

        _layer4 = cfg['layer4']
        self.layer4 = self._make_layer(block, 512, layers[3],  \
                                       stride=_layer4.pool_cfg._stride, \
                                       conv2d=_layer4._conv2d, \
                                       pool_cfg=_layer4.pool_cfg)
        
        if self._use_fc_layer:
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

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, conv2d=nn.Conv2d, pool_cfg=None):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if pool_cfg._ptype == 'skip':
                stride_ = stride
            else:
                stride_ = 1
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride_, conv2d=conv2d),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, conv2d=conv2d, pool_cfg=pool_cfg))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, conv2d=conv2d))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.conv1_pool is not None:
            x = self.conv1_pool(x)
        if self.pool is not None:
            x = self.pool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        if self._use_fc_layer:
            y = self.avgpool(x4)
            y = y.reshape(y.size(0), -1)
            y = self.fc(y)
            return y
        else:
            return (x1, x2, x3, x4)


def _resnet(arch, cfg, block, layers, pth_file, **kwargs):
    model = ResNet_v2(cfg, block, layers, **kwargs)
    if pth_file!=None:
        pretrained_resnet = torch.load(pth_file)
        model_dict = model.state_dict()

        # print(pretrained_resnet.keys())
        # print(model_dict.keys())

        common_keys = [k for k in pretrained_resnet.keys() if k in model_dict.keys()]
        state_dict = {}
        for k in common_keys:
            if 'fc' in k:
                continue
            pv = pretrained_resnet[k]
            mv = model_dict[k]
            if mv.shape!=pv.shape:
                print('Mismatched Shape: {:s}!'.format(k), pv.shape, '=>', mv.shape)
                state_dict[k] = pv.expand(mv.shape)
            else:
                print('Matched Shape: {:s}!'.format(k))
                state_dict[k] = pv

        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
    print(model)
    return model


def resnet18_v2(cfg, pth_file=None, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pth_file: pre-trained file root
    """
    return _resnet('resnet18', cfg, BasicBlock, [2, 2, 2, 2], pth_file,
                   **kwargs)


def resnet34_v2(cfg, pth_file=None, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pth_file: pre-trained file root
    """
    return _resnet('resnet34', cfg, BasicBlock, [3, 4, 6, 3], pth_file,
                   **kwargs)


def resnet50_v2(cfg, pth_file=None, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pth_file: pre-trained file root
    """
    return _resnet('resnet50', cfg, Bottleneck, [3, 4, 6, 3], pth_file,
                   **kwargs)


def resnet101_v2(cfg, pth_file=None, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pth_file: pre-trained file root
    """
    return _resnet('resnet101', cfg, Bottleneck, [3, 4, 23, 3], pth_file,
                   **kwargs)


def resnet152_v2(cfg, pth_file=None, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pth_file: pre-trained file root
    """
    return _resnet('resnet152', cfg, Bottleneck, [3, 8, 36, 3], pth_file,
                   **kwargs)


def resnext50_32x4d_v2(cfg, pth_file=None, **kwargs):
    """Constructs a ResNeXt-50 32x4d model.
    Args:
        pth_file: pre-trained file root
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', cfg, Bottleneck, [3, 4, 6, 3],
                   pth_file, **kwargs)


def resnext101_32x8d_v2(cfg, pretrained=False, progress=True, **kwargs):
    """Constructs a ResNeXt-101 32x8d model.
    Args:
        pth_file: pre-trained file root
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', cfg, Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)