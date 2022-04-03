import warnings

import torch
import torch.nn as nn


def conv3x3(conv2d, in_planes, out_planes, stride=1, groups=1, dilation=1):
    return conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(conv2d, in_planes, out_planes, stride=1):
    return conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, conv2d, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(conv2d, inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(conv2d, planes, planes)
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

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, conv2d, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(conv2d, inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(conv2d, width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(conv2d, width, planes * self.expansion)
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

        return out


__all__ = ['ResNet', 'resnet50']

class ResNet(nn.Module):

    def __init__(self, conv2d, pool2d, pool_strides, block, layers, num_classes=1000, zero_init_residual=False,
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
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        if pool2d == None:
            self.layer1 = self._make_layer(conv2d, block, 64, layers[0], stride=pool_strides[0])
            self.layer2 = self._make_layer(conv2d, block, 128, layers[1], stride=pool_strides[1],
                                           dilate=replace_stride_with_dilation[0])
            self.layer3 = self._make_layer(conv2d, block, 256, layers[2], stride=pool_strides[2],
                                           dilate=replace_stride_with_dilation[1])
            self.layer4 = self._make_layer(conv2d, block, 512, layers[3], stride=pool_strides[3],
                                           dilate=replace_stride_with_dilation[2])
        else:
            if pool_strides[0]==1:
                self.layer1 = self._make_layer(conv2d, block, 64, layers[0], stride=1)
            else:
                pool_stride = pool_strides[0]
                self.layer1 = nn.Sequential(
                    self._make_layer(conv2d, block, 64, layers[0], stride=1),
                    pool2d(64*block.expansion, kernel_size=pool_stride, stride=pool_stride, patch_size=pool_stride, embed_dim=None, num_heads=pool_stride)
                    )

            if pool_strides[1]==1:
                self.layer2 = self._make_layer(conv2d, block, 128, layers[1], stride=1,
                                           dilate=replace_stride_with_dilation[0])
            else:
                pool_stride = pool_strides[1]
                self.layer2 = nn.Sequential(
                    self._make_layer(conv2d, block, 128, layers[1], stride=1,
                                           dilate=replace_stride_with_dilation[0]),
                    pool2d(128*block.expansion, kernel_size=pool_stride, stride=pool_stride, patch_size=pool_stride, embed_dim=None, num_heads=pool_stride)
                    )

            if pool_strides[2]==1:
                self.layer3 = self._make_layer(conv2d, block, 256, layers[2], stride=1,
                                           dilate=replace_stride_with_dilation[1])
            else:
                pool_stride = pool_strides[2]
                self.layer3 = nn.Sequential(
                    self._make_layer(conv2d, block, 256, layers[2], stride=1,
                                           dilate=replace_stride_with_dilation[1]),
                    pool2d(256*block.expansion, kernel_size=pool_stride, stride=pool_stride, patch_size=pool_stride, embed_dim=None, num_heads=2*pool_stride)
                    )

            if pool_strides[3]==1:
                self.layer4 = self._make_layer(conv2d, block, 512, layers[3], stride=1,
                                           dilate=replace_stride_with_dilation[2])
            else:
                pool_stride = pool_strides[3]
                self.layer4 = nn.Sequential(
                    self._make_layer(conv2d, block, 512, layers[3], stride=1,
                                           dilate=replace_stride_with_dilation[2]),
                    pool2d(512*block.expansion, kernel_size=pool_stride, stride=pool_stride, patch_size=pool_stride, embed_dim=None, num_heads=4*pool_stride)
                    )


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


    def _make_layer(self, conv2d, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(conv2d, self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(conv2d, self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(conv2d, self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        y = self.avgpool(x4)
        y = torch.flatten(y, 1)
        y = self.fc(y)

        return y

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, conv2d, pool2d, pool_strides, block, layers, pretrained, pth_file, **kwargs):
    model = ResNet(conv2d, pool2d, pool_strides, block, layers, **kwargs)
    if pretrained:
        pretrained_resnet = torch.load(pth_file)
        model_dict = model.state_dict()
        common_keys = [k for k in pretrained_resnet.keys() if(k in model_dict.keys())]
        state_dict = {}
        for k in common_keys:
            pv = pretrained_resnet[k]
            mv = model_dict[k]
            if mv.shape!=pv.shape:
                v = pv.expand(mv.shape)
                state_dict[k] = v
            else:
                state_dict[k] = pv
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
    return model


def resnet50(conv2d, pool2d, pool_strides=[1,2,2,2], pretrained=False, pth_file=None, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        pth_file (string): .pth file path
    """
    return _resnet('resnet50', conv2d, pool2d, pool_strides, Bottleneck, [3, 4, 6, 3], pretrained, pth_file,
                   **kwargs)