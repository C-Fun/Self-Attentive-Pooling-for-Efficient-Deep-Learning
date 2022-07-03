"""
Creates a MobileNetV2 Model as defined in:
Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen. (2018).
MobileNetV2: Inverted Residuals and Linear Bottlenecks
arXiv preprint arXiv:1801.04381.
import from https://github.com/tonylins/pytorch-mobilenet-v2
"""

import torch.nn as nn
import math

__all__ = ['mobilenetv2']


def pooling(inc, pool_cfg):
    return pool_cfg._pool2d(inc=inc,
							kernel_size=pool_cfg._ksize,
							stride=pool_cfg._stride,
							padding=pool_cfg._padding,
							patch_size=pool_cfg._psize,
							dim_reduced_ratio=pool_cfg._dim_reduced_ratio,
							num_heads=pool_cfg._num_heads,
							conv2d=pool_cfg._conv2d,
							win_norm=pool_cfg._win_norm,
                            )


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride, conv2d=nn.Conv2d):
    return nn.Sequential(
        conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup, conv2d=nn.Conv2d):
    return nn.Sequential(
        conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, conv2d=nn.Conv2d, pool_cfg=None):
        super(InvertedResidual, self).__init__()
        # assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if stride > 1:
            if pool_cfg._ptype == 'skip': # skip-pooling
                s = stride
                self.pooling = None
            else:
                s = 1
                self.pooling = pooling(oup, pool_cfg) # other pooling
        else:
            s = 1 # no pooling
            self.pooling = None

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                conv2d(hidden_dim, hidden_dim, 3, s, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                conv2d(hidden_dim, hidden_dim, 3, s, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )


    def forward(self, x):
        if self.identity:
            out = x + self.conv(x)
        else:
            out = self.conv(x)

        if self.pooling is not None:
            out = self.pooling(out)

        return out


class MobileNetV2(nn.Module):
    def __init__(self, cfg, num_classes=1000, width_mult=1., use_fc_layer=True, out_indices=None):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.arch_cfgs = [
            # # t, c, n, s
            [1,  16, 1, -1], # 1
            [6,  24, 2, -1], # 2
            [6,  32, 3, -1], # 2
            [6,  64, 4, -1], # 2
            [6,  96, 3, -1], # 1
            [6, 160, 3, -1], # 2
            [6, 320, 1, -1], # 1
        ]
        self._use_fc_layer = use_fc_layer
        self._out_indices = out_indices

        # building first layer
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        self.conv1 = conv_3x3_bn(3, input_channel, 2, conv2d=cfg['conv1']._conv2d)

        layers = []
        # building inverted residual blocks
        block = InvertedResidual
        for layer_i, (t, c, n, _) in enumerate(self.arch_cfgs):
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)

            layer_cfg = cfg['layer' + str(layer_i + 1)]
            pool_cfg = layer_cfg.pool_cfg
            s = pool_cfg._stride

            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t, conv2d=layer_cfg._conv2d, pool_cfg=pool_cfg))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)

        # building last several layers
        output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
        self.conv2 = conv_1x1_bn(input_channel, output_channel, conv2d=cfg['conv2']._conv2d)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, num_classes)

        self._initialize_weights()

    def forward(self, x):
        out = self.conv1(x)

        outs = []
        for layer_i in self.features:
            out = layer_i(out)
            outs.append(out)

        if self._use_fc_layer:
            y = self.conv2(out)
            y = self.avgpool(y)
            y = y.view(y.size(0), -1)
            y = self.classifier(y)
            return y
        else:
            if self._out_indices == None:
                return outs
            else:
                return outs[self._out_indices]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def mobilenetv2(**kwargs):
    """
    Constructs a MobileNet V2 model
    """
    return MobileNetV2(**kwargs)