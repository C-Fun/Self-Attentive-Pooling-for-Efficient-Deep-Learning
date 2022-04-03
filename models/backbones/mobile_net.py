import torch.nn as nn
import math

def conv3x3(conv2d, in_planes, out_planes, stride=1, groups=1, dilation=1):
    return conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(conv2d, in_planes, out_planes, stride=1):
    return conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv_bn(conv2d, inp, oup, stride):
    return nn.Sequential(
        conv2d(inp, oup, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(conv2d, inp, oup):
    return nn.Sequential(
        conv2d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class InvertedResidual(nn.Module):
    def __init__(self, conv2d, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                # conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
                conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                conv2d(hidden_dim, oup, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                conv2d(inp, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                # conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
                conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                conv2d(hidden_dim, oup, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, conv2d, pool2d, num_classes=1000, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        self.conv2d = conv2d
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        # input_channel = make_divisible(input_channel * width_mult)  # first channel is always 32!
        self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        
        # self.features = [conv_bn(conv2d, 3, input_channel, 2)]
        s1 = 2 # 1 for cifar10
        if pool2d == None:
            self.features = [conv_bn(conv2d, 3, input_channel, s1)]
        else:
            self.features = [conv_bn(conv2d, 3, input_channel, 1)]
            self.features.append(pool2d(input_channel, kernel_size=s1, stride=s1, patch_size=s1, embed_dim=None, num_heads=2))
        
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:
                    if s==1 or pool2d==None:
                        self.features.append(block(conv2d, input_channel, output_channel, s, expand_ratio=t))
                    else:
                        self.features.append(block(conv2d, input_channel, output_channel, 1, expand_ratio=t))
                        self.features.append(pool2d(output_channel, kernel_size=s, stride=s, patch_size=s, embed_dim=None, num_heads=s))
                else:
                    self.features.append(block(conv2d, input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        
        # building last several layers
        self.features.append(conv_1x1_bn(conv2d, input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Linear(self.last_channel, num_classes)

        # self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, self.conv2d):
    #             n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #             m.weight.data.normal_(0, math.sqrt(2. / n))
    #             if m.bias is not None:
    #                 m.bias.data.zero_()
    #         elif isinstance(m, nn.BatchNorm2d):
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()
    #         elif isinstance(m, nn.Linear):
    #             n = m.weight.size(1)
    #             m.weight.data.normal_(0, 0.01)
    #             m.bias.data.zero_()


def mobilenet_v2(conv2d, pool2d, pretrained=False, pth_file=None, **kwargs):
    model = MobileNetV2(conv2d, pool2d, width_mult=1, **kwargs)

    if pretrained:
        try:
            from torch.hub import load_state_dict_from_url
        except ImportError:
            from torch.utils.model_zoo import load_url as load_state_dict_from_url
        state_dict = load_state_dict_from_url(
            'https://www.dropbox.com/s/47tyzpofuuyyv1b/mobilenetv2_1.0-f2a8633.pth.tar?dl=1', progress=True)
        model.load_state_dict(state_dict)
    return model