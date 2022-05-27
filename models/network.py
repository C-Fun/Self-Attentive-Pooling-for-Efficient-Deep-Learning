import torch
import torch.nn as nn

from .backbones.resnet_v2 import resnet18, resnet50
# from .backbones.mobile_net import mobilenetv2

from .utils.pool_models.common import *
from .utils.dynamic_convs.dynamic_conv import Dynamic_conv2d

def pool2d(_ptype):
    if _ptype=='skip':
        return skip_pool2d
    elif _ptype=='maxp':
        return max_pool2d
    elif _ptype=='avgp':
        return avg_pool2d
    elif _ptype=='lip':
        return lip2d
    elif _ptype=='gaussian_pool':
        return gaussian_pool2d
    elif _ptype=='nlp':
        return nlp2d
    elif _ptype=='dfm_nlp':
        return dfm_nlp2d
    elif _ptype=='mixp':
        return mixp2d
    else:
        raise Exception("Undefined Pooling Type!")

def conv2d(_ctype):
    if _ctype == None:
        return None
    elif _ctype == 'norm':
        return nn.Conv2d
    elif _ctype == 'dynamic':
        return Dynamic_conv2d
    else:
        raise Exception("Undefined Convolutional Type!")

class PoolConfig:
    def __init__(self, cfg):
        self._ptype = cfg['_ptype']
        self._pool2d = pool2d(cfg['_ptype'])
        self._ksize = cfg['_ksize']
        self._stride = cfg['_stride']
        self._padding = cfg['_padding']
        self._psize = cfg['_psize']
        self._dim_reduced_ratio = cfg['_dim_reduced_ratio']
        self._num_heads = cfg['_num_heads']
        self._conv2d = conv2d(cfg['_conv2d'])
        self._win_norm = cfg['_win_norm']


class Config:
    def __init__(self, cfg):
        self._conv2d = conv2d(cfg['_conv2d'])

        pool_keys = ('_ptype', '_ksize', '_stride', '_padding', '_psize', '_dim_reduced_ratio', '_num_heads', '_conv2d', '_win_norm')
        pool_cfg = cfg['pool_cfg']
        for key in pool_keys:
            if key not in pool_cfg.keys():
                pool_cfg[key] = None
        self.pool_cfg = PoolConfig(pool_cfg)


def cfg_parse(cfg):
    for k in cfg.keys():
        if k=='arch':
            continue
        else:
            cfg[k] = Config(cfg[k])
    return cfg


class Network(nn.Module):
    def __init__(self, cfg, pth_file=None, **kwargs):
        super(Network, self).__init__()
        arch = cfg['arch']
        cfg = cfg_parse(cfg)

        # ========== backbone =================
        if arch == 'resnet18':
            self.net = resnet18(cfg, pth_file=pth_file, **kwargs)
        elif arch == 'resnet50':
            self.net = resnet50(cfg, pth_file=pth_file, **kwargs)
        # elif arch == 'mobilenet':
        #     self.net = mobilenetv2(cfg, pth_file=pth_file, **kwargs)
    def forward(self, x):
        outs = self.net(x)
        return outs