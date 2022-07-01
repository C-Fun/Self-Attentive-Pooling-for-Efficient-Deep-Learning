import json

import torch
import torch.nn as nn

from .backbones.resnet_v2 import resnet18_v2, resnet50_v2
from .backbones.resnet import resnet18, resnet50
from .backbones.mobilenet import MobileNetV2 as mobilenetv2
from .backbones.mobilenet_v2 import MobileNetV2 as mobilenetv2_v2

from .utils.pool_models.common import *
from .utils.dynamic_convs.dynamic_conv import Dynamic_conv2d

def name_parse(name):
    cfg = None

    name_list = name.split('-')
    print("Name Type: 'prepool(optional)'-'netarch'-'pooling'-'strides'")
    print("Check Name List:", name_list)

    if 'prepool' in name_list:
        assert name_list[0] == 'prepool'
        prepool = True
        stride_str = name_list[-1]
        strides = [int(stride_str)]
    else:
        prepool = False
        stride_str = name_list[-1]
        strides = [int(_) for _ in list(stride_str)]

    net_str = name_list[-3]
    if 'dy' in net_str:
        _ctype = 'dynamic'
        _arch = net_str[2:]
    else:
        _ctype = 'norm'
        _arch = net_str


    pool_str = name_list[-2]
    def pool_dict(pt, s, ps=None, rt=None, nh=None, ct=None, wn=None):
        return {'_ptype': pt,
                '_stride': s,

                '_psize': ps,
                '_dim_reduced_ratio': rt,
                '_num_heads': nh,
                '_conv2d': ct,
                '_win_norm': wn
            }
    if prepool:
        def pool_parse(pool_str):
            _ptype = pool_str.split('_')[0]
            if pool_str in ('skip', 'maxp', 'lip', 'gaussian_pool'):
                _ptype = pool_str
                s0 = strides[0]
                pdicts = (pool_dict(_ptype, s0))
            elif _ptype in ('nlp', 'dfmnlp', 'mixp', 'dfmixp'):
                s0 = strides[0]
                pdicts = (
                    pool_dict(_ptype, s0, ps=2, rt=1, nh=8, ct=_ctype, wn=True),
                )
                if 'reduced' in pool_str:
                    pdicts = (
                        pool_dict(_ptype, s0, ps=s0, rt=1/4, nh=8, ct=_ctype, wn=True),
                    )
                if 'headfix2' in pool_str:
                    for _pd in pdicts:
                        _pd['_num_heads'] = 2
            else:
                raise Exception("Undefined Pooling Type!")
            return pdicts
    else:
        def pool_parse(pool_str):
            _ptype = pool_str.split('_')[0]
            if pool_str in ('skip', 'maxp', 'lip', 'gaussian_pool'):
                _ptype = pool_str
                pdicts = []
                for s in strides:
                    pdicts.append(pool_dict('skip', s) if s==1 else pool_dict(_ptype, s))
                pdicts = tuple(pdicts)
            elif _ptype in ('nlp', 'dfmnlp', 'mixp', 'dfmixp'):
                s1, s2, s3, s4 = strides
                pdicts = (
                    pool_dict('skip', s1) if s1==1 else pool_dict(_ptype, s1, ps=1, rt=1, nh=4, ct=_ctype, wn=True),
                    pool_dict('skip', s2) if s2==1 else pool_dict(_ptype, s2, ps=1, rt=1, nh=8, ct=_ctype, wn=True),
                    pool_dict('skip', s3) if s3==1 else pool_dict(_ptype, s3, ps=1, rt=1, nh=16, ct=_ctype, wn=True),
                    pool_dict('skip', s4) if s4==1 else pool_dict(_ptype, s4, ps=1, rt=1, nh=32, ct=_ctype, wn=True),
                )
                if 'reduced' in pool_str:
                    pdicts = (
                        pool_dict('skip', s1) if s1==1 else pool_dict(_ptype, s1, ps=16, rt=1/4, nh=4, ct=_ctype, wn=True),
                        pool_dict('skip', s2) if s2==1 else pool_dict(_ptype, s2, ps=8, rt=1/4, nh=8, ct=_ctype, wn=True),
                        pool_dict('skip', s3) if s3==1 else pool_dict(_ptype, s3, ps=4, rt=1/4, nh=16, ct=_ctype, wn=True),
                        pool_dict('skip', s4) if s4==1 else pool_dict(_ptype, s4, ps=2, rt=1/4, nh=32, ct=_ctype, wn=True),
                    )
                if 'headfix2' in pool_str:
                    for _pd in pdicts:
                        _pd['_num_heads'] = 2
            else:
                raise Exception("Undefined Pooling Type!")
            return pdicts

    pdicts = pool_parse(pool_str)

    if _arch in ('resnet18', 'resnet18_v2', 'resnet50', 'resnet50_v2'):
        cfg = {
            'arch': _arch,
            'conv1': None,
            'pool': None,
            'layer1': None,
            'layer2': None,
            'layer3': None,
            'layer4': None,
        }

        if prepool:
            cfg['conv1'] = {'_conv2d':_ctype, 'pool_cfg': pool_dict('skip', 1)}
            cfg['pool'] = {'_conv2d':_ctype, 'pool_cfg': pdicts[0]}
            cfg['layer1'] = {'_conv2d':_ctype, 'pool_cfg': pool_dict('skip', 1)}
            cfg['layer2'] = {'_conv2d':_ctype, 'pool_cfg': pool_dict('skip', 2)}
            cfg['layer3'] = {'_conv2d':_ctype, 'pool_cfg': pool_dict('skip', 2)}
            cfg['layer4'] = {'_conv2d':_ctype, 'pool_cfg': pool_dict('skip', 2)}
        else:
            cfg['conv1'] = {'_conv2d':_ctype, 'pool_cfg': pool_dict('skip', 2)}
            cfg['pool'] = {'_conv2d':None, 'pool_cfg': pool_dict('maxp', 2)}
            cfg['layer1'] = {'_conv2d':_ctype, 'pool_cfg': pdicts[0]}
            cfg['layer2'] = {'_conv2d':_ctype, 'pool_cfg': pdicts[1]}
            cfg['layer3'] = {'_conv2d':_ctype, 'pool_cfg': pdicts[2]}
            cfg['layer4'] = {'_conv2d':_ctype, 'pool_cfg': pdicts[3]}

    elif _arch in ('mobilenet', 'mobilenet_v2'):
        cfg = {
            'arch': _arch,
            'conv1': None,
            'layer1': None,
            'layer2': None,
            'layer3': None,
            'layer4': None,
            'conv2': None,
        }

        assert prepool==False
        cfg['conv1'] = {'_conv2d':_ctype, 'pool_cfg': {}}
        cfg['layer1'] = {'_conv2d':_ctype, 'pool_cfg': pdicts[0]}
        cfg['layer2'] = {'_conv2d':_ctype, 'pool_cfg': pdicts[1]}
        cfg['layer3'] = {'_conv2d':_ctype, 'pool_cfg': pdicts[2]}
        cfg['layer4'] = {'_conv2d':_ctype, 'pool_cfg': pdicts[3]}
        cfg['conv2'] = {'_conv2d':_ctype, 'pool_cfg': {}}

    if cfg==None:
        raise Exception("Undefined Network Type!")

    print(json.dumps(cfg, sort_keys=False, indent=4, separators=(',', ': ')))

    return cfg






# def name_parse(name):
#     cfg = None
#
#     # ==== resnet18 1222 ==== #
#     if name == "resnet18":
#         from .configs.resnet18.base import cfg
#     if name == "resnet18_lip":
#         from .configs.resnet18.lip import cfg
#     if name == "resnet18_gaussian_pool":
#         from .configs.resnet18.gaussian_pool import cfg
#     if name == "resnet18_nlp":
#         from .configs.resnet18.nlp import cfg
#     if name == "resnet18_dfmnlp":
#         from .configs.resnet18.dfmnlp import cfg
#
#     if name == "resnet18_nlp_headfix2":
#         from .configs.resnet18.nlp_headfix2 import cfg
#
#     # ==== resnet18 2222 ==== #
#     if name == "resnet18_2222":
#         from .configs.resnet18_2222.base import cfg
#     if name == "resnet18_lip_2222":
#         from .configs.resnet18_2222.lip import cfg
#     if name == "resnet18_gaussian_pool_2222":
#         from .configs.resnet18_2222.gaussian_pool import cfg
#     if name == "resnet18_nlp_2222":
#         from .configs.resnet18_2222.nlp import cfg
#     if name == "resnet18_dfmnlp_2222":
#         from .configs.resnet18_2222.dfmnlp import cfg
#
#     if name == "resnet18_nlp_headfix2_2222":
#         from .configs.resnet18_2222.nlp_headfix2 import cfg
#
#     # ==== resnet18 4222 ==== #
#     if name == "resnet18_4222":
#         from .configs.resnet18_4222.base import cfg
#     if name == "resnet18_lip_4222":
#         from .configs.resnet18_4222.lip import cfg
#     if name == "resnet18_gaussian_pool_4222":
#         from .configs.resnet18_4222.gaussian_pool import cfg
#     if name == "resnet18_nlp_4222":
#         from .configs.resnet18_4222.nlp import cfg
#     if name == "resnet18_dfmnlp_4222":
#         from .configs.resnet18_4222.dfmnlp import cfg
#
#     if name == "resnet18_nlp_headfix2_4222":
#         from .configs.resnet18_4222.nlp_headfix2 import cfg
#
#     # ==== resnet50 1222 ==== #
#     if name == "resnet50":
#         from .configs.resnet50.base import cfg
#     if name == "resnet50_lip":
#         from .configs.resnet50.lip import cfg
#     if name == "resnet50_gaussian_pool":
#         from .configs.resnet50.gaussian_pool import cfg
#     if name == "resnet50_nlp":
#         from .configs.resnet50.nlp import cfg
#     if name == "resnet50_nlp_reduced":
#         from .configs.resnet50.nlp_reduced import cfg
#     if name == "resnet50_dfmnlp":
#         from .configs.resnet50.dfmnlp import cfg
#     if name == "resnet50_mixp":
#         from .configs.resnet50.mixp import cfg
#
#     if name == "resnet50_nlp_headfix2":
#         from .configs.resnet50.nlp_headfix2 import cfg
#
#     # ==== resnet50 2222 ==== #
#     if name == "resnet50_2222":
#         from .configs.resnet50_2222.base import cfg
#     if name == "resnet50_lip_2222":
#         from .configs.resnet50_2222.lip import cfg
#     if name == "resnet50_gaussian_pool_2222":
#         from .configs.resnet50_2222.gaussian_pool import cfg
#     if name == "resnet50_nlp_2222":
#         from .configs.resnet50_2222.nlp import cfg
#     if name == "resnet50_dfmnlp_2222":
#         from .configs.resnet50_2222.dfmnlp import cfg
#
#     if name == "resnet50_nlp_headfix2_2222":
#         from .configs.resnet50_2222.nlp_headfix2 import cfg
#
#     # ==== resnet50 4222 ==== #
#     if name == "resnet50_4222":
#         from .configs.resnet50_4222.base import cfg
#     if name == "resnet50_lip_4222":
#         from .configs.resnet50_4222.lip import cfg
#     if name == "resnet50_gaussian_pool_4222":
#         from .configs.resnet50_4222.gaussian_pool import cfg
#     if name == "resnet50_nlp_4222":
#         from .configs.resnet50_4222.nlp import cfg
#     if name == "resnet50_dfmnlp_4222":
#         from .configs.resnet50_4222.dfmnlp import cfg
#
#     if name == "resnet50_nlp_headfix2_4222":
#         from .configs.resnet50_4222.nlp_headfix2 import cfg
#
#
#     # ==== dyresnet18 1222 ==== #
#     if name == "dyresnet18":
#         from .configs.dyresnet18.base import cfg
#     if name == "dyresnet18_lip":
#         from .configs.dyresnet18.lip import cfg
#     if name == "dyresnet18_gaussian_pool":
#         from .configs.dyresnet18.gaussian_pool import cfg
#     if name == "dyresnet18_nlp":
#         from .configs.dyresnet18.nlp import cfg
#     if name == "dyresnet18_dfmnlp":
#         from .configs.dyresnet18.dfmnlp import cfg
#
#     # ==== dyresnet18 2222 ==== #
#     if name == "dyresnet18_2222":
#         from .configs.dyresnet18_2222.base import cfg
#     if name == "dyresnet18_lip_2222":
#         from .configs.dyresnet18_2222.lip import cfg
#     if name == "dyresnet18_gaussian_pool_2222":
#         from .configs.dyresnet18_2222.gaussian_pool import cfg
#     if name == "dyresnet18_nlp_2222":
#         from .configs.dyresnet18_2222.nlp import cfg
#     if name == "dyresnet18_dfmnlp_2222":
#         from .configs.dyresnet18_2222.dfmnlp import cfg
#
#     # ==== dyresnet18 4222 ==== #
#     if name == "dyresnet18_4222":
#         from .configs.dyresnet18_4222.base import cfg
#     if name == "dyresnet18_lip_4222":
#         from .configs.dyresnet18_4222.lip import cfg
#     if name == "dyresnet18_gaussian_pool_4222":
#         from .configs.dyresnet18_4222.gaussian_pool import cfg
#     if name == "dyresnet18_nlp_4222":
#         from .configs.dyresnet18_4222.nlp import cfg
#     if name == "dyresnet18_dfmnlp_4222":
#         from .configs.dyresnet18_4222.dfmnlp import cfg
#
#
#     # ==== dyresnet50 1222 ==== #
#     if name == "dyresnet50":
#         from .configs.dyresnet50.base import cfg
#     if name == "dyresnet50_lip":
#         from .configs.dyresnet50.lip import cfg
#     if name == "dyresnet50_gaussian_pool":
#         from .configs.dyresnet50.gaussian_pool import cfg
#     if name == "dyresnet50_nlp":
#         from .configs.dyresnet50.nlp import cfg
#     if name == "dyresnet50_nlp_reduced":
#         from .configs.dyresnet50.nlp_reduced import cfg
#     if name == "dyresnet50_dfmnlp":
#         from .configs.dyresnet50.dfmnlp import cfg
#     if name == "dyresnet50_mixp":
#         from .configs.dyresnet50.mixp import cfg
#
#     # ==== dyresnet50 2222 ==== #
#     if name == "dyresnet50_2222":
#         from .configs.dyresnet50_2222.base import cfg
#     if name == "dyresnet50_lip_2222":
#         from .configs.dyresnet50_2222.lip import cfg
#     if name == "dyresnet50_gaussian_pool_2222":
#         from .configs.dyresnet50_2222.gaussian_pool import cfg
#     if name == "dyresnet50_nlp_2222":
#         from .configs.dyresnet50_2222.nlp import cfg
#     if name == "dyresnet50_dfmnlp_2222":
#         from .configs.dyresnet50_2222.dfmnlp import cfg
#
#     # ==== dyresnet50 4222 ==== #
#     if name == "dyresnet50_4222":
#         from .configs.dyresnet50_4222.base import cfg
#     if name == "dyresnet50_lip_4222":
#         from .configs.dyresnet50_4222.lip import cfg
#     if name == "dyresnet50_gaussian_pool_4222":
#         from .configs.dyresnet50_4222.gaussian_pool import cfg
#     if name == "dyresnet50_nlp_4222":
#         from .configs.dyresnet50_4222.nlp import cfg
#     if name == "dyresnet50_dfmnlp_4222":
#         from .configs.dyresnet50_4222.dfmnlp import cfg
#
#     # ==== prepool resnet18 ==== #
#     if name == "prepool_resnet18":
#         from .configs.resnet18.base import cfg
#
#     if name == "prepool_resnet18_maxp_4":
#         from .configs.prepool_resnet18.maxp_4 import cfg
#     if name == "prepool_resnet18_maxp_6":
#         from .configs.prepool_resnet18.maxp_6 import cfg
#     if name == "prepool_resnet18_maxp_8":
#         from .configs.prepool_resnet18.maxp_8 import cfg
#
#     if name == "prepool_resnet18_lip_4":
#         from .configs.prepool_resnet18.lip_4 import cfg
#     if name == "prepool_resnet18_lip_6":
#         from .configs.prepool_resnet18.lip_6 import cfg
#     if name == "prepool_resnet18_lip_8":
#         from .configs.prepool_resnet18.lip_8 import cfg
#
#     if name == "prepool_resnet18_gaussian_pool_4":
#         from .configs.prepool_resnet18.gaussian_pool_4 import cfg
#     if name == "prepool_resnet18_gaussian_pool_6":
#         from .configs.prepool_resnet18.gaussian_pool_6 import cfg
#     if name == "prepool_resnet18_gaussian_pool_8":
#         from .configs.prepool_resnet18.gaussian_pool_8 import cfg
#
#     if name == "prepool_resnet18_nlp_4":
#         from .configs.prepool_resnet18.nlp_4 import cfg
#     if name == "prepool_resnet18_nlp_6":
#         from .configs.prepool_resnet18.nlp_6 import cfg
#     if name == "prepool_resnet18_nlp_8":
#         from .configs.prepool_resnet18.nlp_8 import cfg
#
#     # ==== prepool resnet50 ==== #
#     if name == "prepool_resnet50":
#         from .configs.resnet50.base import cfg
#
#     if name == "prepool_resnet50_maxp_4":
#         from .configs.prepool_resnet50.maxp_4 import cfg
#     if name == "prepool_resnet50_maxp_6":
#         from .configs.prepool_resnet50.maxp_6 import cfg
#     if name == "prepool_resnet50_maxp_8":
#         from .configs.prepool_resnet50.maxp_8 import cfg
#
#     if name == "prepool_resnet50_lip_4":
#         from .configs.prepool_resnet50.lip_4 import cfg
#     if name == "prepool_resnet50_lip_6":
#         from .configs.prepool_resnet50.lip_6 import cfg
#     if name == "prepool_resnet50_lip_8":
#         from .configs.prepool_resnet50.lip_8 import cfg
#
#     if name == "prepool_resnet50_gaussian_pool_4":
#         from .configs.prepool_resnet50.gaussian_pool_4 import cfg
#     if name == "prepool_resnet50_gaussian_pool_6":
#         from .configs.prepool_resnet50.gaussian_pool_6 import cfg
#     if name == "prepool_resnet50_gaussian_pool_8":
#         from .configs.prepool_resnet50.gaussian_pool_8 import cfg
#
#     if name == "prepool_resnet50_nlp_4":
#         from .configs.prepool_resnet50.nlp_4 import cfg
#     if name == "prepool_resnet50_nlp_6":
#         from .configs.prepool_resnet50.nlp_6 import cfg
#     if name == "prepool_resnet50_nlp_8":
#         from .configs.prepool_resnet50.nlp_8 import cfg
#
#
#     if name == 'mobilenet':
#         from .configs.mobilenet.base import cfg
#
#     if cfg==None:
#         raise Exception("Undefined Network Type!")
#
#     return cfg


def pool2d(_ptype):
    if _ptype=='skip':
        return None
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
    elif _ptype=='dfmixp':
        return dfmixp2d
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
        if cfg['_ptype'] == None:
            cfg['_ptype'] = 'skip'
        self._ptype = cfg['_ptype']
        self._pool2d = pool2d(cfg['_ptype'])
        self._ksize = cfg['_stride']
        self._stride = cfg['_stride']
        self._padding = 0
        self._psize = cfg['_psize']
        self._dim_reduced_ratio = cfg['_dim_reduced_ratio']
        self._num_heads = cfg['_num_heads']
        self._conv2d = conv2d(cfg['_conv2d'])
        self._win_norm = cfg['_win_norm']


class Config:
    def __init__(self, cfg):
        self._conv2d = conv2d(cfg['_conv2d'])

        pool_keys = ('_ptype', '_stride', '_psize', '_dim_reduced_ratio', '_num_heads', '_conv2d', '_win_norm')
        pool_cfg = cfg['pool_cfg']
        for key in pool_keys:
            if key not in pool_cfg.keys():
                pool_cfg[key] = None
        self.pool_cfg = PoolConfig(pool_cfg)


def cfg_parse(cfg):
    for k in cfg.keys():
        if k=='arch':
            arch = cfg[k]
        else:
            cfg[k] = Config(cfg[k])
    return arch, cfg


class Network(nn.Module):
    def __init__(self, cfg, pth_file=None, **kwargs):
        super(Network, self).__init__()
        arch, cfg = cfg_parse(cfg)

        # ========== backbone =================
        if arch == 'resnet18':
            self.net = resnet18(cfg, pth_file=pth_file, **kwargs)
        elif arch == 'resnet18_v2':
            self.net = resnet18_v2(cfg, pth_file=pth_file, **kwargs)
        elif arch == 'resnet50':
            self.net = resnet50(cfg, pth_file=pth_file, **kwargs)
        elif arch == 'resnet50_v2':
            self.net = resnet50_v2(cfg, pth_file=pth_file, **kwargs)
        elif arch == 'mobilenet':
            self.net = mobilenetv2(cfg, **kwargs)
        elif arch == 'mobilenet_v2':
            self.net = mobilenetv2_v2(cfg, **kwargs)
        else:
            raise Exception("Undefined Backbone Type!")

    def forward(self, x):
        outs = self.net(x)
        return outs


# class NetworkByName(Network):
#     def __init__(self, name, pth_file=None, **kwargs):
#         cfg = name_parse(name)
#         super(NetworkByName, self).__init__(cfg, pth_file, **kwargs)