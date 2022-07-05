import json

import torch
import torch.nn as nn

from .backbones.resnet_v2 import resnet18_v2, resnet50_v2
from .backbones.resnet import resnet18, resnet50
from .backbones.mobilenet import mobilenetv2 as mobilenetv2
from .backbones.mobilenet_v2 import mobilenetv2 as mobilenetv2_v2

from .utils.pool_models.common import *
from .utils.dynamic_convs.dynamic_conv import Dynamic_conv2d

def name_parse(name):
    cfg = None
    resnet_tpls = ('resnet18', 'resnet18_v2', 'resnet50', 'resnet50_v2')
    mobilenet_tpls = ('mobilenet', 'mobilenet_v2')
    sota_pool_tpls = ('skip', 'maxp', 'avgp', 'lip', 'gaussian_pool')
    my_pool_tpls = ('nlp', 'dfmnlp', 'mixp', 'dfmixp')

    name_list = name.split('-')
    print("Name Type: '[prepool-pooling-stride](optional)'-'netarch-pooling-strides'")
    print("Check Name List:", name_list)

    if 'prepool' in name_list:
        assert name_list[0] == 'prepool'
        prepool = True
        prepool_str = name_list[1]
        prepool_stride_str = name_list[2]
        prepool_stride = int(prepool_stride_str)
    else:
        prepool = False

    net_str = name_list[-3]
    pool_str = name_list[-2]
    stride_str = name_list[-1]
    strides = [int(_) for _ in list(stride_str)]

    if 'dy' in net_str:
        _ctype = 'dynamic'
        _arch = net_str[2:]
    else:
        _ctype = 'norm'
        _arch = net_str

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
        _pre_ptype = prepool_str.split('_')[0]
        _pre_pstride = prepool_stride
        if prepool_str in sota_pool_tpls:
            _pre_ptype = prepool_str
            pre_pdict = pool_dict(_pre_ptype, _pre_pstride)
        elif _pre_ptype in my_pool_tpls:
            pre_pdict = pool_dict(_pre_ptype, _pre_pstride, ps=2, rt=1, nh=8, ct=_ctype, wn=True)
            if 'reduced' in prepool_str:
                pre_pdict = pool_dict(_pre_ptype, _pre_pstride, ps=_pre_pstride, rt=1/4, nh=8, ct=_ctype, wn=True)
            if 'headfix2' in prepool_str:
                pre_pdict['_num_heads'] = 2
        else:
            raise Exception("Undefined Pre-Pooling Type!")

    _ptype = pool_str.split('_')[0]
    if pool_str in sota_pool_tpls:
        _ptype = pool_str
        pdicts = []
        for s in strides:
            pdicts.append(pool_dict('skip', s) if s==1 else pool_dict(_ptype, s))
        pdicts = tuple(pdicts)
    elif _ptype in my_pool_tpls:
        if len(strides)==4:
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
        elif len(strides)==7:
            s1, s2, s3, s4, s5, s6, s7 = strides
            pdicts = (
                pool_dict('skip', s1) if s1==1 else pool_dict(_ptype, s1, ps=1, rt=1, nh=2, ct=_ctype, wn=True),
                pool_dict('skip', s2) if s2==1 else pool_dict(_ptype, s2, ps=1, rt=1, nh=4, ct=_ctype, wn=True),
                pool_dict('skip', s3) if s3==1 else pool_dict(_ptype, s3, ps=1, rt=1, nh=8, ct=_ctype, wn=True),
                pool_dict('skip', s4) if s4==1 else pool_dict(_ptype, s4, ps=1, rt=1, nh=16, ct=_ctype, wn=True),
                pool_dict('skip', s5) if s5==1 else pool_dict(_ptype, s5, ps=1, rt=1, nh=32, ct=_ctype, wn=True),
                pool_dict('skip', s6) if s6==1 else pool_dict(_ptype, s6, ps=1, rt=1, nh=64, ct=_ctype, wn=True),
                pool_dict('skip', s7) if s7==1 else pool_dict(_ptype, s7, ps=1, rt=1, nh=128, ct=_ctype, wn=True),
            )
            if 'reduced' in pool_str:
                pdicts = (
                    pool_dict('skip', s1) if s1==1 else pool_dict(_ptype, s1, ps=4, rt=1, nh=2, ct=_ctype, wn=True),
                    pool_dict('skip', s2) if s2==1 else pool_dict(_ptype, s2, ps=4, rt=1, nh=4, ct=_ctype, wn=True),
                    pool_dict('skip', s3) if s3==1 else pool_dict(_ptype, s3, ps=4, rt=1, nh=8, ct=_ctype, wn=True),
                    pool_dict('skip', s4) if s4==1 else pool_dict(_ptype, s4, ps=2, rt=1, nh=16, ct=_ctype, wn=True),
                    pool_dict('skip', s5) if s5==1 else pool_dict(_ptype, s5, ps=2, rt=1, nh=32, ct=_ctype, wn=True),
                    pool_dict('skip', s6) if s6==1 else pool_dict(_ptype, s6, ps=1, rt=1, nh=64, ct=_ctype, wn=True),
                    pool_dict('skip', s7) if s7==1 else pool_dict(_ptype, s7, ps=1, rt=1, nh=128, ct=_ctype, wn=True),
                )
        if 'headfix2' in pool_str:
            for _pd in pdicts:
                _pd['_num_heads'] = 2
        if 'nowin' in pool_str:
            for _pd in pdicts:
                _pd['_win_norm'] = False
    else:
        raise Exception("Undefined Pooling Type!")

    if _arch in resnet_tpls:
        assert len(strides)==4
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
            cfg['pool'] = {'_conv2d':_ctype, 'pool_cfg': pre_pdict}
        else:
            cfg['conv1'] = {'_conv2d':_ctype, 'pool_cfg': pool_dict('skip', 2)}
            cfg['pool'] = {'_conv2d':None, 'pool_cfg': pool_dict('maxp', 2)}

        cfg['layer1'] = {'_conv2d':_ctype, 'pool_cfg': pdicts[0]}
        cfg['layer2'] = {'_conv2d':_ctype, 'pool_cfg': pdicts[1]}
        cfg['layer3'] = {'_conv2d':_ctype, 'pool_cfg': pdicts[2]}
        cfg['layer4'] = {'_conv2d':_ctype, 'pool_cfg': pdicts[3]}

    elif _arch in mobilenet_tpls:
        assert len(strides)==7
        cfg = {
            'arch': _arch,
            'conv1': None,
            'layer1': None,
            'layer2': None,
            'layer3': None,
            'layer4': None,
            'layer5': None,
            'layer6': None,
            'layer7': None,
            'conv2': None,
        }

        if prepool:
            cfg['conv1'] = {'_conv2d':_ctype, 'pool_cfg': pre_pdict}
        else:
            cfg['conv1'] = {'_conv2d':_ctype, 'pool_cfg': pool_dict('skip', 2)}

        cfg['layer1'] = {'_conv2d':_ctype, 'pool_cfg': pdicts[0]}
        cfg['layer2'] = {'_conv2d':_ctype, 'pool_cfg': pdicts[1]}
        cfg['layer3'] = {'_conv2d':_ctype, 'pool_cfg': pdicts[2]}
        cfg['layer4'] = {'_conv2d':_ctype, 'pool_cfg': pdicts[3]}
        cfg['layer5'] = {'_conv2d':_ctype, 'pool_cfg': pdicts[4]}
        cfg['layer6'] = {'_conv2d':_ctype, 'pool_cfg': pdicts[5]}
        cfg['layer7'] = {'_conv2d':_ctype, 'pool_cfg': pdicts[6]}
        cfg['conv2'] = {'_conv2d':_ctype, 'pool_cfg': {}}

    else:
        raise Exception("Undefined BackBone Type!")

    if cfg==None:
        raise Exception("Undefined Achitecture Type!")

    print(json.dumps(cfg, sort_keys=False, indent=4, separators=(',', ': ')))

    return cfg

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
    elif _ptype=='dfmnlp':
        return dfmnlp2d
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
        print(cfg)
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
            self.net = mobilenetv2(cfg, pth_file=pth_file, **kwargs)
        elif arch == 'mobilenet_v2':
            self.net = mobilenetv2_v2(cfg, pth_file=pth_file, **kwargs)
        else:
            raise Exception("Undefined Backbone Type!")

    def forward(self, x):
        outs = self.net(x)
        return outs