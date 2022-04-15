import torch
import torch.nn as nn

from .backbones.resnet import resnet50
from .backbones.mobile_net import mobilenetv2

from .utils.dynamic_conv import dynamic_convolution_generator
from .utils.pool_models import LIP2d, NLP2d, MixedPool, DeformNLP, PosEncodeNLP

class Network(nn.Module):
    def __init__(self, cfg, pretrained=False, pth_file=None, **kwargs):
        super(Network, self).__init__()
        # ====== conv1 args parse =========
        conv1_type = cfg.conv1[0]
        if conv1_type == 'normal':
            cfg.conv1[0] = nn.Conv2d
        else:
            raise Exception("Undefined Conv1 Type!")

        # ====== conv2d args parse =========
        conv2d_type = cfg._convtype
        if conv2d_type == 'normal':
            cfg.conv2d = nn.Conv2d
        elif conv2d_type == 'dynamic':
            cfg.conv2d = dynamic_convolution_generator(4, 4)
        else:
            raise Exception("Undefined Conv2d Type!")

        # ====== pool1 args parse =========
        try:
            print(cfg.pool1)
        except:
            cfg.pool1 = None           

        if cfg.pool1 != None:
            pool1_type = cfg.pool1[0]
            if pool1_type == 'maxpool':
                cfg.pool1[0] = nn.MaxPool2d
            elif pool1_type == 'lip':
                cfg.pool1[0] = LIP2d
            elif pool1_type == 'nlp':
                cfg.pool1[0] = NLP2d
            elif pool1_type == 'mixp':
                cfg.pool1[0] = MixedPool
            elif pool1_type == 'dfmnlp':
                cfg.pool1[0] = DeformNLP
            elif pool1_type == 'penlp':
                cfg.pool1[0] = PosEncodeNLP
            else:
                raise Exception("Undefined Pool1 Type!")

        # ====== pool params args parse =========
        add_keys = ['ksize', 'psize', 'dim_reduced_ratio', 'num_heads']
        if cfg._poolparams != None:
            for i in range(len(cfg._poolparams)):
                pool_param = cfg._poolparams[i]
                if 'type' not in pool_param.keys():
                    raise Exception("Need to define the pool type!")
                if 'stride' not in pool_param.keys():
                    raise Exception("Need to define the stride!")

                pool_type = pool_param['type']
                if pool_type == 'none':
                    pool2d = None
                elif pool_type == 'lip':
                    pool2d = LIP2d
                elif pool_type == 'nlp':
                    pool2d = NLP2d
                elif pool_type == 'mixp':
                    pool2d = MixedPool
                elif pool_type == 'dfmnlp':
                    pool2d = DeformNLP
                elif pool_type == 'penlp':
                    pool2d = PosEncodeNLP
                else:
                    raise Exception("Undefined Pool2d Type!")

                cfg._poolparams[i]['pool2d'] = pool2d
                for key in add_keys:
                    if key not in pool_param.keys():
                        cfg._poolparams[i][key] = None

        # ========== backbone =================
        if cfg._backbone == 'resnet50':
            self.net = resnet50(cfg, pretrained=pretrained, pth_file=pth_file, **kwargs)
        elif cfg._backbone == 'mobilenet':
            self.net = mobilenetv2(cfg, pretrained=pretrained, pth_file=pth_file, **kwargs)
    def forward(self, x):
        outs = self.net(x)
        return outs