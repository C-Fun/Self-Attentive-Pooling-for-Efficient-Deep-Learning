_arch = 'resnet18_v2'
_ctype = 'dynamic'
_ptype = 'gaussian_pool'

cfg = {'arch': _arch,
	'conv1': {'_conv2d': _ctype,
			    'pool_cfg': {'_ptype': 'skip',
			   				 '_stride': 2,
			   				}
			   },

	'pool': {'_conv2d': None,
			    'pool_cfg': {'_ptype': 'maxp',
			   				 '_stride': 2,
			   				}
			   },

	'layer1': {'_conv2d': _ctype,
			    'pool_cfg': {'_ptype': 'skip',
			   				 '_stride': 1,
			   				}
			   },

	'layer2': {'_conv2d': _ctype,
			    'pool_cfg': {'_ptype': _ptype,
			   				 '_stride': 2,
			   				}
			   },

	'layer3': {'_conv2d': _ctype,
			    'pool_cfg': {'_ptype': _ptype,
			   				 '_stride': 2,
			   				}
			   },

	'layer4': {'_conv2d': _ctype,
			    'pool_cfg': {'_ptype': _ptype,
			   				 '_stride': 2,
			   				}
			   },
}