_arch = 'resnet50'
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
			    			 '_ksize': 2,
			   				 '_stride': 2,
			   				 '_padding': 0,
			   				}
			   },

	'layer1': {'_conv2d': _ctype,
			    'pool_cfg': {'_ptype': 'skip',
			   				 '_stride': 1,
			   				}
			   },

	'layer2': {'_conv2d': _ctype,
			    'pool_cfg': {'_ptype': _ptype,
			    			 '_ksize': 2,
			   				 '_stride': 2,
			   				 '_padding': 0,
			   				}
			   },

	'layer3': {'_conv2d': _ctype,
			    'pool_cfg': {'_ptype': _ptype,
			    			 '_ksize': 2,
			   				 '_stride': 2,
			   				 '_padding': 0,
			   				}
			   },

	'layer4': {'_conv2d': _ctype,
			    'pool_cfg': {'_ptype': _ptype,
			    			 '_ksize': 2,
			   				 '_stride': 2,
			   				 '_padding': 0,
			   				}
			   },
}