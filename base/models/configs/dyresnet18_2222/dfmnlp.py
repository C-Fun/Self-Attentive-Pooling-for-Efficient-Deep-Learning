_arch = 'resnet18'
_ctype = 'dynamic'
_ptype = 'dfm_nlp'
_win_norm = True

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
			    'pool_cfg': {'_ptype': _ptype,
			   				 '_stride': 2,

			   				 '_psize': 1,
			   				 '_dim_reduced_ratio': 1,
			   				 '_num_heads': 4,
			   				 '_conv2d': _ctype,
			   				 '_win_norm': _win_norm
			   				}
			   },

	'layer2': {'_conv2d': _ctype,
			    'pool_cfg': {'_ptype': _ptype,
			   				 '_stride': 2,

			   				 '_psize': 1,
			   				 '_dim_reduced_ratio': 1,
			   				 '_num_heads': 8,
			   				 '_conv2d': _ctype,
			   				 '_win_norm': _win_norm
			   				}
			   },

	'layer3': {'_conv2d': _ctype,
			    'pool_cfg': {'_ptype': _ptype,
			   				 '_stride': 2,

			   				 '_psize': 1,
			   				 '_dim_reduced_ratio': 1,
			   				 '_num_heads': 16,
			   				 '_conv2d': _ctype,
			   				 '_win_norm': _win_norm
			   				}
			   },

	'layer4': {'_conv2d': _ctype,
			    'pool_cfg': {'_ptype': _ptype,
			   				 '_stride': 2,

			   				 '_psize': 1,
			   				 '_dim_reduced_ratio': 1,
			   				 '_num_heads': 32,
			   				 '_conv2d': _ctype,
			   				 '_win_norm': _win_norm
			   				}
			   },
}