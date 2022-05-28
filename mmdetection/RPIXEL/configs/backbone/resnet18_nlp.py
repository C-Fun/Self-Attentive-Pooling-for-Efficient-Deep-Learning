from RPIXEL.configs.backbone._base.resnet18 import _backbone, _convtype, conv1, pool1

_pooltype = "penlp"
_poolparams = [
				{'type':'none', 'stride':1},
				{'type':_pooltype, 'ksize':2, 'stride':2, 'psize':4, 'dim_reduced_ratio':1.0, 'num_heads':2},
				{'type':_pooltype, 'ksize':2, 'stride':2, 'psize':2, 'dim_reduced_ratio':1.0, 'num_heads':4},
				{'type':_pooltype, 'ksize':2, 'stride':2, 'psize':1, 'dim_reduced_ratio':1.0, 'num_heads':8}
			]
