from .._base.resnet import _backbone, _convtype, conv1, pool1

_pooltype = "penlp"
_poolparams = [
				{'type':_pooltype, 'ksize':4, 'stride':4, 'psize':1, 'dim_reduced_ratio':1.0, 'num_heads':2},
				{'type':'none', 'stride':2},
				{'type':'none', 'stride':2},
				{'type':'none', 'stride':2}
			]