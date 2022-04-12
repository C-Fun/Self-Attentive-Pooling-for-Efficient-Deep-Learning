from ._base.resnet import _backbone, _convtype, conv1, pool1

_pooltype = "nlp"
pool1 = [_pooltype, 3, 2, 1]
_poolparams = [
				{'type':'none', 'stride':1},
				{'type':_pooltype, 'ksize':2, 'stride':2, 'psize':1, 'dim_reduced_ratio':1.0, 'num_heads':2},
				{'type':_pooltype, 'ksize':2, 'stride':2, 'psize':1, 'dim_reduced_ratio':1.0, 'num_heads':4},
				{'type':_pooltype, 'ksize':2, 'stride':2, 'psize':1, 'dim_reduced_ratio':1.0, 'num_heads':8}
			]
