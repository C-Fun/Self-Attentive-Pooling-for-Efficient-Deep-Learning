from ._base.resnet import _backbone, _convtype, conv1, pool1

_pooltype = "mixp"
_poolparams = [
				{'type':'none', 'stride':1},
				{'type':_pooltype, 'ksize':2, 'stride':2, 'psize':2, 'dim_reduced_ratio':0.25, 'num_heads':2},
				{'type':_pooltype, 'ksize':2, 'stride':2, 'psize':2, 'dim_reduced_ratio':0.25, 'num_heads':2},
				{'type':_pooltype, 'ksize':2, 'stride':2, 'psize':2, 'dim_reduced_ratio':0.25, 'num_heads':2}
			]
