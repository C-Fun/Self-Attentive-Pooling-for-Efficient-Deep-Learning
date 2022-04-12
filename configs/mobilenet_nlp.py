from ._base.mobilenet import _backbone, _convparams, _convtype, conv1

_pooltype = "nlp"
_poolparams = [
 				{'type':_pooltype, 'ksize':2, 'stride':2, 'psize':1, 'dim_reduced_ratio':1.0, 'num_heads':2},
				{'type':'none', 'stride':1},
				{'type':_pooltype, 'ksize':2, 'stride':2, 'psize':1, 'dim_reduced_ratio':1.0, 'num_heads':2},
				{'type':'none', 'stride':1}
			]