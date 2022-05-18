from ._base.resnet import _backbone, _convtype, conv1, pool1

_pooltype = "gaussianp"
_poolparams = [
				{'type':'none', 'stride':1},
				{'type':_pooltype, 'ksize':2, 'stride':2},
				{'type':_pooltype, 'ksize':2, 'stride':2},
				{'type':_pooltype, 'ksize':2, 'stride':2}
			]