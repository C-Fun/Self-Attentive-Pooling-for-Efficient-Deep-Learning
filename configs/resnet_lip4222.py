from ._base.resnet import _backbone, _convtype, conv1, pool1

_pooltype = "lip"
_poolparams = [
				{'type':_pooltype, 'ksize':4, 'stride':4},
				{'type':_pooltype, 'ksize':2, 'stride':2},
				{'type':_pooltype, 'ksize':2, 'stride':2},
				{'type':_pooltype, 'ksize':2, 'stride':2}
			]
