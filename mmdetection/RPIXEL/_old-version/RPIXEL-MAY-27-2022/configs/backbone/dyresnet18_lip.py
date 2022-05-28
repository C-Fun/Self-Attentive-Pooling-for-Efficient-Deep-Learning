from RPIXEL.configs.backbone._base.dyresnet18 import _backbone, _convtype, conv1, pool1

_pooltype = "lip"
_poolparams = [
				{'type':'none', 'stride':1},
				{'type':_pooltype, 'ksize':2, 'stride':2},
				{'type':_pooltype, 'ksize':2, 'stride':2},
				{'type':_pooltype, 'ksize':2, 'stride':2}
			]
