from ._base.mobilenet import _backbone, _convparams, _convtype, conv1

_pooltype = "lip"
_poolparams = [
 				{'type':_pooltype, 'ksize':2, 'stride':2},
				{'type':'none', 'stride':1},
				{'type':_pooltype, 'ksize':2, 'stride':2},
				{'type':'none', 'stride':1}
			]