from RPIXEL.configs.backbone._base.resnet50 import _backbone, _convtype, conv1, pool1

_pooltype = "none"
_poolparams = [
				{'type':'none', 'stride':2},
				{'type':'none', 'stride':2},
				{'type':'none', 'stride':2},
				{'type':'none', 'stride':2},
			]
