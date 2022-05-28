_backbone = "resnet50"
_convtype = "dynamic"
_pooltype = "none"
_poolparams = None

conv1 = ["dynamic", 7, 2, 3] # ksize 7, stride 2, padding 3
pool1 = ["maxpool", 3, 2, 1] # ksize 3, stride 2, padding 1