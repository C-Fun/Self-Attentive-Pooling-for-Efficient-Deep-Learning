_backbone = "mobilenet"
_convparams = [
                # t(expand_ratio), channel, num, stride (pool_stride)
                # [1,  16, 1, 1],
                # [6,  24, 2, 2],
                # [6,  32, 3, 2],
                [6,  64, 4, 2],
                [6,  96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]
_convtype = "normal"
_poolparams = None

conv1 = ["normal", 3, 2, 1] # ksize 3, stride 2, padding 1
# pool1 = None
