from mmcv.ops.deform_conv import DeformConv2dFunction, DeformConv2dPack

# MMCV Implementation
deform_conv2d = DeformConv2dFunction.apply
class DeformConv2d(DeformConv2dPack):
    def __init__(self, *args, **kwargs):
        super(DeformConv2d, self).__init__(*args, **kwargs)

    def forward(self, x):
        offset = self.conv_offset(x)
        out =  deform_conv2d(x, offset, self.weight, self.stride, self.padding,
                             self.dilation, self.groups, self.deform_groups,
                             False, self.im2col_step)
        return (out, offset)