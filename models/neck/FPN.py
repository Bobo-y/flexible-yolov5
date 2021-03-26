import torch.nn as nn
from models.modules.common import BottleneckCSP, Conv, Concat


class PyramidFeatures(nn.Module):
    """
    this FPN if refer to yolov5, there are many different versions of implementation, and the details will be different
    """

    def __init__(self, C3_size=256, C4_size=512, C5_size=512):
        super(PyramidFeatures, self).__init__()
        self.C3_size = C3_size
        self.C4_size = C4_size
        self.C5_size = C5_size
        self.concat = Concat()
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')

        self.P4_1 = BottleneckCSP(self.C5_size + self.C4_size, self.C4_size, 3, False)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = Conv(self.C4_size, self.C3_size, 1, 1)
        self.out_shape = {'P3_size': self.C3_size + self.C3_size,
                          'P4_size': self.C3_size,
                          'P5_size': self.C5_size}
        print("FPN input channel size: C3 {}, C4 {}, C5 {}".format(self.C3_size, self.C4_size, self.C5_size))
        print("FPN output channel size: P3 {}, P4 {}, P5 {}".format(self.C3_size + self.C3_size, self.C3_size,
                                                                    self.C5_size))

    def forward(self, inputs):
        C3, C4, C5 = inputs
        up5 = self.P5_upsampled(C5)
        concat1 = self.concat([up5, C4])
        p41 = self.P4_1(concat1)
        P4 = self.P4_2(p41)
        up4 = self.P4_upsampled(P4)
        P3 = self.concat([C3, up4])
        P5 = C5
        return P3, P4, P5
