import torch.nn as nn
from models.modules.common import BottleneckCSP, Conv, Concat


class PAN(nn.Module):
    """
        This PAN if refer to yolov5, there are many different versions of implementation, and the details will be different
        """

    def __init__(self, P3_size=512, P4_size=256, P5_size=512, inner_p3=256, inner_p4=512, inner_p5=1024):
        super(PAN, self).__init__()
        self.P3_size = P3_size
        self.P4_size = P4_size
        self.P5_size = P5_size
        self.inner_p3 = inner_p3
        self.inner_p4 = inner_p4
        self.inner_p5 = inner_p5
        self.P3 = BottleneckCSP(self.P3_size, self.inner_p3, 3, False)
        self.convP3 = Conv(self.inner_p3, self.inner_p3, 3, 2)
        self.P4 = BottleneckCSP(self.P4_size + self.inner_p3, self.inner_p4, 3, False)
        self.convP4 = Conv(self.inner_p4, self.inner_p4, 3, 2)
        self.P5 = BottleneckCSP(self.inner_p4 + P5_size, self.inner_p5, 3, False)
        self.concat = Concat()
        self.out_shape = (self.inner_p3, self.inner_p4, self.inner_p5)
        print("PAN input channel size: P3 {}, P4 {}, P5 {}".format(self.P3_size, self.P4_size, self.P5_size))
        print("PAN output channel size: PP3 {}, PP4 {}, PP5 {}".format(self.inner_p3, self.inner_p4, self.inner_p5))

    def forward(self, inputs):
        P3, P4, P5 = inputs
        PP3 = self.P3(P3)
        convp3 = self.convP3(PP3)
        concat3_4 = self.concat([convp3, P4])
        PP4 = self.P4(concat3_4)
        convp4 = self.convP4(PP4)
        concat4_5 = self.concat([convp4, P5])
        PP5 = self.P5(concat4_5)
        return PP3, PP4, PP5
