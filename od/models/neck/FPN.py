import torch.nn as nn
from od.models.modules.common import BottleneckCSP, Conv, Concat, C3
from utils.general import make_divisible


class PyramidFeatures(nn.Module):
    """
    this FPN  refer to yolov5, there are many different versions of implementation, and the details will be different

         concat
    C3 --->   P3
    |          ^
    V   concat | up2
    C4 --->   P4
    |          ^
    V          | up2
    C5 --->    P5
    """

    def __init__(self, C3_size=256, C4_size=512, C5_size=512, inner_p4=512, outer_p4=256, version='L'):
        super(PyramidFeatures, self).__init__()
        self.C3_size = C3_size
        self.C4_size = C4_size
        self.C5_size = C5_size

        self.version = version
        gains = {'s': {'gd': 0.33, 'gw': 0.5},
                 'm': {'gd': 0.67, 'gw': 0.75},
                 'l': {'gd': 1, 'gw': 1},
                 'x': {'gd': 1.33, 'gw': 1.25}}

        if self.version.lower() in gains:
            # only for yolov5
            self.gd = gains[self.version.lower()]['gd']  # depth gain
            self.gw = gains[self.version.lower()]['gw']  # width gain
        else:
            self.gd = 1
            self.gw = 1

        self.channels_out = {
            'inner_p4': inner_p4,
            'outer_p4': outer_p4,
        }
        self.re_channels_out()
        self.concat = Concat()

        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')

        self.P4_1 = C3(self.C5_size + self.C4_size, self.channels_out['inner_p4'], self.get_depth(3), False)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = Conv(self.channels_out['inner_p4'], self.channels_out['outer_p4'], 1, 1)
        self.out_shape = {'P3_size': self.C3_size + self.channels_out['outer_p4'],
                          'P4_size': self.channels_out['outer_p4'],
                          'P5_size': self.C5_size}
        print("FPN input channel size: C3 {}, C4 {}, C5 {}".format(self.C3_size, self.C4_size, self.C5_size))
        print("FPN output channel size: P3 {}, P4 {}, P5 {}".format(self.C3_size + self.channels_out['outer_p4'], self.channels_out['outer_p4'],
                                                                    self.C5_size))

    def get_depth(self, n):
        return max(round(n * self.gd), 1) if n > 1 else n

    def get_width(self, n):
        return make_divisible(n * self.gw, 8)

    def re_channels_out(self):
        for k, v in self.channels_out.items():
            self.channels_out[k] = self.get_width(v)

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
