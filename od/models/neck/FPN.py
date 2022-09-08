import torch.nn as nn
from od.models.modules.common import Conv, Concat, C3
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

    def __init__(self, ch=[256, 512, 1024], channel_outs=[512, 256], version='s'):
        super(PyramidFeatures, self).__init__()

        self.C3_size = ch[0]
        self.C4_size = ch[1]
        self.C5_size = ch[2]
        self.channels_outs = channel_outs
        self.version = version
        gains = {
                'n': {'gd': 0.33, 'gw': 0.25},
                's': {'gd': 0.33, 'gw': 0.5},
                'm': {'gd': 0.67, 'gw': 0.75},
                'l': {'gd': 1, 'gw': 1},
                'x': {'gd': 1.33, 'gw': 1.25}
                }

        if self.version.lower() in gains:
            # only for yolov5
            self.gd = gains[self.version.lower()]['gd']  # depth gain
            self.gw = gains[self.version.lower()]['gw']  # width gain
        else:
            self.gd = 0.33
            self.gw = 0.5

        self.re_channels_out()
        self.concat = Concat()

        self.P5 = Conv(self.C5_size, self.channels_outs[0], 1, 1)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = C3(self.channels_outs[0] + self.C4_size, self.channels_outs[0], self.get_depth(3), False)

        self.P4 = Conv(self.channels_outs[0], self.channels_outs[1], 1, 1)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')

        self.P3 = C3(self.channels_outs[1] + self.C3_size, self.channels_outs[1], self.get_depth(3), False)


        self.out_shape = [ self.channels_outs[1], self.channels_outs[1], self.channels_outs[0]]
        print("FPN input channel size: C3 {}, C4 {}, C5 {}".format(self.C3_size, self.C4_size, self.C5_size))
        print("FPN output channel size: P3 {}, P4 {}, P5 {}".format(self.channels_outs[1], self.channels_outs[1],
                                                                    self.channels_outs[0]))


    def get_depth(self, n):
        return max(round(n * self.gd), 1) if n > 1 else n

    def get_width(self, n):
        return make_divisible(n * self.gw, 8)

    def re_channels_out(self):
        for idx, channel_out in enumerate(self.channels_outs):
            self.channels_outs[idx] = self.get_width(channel_out)

    def forward(self, inputs):
        C3, C4, C5 = inputs
    
        P5 = self.P5(C5) 
        up5 = self.P5_upsampled(P5)
        concat1 = self.concat([up5, C4])
        conv1 = self.conv1(concat1)

        P4 = self.P4(conv1)
        up4 = self.P4_upsampled(P4)
        concat2 = self.concat([C3, up4])
        
        PP3 = self.P3(concat2)

        return PP3, P4, P5
