import torch.nn as nn
from od.models.modules.common import Conv, C3, SPPF, C3TR
from utils.general import make_divisible


class YOLOv5(nn.Module):
    def __init__(self, version='S', with_C3TR=False):
        super(YOLOv5, self).__init__()
        self.version = version
        self.with_c3tr = with_C3TR
        gains = {
                'n': {'gd': 0.33, 'gw': 0.25},
                's': {'gd': 0.33, 'gw': 0.5},
                'm': {'gd': 0.67, 'gw': 0.75},
                'l': {'gd': 1, 'gw': 1},
                'x': {'gd': 1.33, 'gw': 1.25}}

        self.gd = gains[self.version.lower()]['gd']  # depth gain
        self.gw = gains[self.version.lower()]['gw']  # width gain

        self.channels_out = [64, 128, 128, 256, 256, 512, 512, 1024, 1024, 1024]

        self.re_channels_out()

        self.C1 = Conv(3, self.channels_out[0], 6, 2, 2)

        self.C2  = Conv(self.channels_out[0], self.channels_out[1], k=3, s=2)
        self.conv1 = C3(self.channels_out[1], self.channels_out[2], self.get_depth(3))

        self.C3 = Conv(self.channels_out[2], self.channels_out[3], 3, 2)
        self.conv2 = C3(self.channels_out[3], self.channels_out[4], self.get_depth(6))

        self.C4 = Conv(self.channels_out[4], self.channels_out[5], 3, 2)
        self.conv3 = C3(self.channels_out[5], self.channels_out[6], self.get_depth(9))

        self.C5 = Conv(self.channels_out[6], self.channels_out[7], 3, 2)

        if self.with_c3tr:
            self.conv4 = C3TR(self.channels_out[7], self.channels_out[8], self.get_depth(3))
        else:
            self.conv4 = C3(self.channels_out[7], self.channels_out[8], self.get_depth(3))

        self.sppf = SPPF(self.channels_out[8], self.channels_out[9], 5)
        
        self.out_shape = {'C3_size': self.channels_out[3],
                          'C4_size': self.channels_out[5],
                          'C5_size': self.channels_out[9]}

        print("backbone output channel: C3 {}, C4 {}, C5(SPPF) {}".format(self.channels_out[3],
                                                                    self.channels_out[5],
                                                                    self.channels_out[9]))

    def forward(self, x):
        c1 = self.C1(x)

        c2 = self.C2(c1)
        conv1 = self.conv1(c2)

        c3 = self.C3(conv1)
        conv2 = self.conv2(c3)

        c4 = self.C4(conv2)
        conv3 = self.conv3(c4)

        c5 = self.C5(conv3)
        conv4 = self.conv4(c5)

        sppf = self.sppf(conv4)

        return c3, c4, sppf

    def get_depth(self, n):
        return max(round(n * self.gd), 1) if n > 1 else n

    def get_width(self, n):
        return make_divisible(n * self.gw, 8)

    def re_channels_out(self):
        for idx, channel_out in enumerate(self.channels_out):
            self.channels_out[idx] = self.get_width(channel_out)