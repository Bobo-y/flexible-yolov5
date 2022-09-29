# -*- coding: utf-8 -*-
from addict import Dict
from torch import nn
import math
import yaml
import torch
from od.models.modules.common import Conv
from od.models.backbone import build_backbone
from od.models.neck import build_neck
from od.models.head import build_head
from utils.torch_utils import initialize_weights, fuse_conv_and_bn, model_info


class Model(nn.Module):
    def __init__(self, model_config):
        """
        :param model_config:
        """

        super(Model, self).__init__()
        if type(model_config) is str:
            model_config = yaml.load(open(model_config, 'r'), Loader=yaml.SafeLoader)
        model_config = Dict(model_config)
        backbone_type = model_config.backbone.pop('type')
        self.backbone = build_backbone(backbone_type, **model_config.backbone)
        ch_in = self.backbone.out_shape
        
        self.necks = nn.ModuleList()
        necks_config = model_config.neck
        for neck_name, neck_params in necks_config.items():
            neck_params['ch'] = ch_in
            neck = build_neck(neck_name, **neck_params)
            ch_in = neck.out_shape
            self.necks.append(neck)
            
        model_config.head['ch'] = ch_in
        self.detection = build_head('YOLOHead', **model_config.head)
        self.stride = self.detection.stride
        self._initialize_biases()

        initialize_weights(self)

    def _initialize_biases(self, cf=None):
        # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        m = self.detection  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ')
        for module in [self.backbone, self.fpn, self.pan, self.detection]:
            for m in module.modules():
                if type(m) is Conv and hasattr(m, 'bn'):
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                    delattr(m, 'bn')  # remove batchnorm
                    m.forward = m.fuseforward  # update forward
        self.info()
        return self

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

    def forward(self, x):
        out = self.backbone(x)
        for neck in self.necks:
            out = neck(out)
        y = self.detection(list(out))
        return y
    
    def to(self, device):
        self.backbone = self.backbone.to(device)
        for idx in range(len(self.necks)):
            self.necks[idx] = self.necks[idx].to(device)
        self.detection = self.detection.to(device)
        self.detection.stride = self.detection.stride.to(device)
        for idx in range(len(self.detection.grid)):
            self.detection.grid[idx] = self.detection.grid[idx].to(device)
        for idx in range(len(self.detection.anchor_grid)):
            self.detection.anchor_grid[idx] = self.detection.anchor_grid[idx].to(device)
        return self

if __name__ == '__main__':
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    device = torch.device('cpu')
    x = torch.zeros(1, 3, 640, 640).to(device)

    model = Model(model_config='../../configs/model_gnn.yaml').to(device)
    # model.fuse()
    import time

    tic = time.time()
    y = model(x)
    for item in y:
        print(item.shape)
