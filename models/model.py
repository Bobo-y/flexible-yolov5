# -*- coding: utf-8 -*-
from addict import Dict
from torch import nn
import math
import yaml
import torch
from models.modules.common import Conv
from models.backbone import build_backbone
from models.neck import build_neck
from models.head import build_head
from utils.autoanchor import check_anchor_order
from utils.torch_utils import initialize_weights, fuse_conv_and_bn


class Model(nn.Module):
    def __init__(self, model_config):
        """
        :param model_config:
        """
        super(Model, self).__init__()
        if type(model_config) is str:
            model_config = yaml.load(open(model_config, 'r'))
        model_config = Dict(model_config)
        backbone_type = model_config.backbone.pop('type')
        self.backbone = build_backbone(backbone_type, **model_config.backbone)
        backbone_out = self.backbone.out_shape
        self.fpn = build_neck('FPN', **backbone_out)
        fpn_out = self.fpn.out_shape
        self.pan = build_neck('PAN', **fpn_out)
        pan_out = self.pan.out_shape
        model_config.head['ch'] = pan_out
        self.detection = build_head('YOLOHead', **model_config.head)
        self.detection.stride = torch.tensor([8., 16., 32.])
        self.detection.anchors /= self.detection.stride.view(-1, 1, 1)

        check_anchor_order(self.detection)
        self.stride = self.detection.stride
        self._initialize_biases()  # only run once

        initialize_weights(self)

    def _initialize_biases(self, cf=None):
        # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        m = self.detection  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, x):
        out = self.backbone(x)
        out = self.fpn(out)
        out = self.pan(out)
        y = self.head(list(out))
        return y


if __name__ == '__main__':

    device = torch.device('cuda')
    x = torch.zeros(2, 3, 640, 640).to(device)

    model_config = {
        'backbone': {'type': 'shufflenet_v2_x1_0'},
        'head': {'nc': 2},
    }

    model = Model(model_config='../configs/model.yaml').to(device)
    import time

    tic = time.time()
    y = model(x)
    for item in y:
        print(item.shape)