# -*- coding: utf-8 -*-
from .resnet import *
from .shufflenetv2 import *
from .MobilenetV3 import MobileNetV3
from .csp_yolo import YOLOv5
from .efficientnet import *

__all__ = ['build_backbone']

support_backbone = ['resnet18', 'resnet50', 'resnet34', 'resnet101', 'resnet152',
                    'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0',
                    'MobileNetV3', 'YOLOv5', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3',
                    'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7', 'efficientnet_b8',
                    'efficientnet_l2']


def build_backbone(backbone_name, **kwargs):
    assert backbone_name in support_backbone, f'all support backbone is {support_backbone}'
    backbone = eval(backbone_name)(**kwargs)
    return backbone
