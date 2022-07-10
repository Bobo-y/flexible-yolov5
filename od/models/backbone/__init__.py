# -*- coding: utf-8 -*-
from .resnet import resnet
from .shufflenetv2 import shufflenetv2
from .mobilenetv3 import MobileNetV3 as mobilenetv3
from .yolov5 import YOLOv5
from .efficientnet import efficientnet
from .hrnet import hrnet
from .swin_transformer import swin_transformer as swin
from .vgg import vgg
from .repvgg import repvgg
from .gnn import gnn

__all__ = ['build_backbone']

support_backbone = ['resnet', 'shufflenetv2', 'mobilenetv3', 'YOLOv5', 'efficientnet', 'hrnet', 'swin', 'vgg', 'repvgg', 'gnn']


def build_backbone(backbone_name, **kwargs):
    assert backbone_name in support_backbone, f'all support backbone is {support_backbone}'
    backbone = eval(backbone_name)(**kwargs)
    return backbone
