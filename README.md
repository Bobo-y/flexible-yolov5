## flexible-yolov5

Based on [ultralytics/yolov5](https://github.com/ultralytics/yolov5).

The original Yolo V5 was an amazing project. For professionals, it should not be difficult to understand and modify its code. I'm not an expert. When I want to make some changes to the network, it's not so easy, such as adding branches and trying other backbones. Maybe there are people like me, so I split the yolov5 model to {backbone, neck, head} to facilitate the operation of various modules and support more backbones.Basically, I only changed the model, and I didn't change the architecture, training and testing of yolov5. Therefore, if the original code is updated, it is also very convenient to update this code.
## Table of contents
* [Features](#features)
* [Prerequisites](#prerequisites)
* [Getting Started](#getting-started)
    * [Dataset Preparation](#dataset-preparation)
    * [Training and Testing](#Training-and-Testing)
    * [Detection](#Detection)
* [Reference](#Reference)


## Features
- Reorganize model structure, such as backbone, neck, head, can modify the network flexibly and conveniently
- More backbone, mobilenetV3, shufflenetV2, resnet18, 50, 101 and so on


## Prerequisites

please refer requirements.txt

## Getting Started

### Dataset Preparation

Make data for yolov5 format. you can use utils/make_yolov5_data.py convert VOC data to yolov5 data format.

### Training and Testing

For training and Testing, it's same like yolov5.

### Training

1. check out configs/data.yaml, and replace with your data and nc
2. check out configs/model.yaml, choose backbone. please refer support_backbone in models.backbone.__init__.py
3. 
```shell script
$ python train.py  --batch 16 --epochs 5 --data configs/data.yaml --cfg confgis/model.yaml
```

### Testing and Visualize
Same as [ultralytics/yolov5](https://github.com/ultralytics/yolov5)


### Detection

see detector.py

## Some results

I train yolo with backbone of  MobileNetV3, resnet50, shufflenet_v2_x1_0 on my dataset for person detection(27K images).

*For time reason, for each backbone, i only train 15 epochs, Here's the test comparison,*

resnet50:
![](images/resnet50.jpg)

MobileNetV3:

![](images/moblienetv3.jpg)
shufflenet_v2_x1_0:
![](images/shufflenet.jpg)


## Reference

* [ultralytics/yolov5](https://github.com/ultralytics/yolov5)
