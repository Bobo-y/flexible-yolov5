#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""分割检的封装
"""

from functools import wraps

import numpy as np


def nms_test(bounding_boxes, confidence_score, threshold):
    picked_boxes = []
    picked_score = []
    picked_index = []

    if len(bounding_boxes) == 0:
        return picked_boxes, picked_score, picked_index

    # 边界框
    boxes = np.array(bounding_boxes)
    # 边界框坐标
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # 边界框的置信度得分
    score = np.array(confidence_score)

    # 计算边界框的区域
    areas = (x2 - x1) * (y2 - y1)

    # 按边界框的置信度分数排序
    order = np.argsort(score)
    while order.size > 0:
        # 选择置信度最高的比边界框
        index = order[-1]
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])
        picked_index.append(index)

        # 并集区域
        _x1 = np.maximum(x1[index], x1[order[:-1]])
        _y1 = np.maximum(y1[index], y1[order[:-1]])
        _x2 = np.minimum(x2[index], x2[order[:-1]])
        _y2 = np.minimum(y2[index], y2[order[:-1]])
        # 交集区域
        w = np.maximum(0.0, _x2 - _x1 + 1)
        h = np.maximum(0.0, _y2 - _y1 + 1)
        intersection = w * h
        # iou计算 交集/并集
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)
        left = np.where(ratio < threshold)
        order = order[left]
    return picked_boxes, picked_score, picked_index


class SplitDetector():
    def __init__(self, split_width_num=2, split_height_num=2):
        # self.inference_func = inference_func
        # self.func_args = func_args
        self.split_width_num = split_width_num
        self.split_height_num = split_height_num
        self.move_pads = []
        self.sub_images = []
        self.sub_sizes = []

    def split_image(self, image):
        split_width_num = self.split_width_num
        split_height_num = self.split_height_num

        original_height, original_width = image.shape[:2]
        split_height = int(original_height / split_height_num)
        split_width = int(original_width / split_width_num)

        for r in range(split_height_num):
            for c in range(split_width_num):
                top = max(0, int(split_height * r - split_height * 0.2))
                bottom = min(original_height, int(top + split_height * 1.4))
                left = max(0, int(split_width * c - split_width * 0.2))
                right = min(original_width, int(left + split_width * 1.4))
                sub_image = image[top:bottom, left:right]
                self.sub_images.append(sub_image)
                self.move_pads.append([left, top])
                self.sub_sizes.append([bottom - top, right - left])
        return self.sub_images, self.move_pads

    def add_movepad(self, data, move_pad):
        # current only accept dict format
        assert isinstance(data, dict), "data format must be dict like: {'labelid': bboxes}"

        for k in data.keys():
            if isinstance(data[k], list):
                for i, box in enumerate(data[k]):
                    data[k][i][0] += move_pad[0]
                    data[k][i][1] += move_pad[1]
                    data[k][i][2] += move_pad[0]
                    data[k][i][3] += move_pad[1]
            else:
                data[k][:, 0] = data[k][:, 0] + move_pad[0]
                data[k][:, 1] = data[k][:, 1] + move_pad[1]
                data[k][:, 2] = data[k][:, 2] + move_pad[0]
                data[k][:, 3] = data[k][:, 3] + move_pad[1]
        return data

    def filter_edge(self, data, size, pass_side=[]):
        height, width = size
        height_edge = int(height * 0.05)
        width_edge = int(width * 0.05)

        filtered_data = {}
        for k in data.keys():
            filtered_data[k] = []
            for bbox in data[k]:
                x1, y1, x2, y2 = bbox[:4]
                if 'left' not in pass_side and x1 < width_edge:
                    continue
                elif 'right' not in pass_side and x2 > width - width_edge:
                    continue
                elif 'top' not in pass_side and y1 < height_edge:
                    continue
                elif 'bottom' not in pass_side and y2 > height - height_edge:
                    continue
                else:
                    filtered_data[k].append(bbox)
        return filtered_data

    def merge_outputs(self, outputs, move_pads, conf_threshold=0.35, nms_threshold=0.3):
        """make sure x1, y1, x2, y2 were at top4 for every output line
        """
        merged_outputs = {}

        # merge
        merged_datas = {}
        for output in outputs:
            for k in output:
                if k not in merged_outputs:
                    merged_outputs[k] = []
                merged_outputs[k] += [output[k]]

        assert 'data' in merged_outputs
        merged_outputs['data'] = {}

        for index, (output, move_pad) in enumerate(zip(outputs, move_pads)):
            # get output data
            data = output['data']
            # filter edge objs
            pass_side = []
            if index < self.split_width_num:
                pass_side.append('top')
            if (index + self.split_width_num) >= self.split_width_num * self.split_height_num:
                pass_side.append('bottom')
            if index % self.split_height_num == 0:
                pass_side.append('left')
            if (index + 1) % self.split_height_num == 0:
                pass_side.append('right')
            data = self.filter_edge(data, self.sub_sizes[index], pass_side=pass_side)
            # move pixcel
            data = self.add_movepad(data, move_pad)
            for k in data.keys():
                if k not in merged_datas.keys():
                    merged_datas[k] = []
                merged_datas[k] += data[k]
        # nms
        for k in merged_datas.keys():
            temp_bboxes = [b[:4] for b in merged_datas[k]]
            temp_scores = [b[4] for b in merged_datas[k]]
            _temp_bboxes, _temp_scores, indices = nms_test(temp_bboxes, temp_scores, nms_threshold)
            merged_outputs['data'][k] = [bbox + [score] for i, (bbox, score) in
                                         enumerate(zip(_temp_bboxes, _temp_scores))]

        return merged_outputs

    def inference(self, image, inference_func, **kwargs):
        sub_images, move_pads = self.split_image(image, self.split_width_num, self.split_height_num)
        sub_outputs = []
        for index, sub_image in enumerate(sub_images):
            sub_output = inference_func(sub_image, kwargs)
            sub_outputs.append(sub_output)
        outputs = self.merge_outputs(sub_outputs, move_pads)
        return outputs


def SPLITINFERENCE(split_width=2, split_height=2):
    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # assert 'image' in kwargs, f'[ERROR] mast inclued image in args, current args:\n{args}\nkwargs:\n{kwargs}'
            assert 'image' in kwargs, '[ERROR] mast inclued image in args'

            image = kwargs.pop('image')

            spliter = SplitDetector(split_width, split_height)
            sub_images, move_pads = spliter.split_image(image)
            outputs = []
            for index, sub_image in enumerate(sub_images):
                sub_output = func(*args, image=sub_image, **kwargs)
                outputs.append(sub_output)
            merged_outputs = spliter.merge_outputs(outputs, move_pads)
            return merged_outputs

        return wrapper

    return decorate
