import os
import platform
import sys
sys.path.append('.')
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import od.models as models
import yaml
from torch.utils.mobile_optimizer import optimize_for_mobile

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from od.models.modules.experimental import attempt_load
from od.models.head.yolo import YOLOHead
from od.models.modules.activations import Hardswish, SiLU
from od.data.datasets import LoadImages
from utils.general import (LOGGER, check_dataset, check_img_size, check_requirements, check_version, colorstr,
                           file_size, print_args, url2file)
from utils.torch_utils import select_device
from utils.qat_util import *
try:
    from pytorch_quantization import quant_modules
    from pytorch_quantization.tensor_quant import QuantDescriptor
    from pytorch_quantization import calib
    from pytorch_quantization import nn as quant_nn
except:
    print('qat onnx export not support')


def export_onnx(model, im, file, opset=13, train=False, dynamic=False, simplify=True, prefix=colorstr('ONNX:')):
    # YOLOv5 ONNX qat export
    try:
        import onnx

        LOGGER.info(f'\n{prefix} starting export with onnx {onnx.__version__}...')
        f = file.with_suffix('.onnx')

        torch.onnx.export(
            model.cpu() if dynamic else model,  # --dynamic only compatible with cpu
            im.cpu() if dynamic else im,
            f,
            verbose=False,
            opset_version=opset,
            training=torch.onnx.TrainingMode.TRAINING if train else torch.onnx.TrainingMode.EVAL,
            do_constant_folding=not train,
            input_names=['images'],
            output_names=['output'],
            dynamic_axes={
                'images': {
                    0: 'batch',
                    2: 'height',
                    3: 'width'},  # shape(1,3,640,640)
                'output': {
                    0: 'batch',
                    1: 'anchors'}  # shape(1,25200,85)
            } if dynamic else None)

        # Checks
        model_onnx = onnx.load(f)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

        # Metadata
        d = {'stride': int(max(model.stride)), 'names': model.names}
        for k, v in d.items():
            meta = model_onnx.metadata_props.add()
            meta.key, meta.value = k, str(v)
        onnx.save(model_onnx, f)

        # Simplify
        if simplify:
            try:
                check_requirements(('onnx-simplifier',))
                import onnxsim
                LOGGER.info(f'{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...')
                model_onnx, check = onnxsim.simplify(model_onnx,
                                                     dynamic_input_shape=dynamic,
                                                     input_shapes={'images': list(im.shape)} if dynamic else None)
                assert check, 'assert check failed'
                onnx.save(model_onnx, f)
            except Exception as e:
                LOGGER.info(f'{prefix} simplifier failure: {e}')
        LOGGER.info(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        return f
    except Exception as e:
        LOGGER.info(f'{prefix} export failure: {e}')



if __name__ == "__main__":
    pt_path = ''
    model = torch.load(pt_path)['model']
    qat_initialize()
    replace_to_quantization_module(model, None)
    quant_nn.TensorQuantizer.use_fb_fake_quant = True
    im = torch.zeros(1, 3, 640, 640).to('cpu')
    model = model.to('cpu')
    for k, m in model.named_modules():
        if isinstance(m, YOLOHead):
            m.inplace = True
            m.onnx_dynamic = False
            m.export = True
    export_onnx(model, im, 'qat_model.onnx')
