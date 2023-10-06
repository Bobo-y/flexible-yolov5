import argparse
from dataclasses import dataclass
import cv2
import os
from glob import glob
import tensorrt as trt
from data import Dataset
from torch.utils.data import DataLoader 
from calibrator import EntropyCalibrator, MinMaxCalibrator
import ctypes


EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
EXPLICIT_PRECISION = 1 << (int)(
    trt.NetworkDefinitionCreationFlag.EXPLICIT_PRECISION)


def GiB(val):
    return val * 1 << 30

def MiB(val):
    return val * 1 << 20

@dataclass
class BuildConfig:
    min_timing_iterations: int = None
    avg_timing_iterations: int = None
    int8_calibrator: trt.IInt8Calibrator = None
    max_workspace_size: int = MiB(1024)
    flags: int = None
    profile_stream: int = None
    num_optimization_profiles: int = None
    default_device_type: trt.DeviceType = trt.DeviceType.GPU
    DLA_core: int = None
    profiling_verbosity: int = None
    engine_capability: int = None


class Transform():
    def __init__(self, h=640, w=640):
        self.h = h
        self.w = w

    def __call__(self, img):
        img = img.astype("float32")
        img = (img - 128.0)/128.0
        img = cv2.resize(img, (self.w, self.h))
        img = img.transpose(2, 0, 1)
        return img

def build_int8_engine(trt_logger, onnx_path, build_params={}):
    builder = trt.Builder(trt_logger)
    network = builder.create_network(EXPLICIT_BATCH)
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, trt_logger)
    build_config = build_params.get("build_config", None)
    if build_config:
        for key, val in build_config.__dict__.items():
            if val is not None:
                setattr(config, key, val)
    with open(onnx_path, 'rb') as f:
        parser.parse(f.read())
        for index in range(parser.num_errors):
            print(parser.get_error(index))
    if builder.platform_has_tf32:
        config.clear_flag(trt.BuilderFlag.TF32)
    engine = builder.build_serialized_network(network, config)
    return engine


if __name__ == '__main__':
    def parser_arg():
        parser = argparse.ArgumentParser(
            description="calibrate int8 model and generate model")
        parser.add_argument("--onnx", type=str, required=True)
        parser.add_argument("--images-dir", type=str, required=True)
        parser.add_argument("--save-engine", type=str, required=True)
        parser.add_argument('--verbose', action="store_true",
                            default=False, required=False)
        parser.add_argument("--w", type=int, default=640)
        parser.add_argument('--h', type=int, default=640)
        parser.add_argument('--batch-size', type=int, default=32)
        parser.add_argument('--images-num', type=int, default=1000)
        parser.add_argument('--calibrator', type=str, default='kl', help='kl or minmax')
        parser.add_argument('--plugin-dir', type=str, default=None, required=False, help='plugin dir')
        parser.add_argument('--cache-file', type=str, default='sample.cache', required=False)
        args = parser.parse_args()
        return args

    args = parser_arg()
    samples_imgs = args.images_dir
    onnx_model = args.onnx
    save_engine = args.save_engine
    h = args.h
    w = args.w
    bs = args.batch_size
    num = args.images_num
    calibrator = args.calibrator
    plugin_dir = args.plugin_dir
    cache_file = args.cache_file

    if plugin_dir is not None:
        paths = glob(os.path.join(plugin_dir, "*.so"))
        for path in paths:
            ctypes.cdll.LoadLibrary(path)

    build_config = BuildConfig()
    build_config.flags = 1 << int(trt.BuilderFlag.INT8)
    build_config.max_workspace_size = MiB(2048)

    transform = Transform(h=h, w=w)
    dataset = Dataset(samples_imgs, num=num, transform=transform)
    dataloader = DataLoader(dataset, batch_size=bs, num_workers=8, drop_last=True, shuffle=True, prefetch_factor=2)
    if calibrator == 'kl':
        calibr = EntropyCalibrator(dataloader=dataloader, cache_file=cache_file)
    elif calibrator == 'minmax':
        calibr = MinMaxCalibrator(dataloader=dataloader, cache_file=cache_file)
    else:
        assert False, "not support calibrator"

    build_config.int8_calibrator = calibr
    if args.verbose is True:
        logger = trt.Logger(trt.Logger.VERBOSE)
    else:
        logger = trt.Logger(trt.Logger.INFO)
    build_params = {"build_config": build_config}
    engine = build_int8_engine(
        logger, onnx_path=onnx_model, build_params=build_params)
    with open(save_engine, 'wb') as f:
        f.write(engine)
