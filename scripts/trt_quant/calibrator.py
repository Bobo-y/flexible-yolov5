# *** tensorrt  Calibrator ***
import os
import tensorrt as trt
import pycuda.driver as cuda
import ctypes
import logging
import pycuda.autoinit



logger = logging.getLogger(__name__)
ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_char_p
ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]


class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, dataloader, cache_file):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.cache_file = cache_file
        self.data = iter(dataloader)
        self.batch_size = dataloader.batch_size
        self.num_image = len(dataloader.dataset)
        self.current_index = 0
        self.nbytes = dataloader.dataset.nbytes
        self.device_input = cuda.mem_alloc(self.nbytes*self.batch_size)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, name):
        if self.current_index + self.batch_size > self.num_image:
            return None
        batch = next(self.data).numpy().astype("float32").ravel()
        cuda.memcpy_htod(self.device_input, batch)
        self.current_index += self.batch_size
        return [self.device_input]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


class MinMaxCalibrator(trt.IInt8MinMaxCalibrator):
    def __init__(self, dataloader, cache_file):
        trt.IInt8MinMaxCalibrator.__init__(self)
        self.cache_file = cache_file
        self.data = iter(dataloader)
        self.batch_size = dataloader.batch_size
        self.num_image = len(dataloader.dataset)
        self.current_index = 0
        self.nbytes = dataloader.dataset.nbytes
        self.device_input = cuda.mem_alloc(self.nbytes*self.batch_size)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, name):
        if self.current_index + self.batch_size > self.num_image:
            return None
        batch = next(self.data).numpy().astype("float32").ravel()
        cuda.memcpy_htod(self.device_input, batch)
        self.current_index += self.batch_size
        return [self.device_input]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
