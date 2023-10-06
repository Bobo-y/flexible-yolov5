import os
from typing import List, Callable, Union, Dict
import torch
from pytorch_quantization import quant_modules
from pytorch_quantization import nn as quant_nn
from pytorch_quantization.tensor_quant import QuantDescriptor
from absl import logging as quant_logging
from pytorch_quantization.nn.modules import _utils as quant_nn_utils
from pytorch_quantization import calib
import copy
import re
from tqdm import tqdm
import torch.distributed as dist


class disable_quantization:
    def __init__(self, model):
        self.model = model

    def apply(self, disabled=True):
        for name, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                module._disabled = disabled

    def __enter__(self):
        self.apply(True)

    def __exit__(self, *args, **kwargs):
        self.apply(False)


class enable_quantization:
    def __init__(self, model):
        self.model = model

    def apply(self, enabled=True):
        for name, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                module._disabled = not enabled

    def __enter__(self):
        self.apply(True)
        return self

    def __exit__(self, *args, **kwargs):
        self.apply(False)


def have_quantizer(module):
    for name, module in module.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            return True


# Initialize PyTorch Quantization
def qat_initialize():
    quant_desc_input = QuantDescriptor(calib_method="histogram")
    quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantMaxPool2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantConvTranspose2d.set_default_quant_desc_input(
        quant_desc_input)
    quant_nn.QuantAvgPool2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
    quant_logging.set_verbosity(quant_logging.ERROR)


def transfer_torch_to_quantization(nninstance: torch.nn.Module, quantmodule):
    quant_instance = quantmodule.__new__(quantmodule)
    for k, val in vars(nninstance).items():
        setattr(quant_instance, k, val)

    def __init__(self):
        if not isinstance(self, (quant_nn.QuantMaxPool2d, quant_nn.QuantAdaptiveAvgPool2d)):
            quant_desc_input, quant_desc_weight = quant_nn_utils.pop_quant_desc_in_kwargs(
                self.__class__)
        else:
            quant_desc_input = quant_nn_utils.pop_quant_desc_in_kwargs(
                self.__class__, True)
        if isinstance(self, quant_nn_utils.QuantInputMixin):
            self.init_quantizer(quant_desc_input)

            # Turn on torch_hist to enable higher calibration speeds
            if isinstance(self._input_quantizer._calibrator, calib.HistogramCalibrator):
                self._input_quantizer._calibrator._torch_hist = True
        else:
            self.init_quantizer(quant_desc_input, quant_desc_weight)

            # Turn on torch_hist to enable higher calibration speeds
            if isinstance(self._input_quantizer._calibrator, calib.HistogramCalibrator):
                self._input_quantizer._calibrator._torch_hist = True
                self._weight_quantizer._calibrator._torch_hist = True

    __init__(quant_instance)
    return quant_instance


def quantization_ignore_match(ignore_policy: Union[str, List[str], Callable], path: str) -> bool:

    if ignore_policy is None:
        return False
    if isinstance(ignore_policy, Callable):
        return ignore_policy(path)

    if isinstance(ignore_policy, str) or isinstance(ignore_policy, List):

        if isinstance(ignore_policy, str):
            ignore_policy = [ignore_policy]

        if path in ignore_policy:
            return True
        for item in ignore_policy:
            if re.match(item, path):
                return True
    return False


def replace_to_quantization_module(model: torch.nn.Module, ignore_policy: Union[str, List[str], Callable] = None):

    module_dict = {}
    for entry in quant_modules._DEFAULT_QUANT_MAP:
        module = getattr(entry.orig_mod, entry.mod_name)
        module_dict[id(module)] = entry.replace_mod

    def recursive_and_replace_module(module, prefix=""):
        for name in module._modules:
            submodule = module._modules[name]
            path = name if prefix == "" else prefix + "." + name
            recursive_and_replace_module(submodule, path)

            submodule_id = id(type(submodule))
            if submodule_id in module_dict:
                ignored = quantization_ignore_match(ignore_policy, path)
                if ignored:
                    continue
                module._modules[name] = transfer_torch_to_quantization(
                    submodule, module_dict[submodule_id])

    recursive_and_replace_module(model)


def get_attr_with_path(m, path):
    def sub_attr(m, names):
        name = names[0]
        value = getattr(m, name)
        if len(names) == 1:
            return value
        return sub_attr(value, names[1:])
    return sub_attr(m, path.split("."))


def apply_custom_rules_to_quantizer(model, pairs):
    for major, sub in pairs:
        get_attr_with_path(model, sub)._input_quantizer = get_attr_with_path(
            model, major)._input_quantizer


class QAT():
    def __init__(self, model,train_dataloader, calibrate_step=128, ignore_policy=None, qat_graph_optim=None):
        self.model = model
        self.train_dataloader = train_dataloader
        self.ignore_policy = ignore_policy
        self.qat_graph_optim = qat_graph_optim
        self.calibrate_step = calibrate_step

    def qat(self):
        qat_initialize()
        replace_to_quantization_module(self.model, self.ignore_policy)
        if self.qat_graph_optim:
            apply_custom_rules_to_quantizer(self.model, self.qat_graph_optim)
        for name, module in self.model.named_modules():
            try:
                if isinstance(module, quant_nn.TensorQuantizer):
                    cal = getattr(module, "_calibrator")
                    if cal:
                        cal._torch_hist = True
            except:
                print(f"Error in {name}")
                continue

        self.calibrate_model(self.model, self.calibrate_step)

        if int(os.environ.get("WORLD_SIZE")) > 1:
            dist.barrier()
            with torch.no_grad():
                for name, module in self.model.named_modules():
                    if hasattr(module, "_input_quantizer"):
                        if not hasattr(module._input_quantizer, "_amax"):
                            module._input_quantizer.register_buffer(
                                "_amax", torch.tensor(-1.0, dtype=torch.float32, device=torch.cuda.current_device()))
                    if hasattr(module, "_weight_quantizer"):
                        if not hasattr(module._weight_quantizer, "_amax"):
                            module._weight_quantizer.register_buffer("_amax", -1*torch.ones(
                                module.weight.shape[0], dtype=torch.float32, device=torch.cuda.current_device()))
                dist.barrier()
                for name, module in self.model.named_modules():
                    if isinstance(module, quant_nn.TensorQuantizer):
                        dist.all_reduce(module._amax, op=dist.ReduceOp.MAX)
                dist.barrier()
                for name, module in self.model.named_modules():
                    if isinstance(module, quant_nn.TensorQuantizer):
                        if module._amax.max().item() < 0:
                            delattr(module, '_amax')

    def calibrate_model(self, model, num_batch=25):

        def compute_amax(model, device, **kwargs):
            for name, module in model.named_modules():
                try:
                    if isinstance(module, quant_nn.TensorQuantizer):
                        if module._calibrator is not None:
                            if isinstance(module._calibrator, calib.MaxCalibrator):
                                module.load_calib_amax()
                            else:
                                module.load_calib_amax(**kwargs)

                            module._amax = module._amax.to(device)
                except:
                    continue

        def collect_stats(model, device, num_batch=512):
            """Feed data to the network and collect statistics"""
            # Enable calibrators
            model.eval()
            for name, module in model.named_modules():
                if isinstance(module, quant_nn.TensorQuantizer):
                    if module._calibrator is not None:
                        module.disable_quant()
                        module.enable_calib()
                    else:
                        module.disable()
            with torch.no_grad():
                for i in tqdm(range(num_batch), total=num_batch, position=dist.get_rank()):
                    for i, (imgs, targets, paths, _) in enumerate(self.train_dataloader):
                        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) and i < len(self.train_dataloader) - 1:
                            with torch.no_grad():
                                self.model(imgs)
                        else:
                            self.model(imgs)
                    if i >= num_batch:
                        break

            # Disable calibrators
            for name, module in model.named_modules():
                if isinstance(module, quant_nn.TensorQuantizer):
                    if module._calibrator is not None:
                        module.enable_quant()
                        module.disable_calib()
                    else:
                        module.enable()
        device = torch.cuda.current_device()
        collect_stats(model, device, num_batch=num_batch)
        compute_amax(model, device, method="mse")