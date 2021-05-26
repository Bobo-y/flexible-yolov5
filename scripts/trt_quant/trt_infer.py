"""
An example that uses TensorRT's Python api to make inferences.
"""
import time
import cv2
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
import torch
import torchvision


class YoLov5TRT():
    """
    description: A YOLOv5 class that warps TensorRT ops, preprocess and postprocess ops.
    """

    def __init__(self, engine_file_path):
        # Create a Context on this device,
        self.ctx = cuda.Device(0).make_context()
        self.stride = torch.tensor([8., 16., 32.])
        self.grid = [torch.zeros(1)] * 3
        self.anchor_grid = torch.tensor([[[[[[10., 13.]]],
                                           [[[16., 30.]]],
                                           [[[33., 23.]]]]],

                                         [[[[[30., 61.]]],
                                           [[[62., 45.]]],
                                           [[[59., 119.]]]]],

                                         [[[[[116., 90.]]],
                                           [[[156., 198.]]],
                                           [[[373., 326.]]]]]])
        self.no = 6
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)

        # Deserialize the engine from file
        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []

        for binding in engine:
            print('bingding:', binding, engine.get_binding_shape(binding))
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                self.input_w = engine.get_binding_shape(binding)[-1]
                self.input_h = engine.get_binding_shape(binding)[-2]
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings
        self.batch_size = engine.max_batch_size

    def infer(self, raw_image):
        self.ctx.push()
        # Restore
        stream = self.stream
        context = self.context
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings
        # Do image preprocess
        ori_shape = raw_image.shape
        batch_input_image = np.empty(shape=[self.batch_size, 3, self.input_h, self.input_w])
        input_image = self.preprocess_one(raw_image)
        np.copyto(batch_input_image, input_image)
        batch_input_image = np.ascontiguousarray(batch_input_image)

        # Copy input image to host buffer
        np.copyto(host_inputs[0], batch_input_image.ravel())
        start = time.time()
        # Transfer input data  to the GPU.
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        # Run inference.
        context.execute_async(batch_size=self.batch_size, bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        cuda.memcpy_dtoh_async(host_outputs[1], cuda_outputs[1], stream)
        cuda.memcpy_dtoh_async(host_outputs[2], cuda_outputs[2], stream)
        # Synchronize the stream
        stream.synchronize()
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()
        # Here we use the first row of output in that batch_size = 1
        output = host_outputs
        # Do postprocess
        bboxes = self.post_process(output, ori_shape)
        return bboxes

    def destroy(self):
        self.ctx.pop()

    def get_image(self, image_path):
        return cv2.imread(image_path)

    def scale_coords(self, img1_shape, coords, img0_shape, ratio_pad=None):
        if ratio_pad is None:
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]
        coords[:, [0, 2]] -= pad[0]
        coords[:, [1, 3]] -= pad[1]
        coords[:, :4] /= gain
        self.clip_coords(coords, img0_shape)
        return coords

    def clip_coords(self, boxes, img_shape):
        boxes[:, 0].clamp_(0, img_shape[1])
        boxes[:, 1].clamp_(0, img_shape[0])
        boxes[:, 2].clamp_(0, img_shape[1])
        boxes[:, 3].clamp_(0, img_shape[0])

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
        shape = img.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:
            r = min(r, 1.0)
        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        if auto:
            dw, dh = np.mod(dw, 64), np.mod(dh, 64)
        elif scaleFill:
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
        dw /= 2
        dh /= 2
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, ratio, (dw, dh)

    def preprocess_one(self, image):
        img = self.letterbox(image, new_shape=640)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img)
        img = img.float()
        img /= 255.0
        img = np.expand_dims(img, 0)
        return img

    def xywh2xyxy(self, x):
        y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        return y

    def make_grid(self, nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def non_max_suppression(self, prediction, conf_thres=0.4, iou_thres=0.5, classes=0, agnostic=False):
        """Performs Non-Maximum Suppression (NMS) on inference results

        Returns:
             detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
        """
        nc = prediction[0].shape[1] - 5
        xc = prediction[..., 4] > conf_thres
        min_wh, max_wh = 2, 4096
        max_det = 100
        time_limit = 10.0
        multi_label = nc > 1
        output = [None] * prediction.shape[0]
        t = time.time()
        for xi, x in enumerate(prediction):
            x = x[xc[xi]]
            if not x.shape[0]:
                continue
            x[:, 5:] *= x[:, 4:5]
            box = self.xywh2xyxy(x[:, :4])
            if multi_label:
                i, j = (x[:, 5:] > conf_thres).nonzero().t()
                x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
            else:
                conf, j = x[:, 5:].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
            if classes:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
            n = x.shape[0]
            if not n:
                continue
            c = x[:, 5:6] * (0 if agnostic else max_wh)
            boxes, scores = x[:, :4] + c, x[:, 4]
            i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
            if i.shape[0] > max_det:
                i = i[:max_det]
            output[xi] = x[i]
            if (time.time() - t) > time_limit:
                break
        return output

    def post_process(self, outputs, ori_shape):
        z = []
        grids = [80, 40, 20]
        for idx, output in enumerate(outputs):
            output = torch.tensor(output)
            print(output.shape)
            output = output.view(1, 3, grids[idx], grids[idx], self.no).contiguous()
            print(output)
            self.grid[idx] = self.make_grid(grids[idx], grids[idx])
            y = output.sigmoid()
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[idx]) * self.stride[idx]  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[idx]  # wh
            z.append(y.view(1, -1, self.no))
        pred = torch.cat(z, 1).cuda()
        boxes = self.non_max_suppression(pred)

        re_boxes = []
        for i, det in enumerate(boxes):
            if det is not None and len(det):
                det[:, :4] = self.scale_coords((640, 640), det[:, :4], ori_shape).round()
                for *xyxy, conf, cls in det:
                    x_min = xyxy[0].cpu()
                    y_min = xyxy[1].cpu()
                    x_max = xyxy[2].cpu()
                    y_max = xyxy[3].cpu()
                    re_boxes.append([x_min, y_min, x_max, y_max, conf.cpu(), cls.cpu()])
        return re_boxes


if __name__ == "__main__":

    engine_file_path = ""
    yolov5_wrapper = YoLov5TRT(engine_file_path)
    img_path = ''
    try:
        bboxes = yolov5_wrapper.infer(yolov5_wrapper.get_image(img_path))
    finally:
        yolov5_wrapper.destroy()