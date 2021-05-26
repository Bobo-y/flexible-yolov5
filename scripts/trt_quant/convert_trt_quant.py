import numpy as np
import util_trt
import argparse
import glob, os, cv2


def preprocess(image_raw, height, width):
    h, w, c = image_raw.shape
    # Calculate widht and height and paddings
    if float(width) / float(image_raw.shape[1]) < float(height) / float(image_raw.shape[0]):
        ratio = float(width) / float(image_raw.shape[1])
    else:
        ratio = float(height) / float(image_raw.shape[0])
    # Resize the image with long side while maintaining ratio
    rz_image = cv2.resize(image_raw, (int(image_raw.shape[0] * ratio), int(image_raw.shape[1] * ratio)))
    # Pad the short side with (0,0,0)
    image = np.zeros((width, height, 3), np.float32)
    image[0:int(image_raw.shape[1] * ratio), 0:int(image_raw.shape[0] * ratio)] = rz_image
    image = image.astype(np.float32)
    # Normalize to [0,1]
    image /= 255.0
    # HWC to CHW format:
    image = np.transpose(image, [2, 0, 1])
    return image


class DataLoader:
    def __init__(self, img_size, batch, batch_size, img_dir):
        self.index = 0
        self.length = batch
        self.batch_size = batch_size
        self.img_size = img_size
        self.img_dir = img_dir
        # self.img_list = [i.strip() for i in open('calib.txt').readlines()]
        self.img_list = glob.glob(os.path.join(self.img_dir, "*.jpg"))
        assert len(self.img_list) > self.batch_size * self.length, '{} must contains more than '.format(
            self.img_dir) + str(self.batch_size * self.length) + ' images to calib'
        print('found all {} images to calib.'.format(len(self.img_list)))
        self.calibration_data = np.zeros((self.batch_size, 3, self.img_size[0], self.img_size[1]), dtype=np.float32)

    def reset(self):
        self.index = 0

    def next_batch(self):
        if self.index < self.length:
            for i in range(self.batch_size):
                assert os.path.exists(self.img_list[i + self.index * self.batch_size]), 'not found!!'
                img = cv2.imread(self.img_list[i + self.index * self.batch_size])
                img = preprocess(img, self.img_size[0], self.img_size[1])
                self.calibration_data[i] = img

            self.index += 1
            # example only
            return np.ascontiguousarray(self.calibration_data, dtype=np.float32)
        else:
            return np.array([])

    def __len__(self):
        return self.length


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-dir', type=str, default='', help='calibration image path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')
    parser.add_argument('--batch-size', type=int, default=8, help='batch size')
    parser.add_argument('--batch', type=int, default=100, help='batch')
    parser.add_argument('--onnx-model', type=str, default='', help='onnx model path')
    parser.add_argument('--mode', type=str, default='fp16', help='tensorrt model fp16 or int8')
    parser.add_argument('--save-model', type=str, default='./trt_model', help='save_model_path')
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1
    print(opt)
    if opt.mode == 'fp16':
        fp16_mode = True
        int8_mode = False
    else:
        int8_mode = True
        fp16_mode = False
    print('*** onnx to tensorrt begin ***')
    # calibration
    calibration_stream = DataLoader(img_size=opt.img_size, batch=opt.batch, batch_size=opt.batch_size, img_dir=opt.img_dir)
    if not os.path.exists(opt.save_model):
        os.mkdir(opt.save_model)
    onnx_model_path = opt.onnx_model
    engine_model_path = os.path.join(opt.save_model, opt.model + '_model.trt')
    calibration_table = os.path.join(opt.save_model, opt.model + '_calibration.cache')
    # fixed_engine,校准产生校准表
    engine_fixed = util_trt.get_engine(opt.batch_size, onnx_model_path, engine_model_path, fp16_mode=fp16_mode,
                                       int8_mode=int8_mode, calibration_stream=calibration_stream,
                                       calibration_table_path=calibration_table, save_engine=True)
    assert engine_fixed, 'Broken engine_fixed'
    print('*** onnx to tensorrt completed ***\n')
