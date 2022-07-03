from .triton_serving_client import TritonServingClient
from .tf_serving_client import TFServingClient
from .yolov5_client import Yolov5

# start a triton server or tf_sering
serving_client = TritonServingClient('0.0.0.0:8001')

yolov5 = Yolov5(model_serving=serving_client)
img_path = ''
results = yolov5(img_path)

