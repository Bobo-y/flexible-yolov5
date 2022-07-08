 # This only a simple demo for use tensorrt model inference a image.

 ## Prepare serialized engine file


 ## Build the demo


you should set the TensorRT path and CUDA path in CMakeLists.txt.

And set test image path and your trt model path.

you can first build the demo:

```shell
cd scripts/trt_infer/cpp
mkdir build
cd  build
cmake ..
make
```

Then you can run this demo

```shell
./build/yolov5
```
