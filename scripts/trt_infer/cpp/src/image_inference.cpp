#include <fstream>
#include <iostream>
#include <sstream>
#include <chrono>
#include <vector>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            cerr << "Cuda failure: " << ret << endl;\
            abort();\
        }\
    } while (0)

#define DEVICE 0 // GPU id
#define NMS_THRESH 0.4
#define BBOX_CONF_THRESH 0.5
static Logger gLogger;
using namespace nvinfer1;
using namespace std;
using namespace cv;

static const int INPUT_W = 640;
static const int INPUT_H = 640;
static const int SINGLE_OUTPUT_SIZE = 80 + 5;
static const int GRIDS_NUM = 25200;

const char *INPUT_BLOB_NAME = "images";
const char *OUTPUT_BLOB_NAME = "output";


struct Object
{
    Rect rect;
    int label;
    float prob;
};


float *blobFromImage(cv::Mat &img)
{
    float *blob = new float[img.total() * 3];
    int channels = 3;
    int img_h = img.rows;
    int img_w = img.cols;
    for (size_t c = 0; c < channels; c++)
    {
        for (size_t h = 0; h < img_h; h++)
        {
            for (size_t w = 0; w < img_w; w++)
            {
                blob[c * img_w * img_h + h * img_w + w] = ((float)img.at<cv::Vec3b>(h, w)[c]) / 255.0f;
            }
        }
    }
    return blob;
}

float * slice(float* arr, int begin, int n){
    float * tmp= new float[n];
    for(size_t i = 0; i < n; i++){
        tmp[i] = arr[i + begin];
    }
    return tmp;
}

int getClassIndex(float * arr, int length){
    int max_index = 0;
    float score = arr[0];
    for(size_t i=1; i< length; i++){
        if(arr[i] > score){
            max_index = i;
            score = arr[i];
        }
    }
    return max_index;
}

vector<Object> decodeOutput(float * prob){
    vector<Object> outputs {};
    vector<Rect> bboxes {};
    vector<float> scores {};
    vector<int>  classs {};
    for (size_t i=0; i < GRIDS_NUM; i++){
        float * grid_output = slice(prob, i * SINGLE_OUTPUT_SIZE, SINGLE_OUTPUT_SIZE);
        if(grid_output[4] >=BBOX_CONF_THRESH){
            Object obj;
            int class_index = getClassIndex(grid_output, SINGLE_OUTPUT_SIZE);
            bboxes.push_back(Rect( grid_output[0] - grid_output[2] / 2, grid_output[1] - grid_output[3] / 2, grid_output[2], grid_output[3]));
            scores.push_back(grid_output[4]);
            classs.push_back(class_index);
        }
        delete grid_output;
    }
    vector<int> indices {};
    dnn::NMSBoxes(bboxes, scores, BBOX_CONF_THRESH, NMS_THRESH, indices);
    for(size_t i=0; i< indices.size(); i++){
        int idx = indices[i];
        Object obj;
        obj.rect = bboxes[idx];
        obj.prob = scores[idx];
        obj.label = classs[idx];
        outputs.push_back(obj);
    }
    return outputs;
}



void doInference(IExecutionContext& context, float* input, float* output, const int output_size, Size input_shape) {
    const ICudaEngine& engine = context.getEngine();
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);

    assert(engine.getBindingDataType(inputIndex) == nvinfer1::DataType::kFLOAT);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);
    assert(engine.getBindingDataType(outputIndex) == nvinfer1::DataType::kFLOAT);
    int mBatchSize = engine.getMaxBatchSize();

    CHECK(cudaMalloc(&buffers[inputIndex], 3 * input_shape.height * input_shape.width * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], output_size*sizeof(float)));

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, 3 * input_shape.height * input_shape.width * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(1, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], output_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}


int main(int argc, char** argv) {
    cudaSetDevice(DEVICE);
    
    char *trtModelStream{nullptr};
    size_t size{0};
    
    const string engine_file_path = "model.trt";
    ifstream file(engine_file_path, ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);

    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr); 

    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);

    delete[] trtModelStream;

    auto out_dims = engine->getBindingDimensions(1);
    auto output_size = 1;
    for(int j=0;j<out_dims.nbDims;j++) {
        output_size *= out_dims.d[j];
    }
    static float* prob = new float[output_size];

    const string input_img_path = "test_img/cat_dog.jpg";
    cv::Mat img = cv::imread(input_img_path);

    // cv::Mat pr_img = static_resize(img);
    cv::Mat pr_img {};
    resize(img, pr_img, Size(640, 640));

    float* blob;
    blob = blobFromImage(pr_img);
    float scale = min(INPUT_W / (img.cols*1.0), INPUT_H / (img.rows*1.0));
        
    // run inference
    auto start = chrono::system_clock::now();
    doInference(*context, blob, prob, output_size, pr_img.size());

    vector<Object> objects = decodeOutput(prob);
    for(auto obj : objects){
        rectangle(pr_img, Rect(obj.rect), Scalar(255,0, 255), 2);
    }
    imwrite("test_img/result.jpg", pr_img);

    // destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
    return 0;
}
