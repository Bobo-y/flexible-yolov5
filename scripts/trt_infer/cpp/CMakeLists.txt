cmake_minimum_required(VERSION 2.6)

project(yolov5_trt)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)
# include_directories(/usr/local/include/eigen3)
link_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# tensorrt
include_directories(/home/Resources/TensorRT-8.4.0.6/include)
link_directories(/home/Resources/TensorRT-8.4.0.6/lib)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB My_Source_Files ${PROJECT_SOURCE_DIR}/src/*.cpp)
add_executable(yolov5 ${My_Source_Files})
target_link_libraries(yolov5 nvinfer)
target_link_libraries(yolov5 cudart)
target_link_libraries(yolov5 ${OpenCV_LIBS})

add_definitions(-O2 -pthread)
