cmake_minimum_required(VERSION 3.20)
project(uav_reid_trt LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Orin = sm_87. Đổi bằng -DCMAKE_CUDA_ARCHITECTURES=... nếu chạy GPU khác.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 87)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

# TensorRT không có FindTensorRT chuẩn; dò trực tiếp. JetPack đặt header trong
# /usr/include/aarch64-linux-gnu và lib trong /usr/lib/aarch64-linux-gnu.
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS
    /usr/include/aarch64-linux-gnu
    /usr/include/x86_64-linux-gnu
    /usr/include
    /usr/local/cuda/include)

find_library(TENSORRT_LIBRARY nvinfer
  HINTS
    /usr/lib/aarch64-linux-gnu
    /usr/lib/x86_64-linux-gnu
    /usr/lib
    /usr/local/cuda/lib64)

if(NOT TENSORRT_INCLUDE_DIR)
  message(FATAL_ERROR "không tìm thấy NvInfer.h — cài TensorRT hoặc đặt TENSORRT_INCLUDE_DIR")
endif()
if(NOT TENSORRT_LIBRARY)
  message(FATAL_ERROR "không tìm thấy libnvinfer — cài TensorRT hoặc đặt TENSORRT_LIBRARY")
endif()

message(STATUS "TensorRT header : ${TENSORRT_INCLUDE_DIR}")
message(STATUS "TensorRT lib    : ${TENSORRT_LIBRARY}")
message(STATUS "OpenCV          : ${OpenCV_VERSION}")
message(STATUS "CUDA arch       : ${CMAKE_CUDA_ARCHITECTURES}")

# ---------------------------------------------------------------------------
# Lõi dùng chung cho cả CLI và thư viện cho Python.
# ---------------------------------------------------------------------------
add_library(uav_core STATIC
  src/engine.cpp
  src/pipeline.cpp
  src/preprocess.cu)

target_include_directories(uav_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${TENSORRT_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS})

target_link_libraries(uav_core PUBLIC
  ${TENSORRT_LIBRARY}
  CUDA::cudart
  ${OpenCV_LIBS})

# Chỉ bật cảnh báo mức cao; không dùng -Werror để tránh vỡ khi đổi phiên bản TRT.
target_compile_options(uav_core PUBLIC
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wno-unused-parameter>
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)

set_target_properties(uav_core PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
add_executable(uav_reid_trt src/main.cpp)
target_link_libraries(uav_reid_trt PRIVATE uav_core)
set_target_properties(uav_reid_trt PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

# ---------------------------------------------------------------------------
# Thư viện cho Python (nạp bằng ctypes trong uav_reid_engine.py).
# Dùng C ABI thuần để không phụ thuộc pybind11 / header Python.
# ---------------------------------------------------------------------------
add_library(uav_reid_engine SHARED src/capi.cpp)
target_link_libraries(uav_reid_engine PRIVATE uav_core)
set_target_properties(uav_reid_engine PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  POSITION_INDEPENDENT_CODE ON
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
