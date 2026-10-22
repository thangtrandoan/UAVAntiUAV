// Macro kiểm lỗi CUDA và TensorRT. Không dùng exception để tránh phụ thuộc.
#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

namespace uav {

// Dừng chương trình kèm vị trí lỗi. Dùng cho mọi lời gọi CUDA có thể thất bại.
#define UAV_CUDA_CHECK(call)                                                   \
  do {                                                                         \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) {                                                   \
      std::fprintf(stderr, "[CUDA] %s:%d  %s\n    -> %s (%d)\n", __FILE__,      \
                   __LINE__, #call, cudaGetErrorString(_e), (int)_e);          \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

// Kiểm lỗi của lần launch kernel ngay trước đó (lỗi launch là bất đồng bộ).
#define UAV_CUDA_CHECK_KERNEL() UAV_CUDA_CHECK(cudaGetLastError())

// Kiểm lỗi TensorRT: trả true nếu hợp lệ.
#define UAV_TRT_CHECK(cond, what)                                              \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::fprintf(stderr, "[TRT] %s:%d  %s\n", __FILE__, __LINE__, what);     \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

}  // namespace uav
