// Tiền xử lý ảnh trên GPU: uint8 HWC -> float NCHW, đã chuẩn hoá ImageNet.
//
// Khớp với torchvision ở evaluate_reid.py:283-288 và infer.py:913-916:
//     Resize((256, 256)) -> CenterCrop((224, 224)) -> ToTensor() -> Normalize(...)
//
// Ảnh trong processed/ đã đúng 256x256 nên Resize là no-op; khi đó dùng
// launch_crop_normalize để kết quả khớp tuyệt đối (không nội suy).
//
// Với ảnh kích thước bất kỳ, launch_resize_pil dựng lại đúng thuật toán của
// Pillow (xem build_pil_taps) chứ không nội suy 2 điểm: Pillow NHÂN rộng nhân
// lọc theo tỉ lệ thu nhỏ (antialias), nên nội suy 2 điểm lệch tới ~1.4 đơn vị
// đã chuẩn hoá — đủ làm lệch embedding.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

namespace uav {

enum class PixelType { kFloat32, kFloat16 };
enum class ChannelOrder { kRGB, kBGR };

constexpr int kSrcSize = 256;                           // Resize((256,256))
constexpr int kDstSize = 224;                           // CenterCrop((224,224))
constexpr int kCropOrigin = (kSrcSize - kDstSize) / 2;  // = 16
constexpr int kDstPixels = kDstSize * kDstSize;         // = 50176
constexpr size_t kDstFrameBytes32 = (size_t)kDstPixels * 3 * 4;
constexpr size_t kDstFrameBytes16 = (size_t)kDstPixels * 3 * 2;

// Bảng trọng số nội suy kiểu Pillow cho MỘT trục, chỉ cho cửa sổ cắt [16,240).
//
// Công thức lấy từ Pillow libImaging/Resample.c::precompute_coeffs:
//   scale        = in_size / 256
//   filterscale  = max(1, scale)
//   support      = support_filter(=1.0) * filterscale
//   center       = (xx + 0.5) * scale
//   w_i          = triangle((i + xmin - center + 0.5) / filterscale)
//   w_i         /= sum(w)
struct ResizeTaps {
  int ksize = 0;               // số tap tối đa cho mỗi đầu ra
  std::vector<int> offset;     // [kDstSize] chỉ số nguồn đầu tiên
  std::vector<float> weights;  // [kDstSize * ksize], mỗi hàng tổng = 1
};

ResizeTaps build_pil_taps(int in_size);

// Vùng nhớ tạm cho lượt nội suy ngang: [frame][src_h][224][3] uint8.
// Đúng như Pillow: kết quả lượt ngang được ghi 8-bit trước khi làm lượt dọc.
size_t resize_scratch_bytes(int n_frames, int src_h);

// Vùng nguồn đã cắt sẵn về 224x224 (device), mỗi frame `frame_pitch` byte.
// Dùng khi ảnh nguồn là 256x256: pipeline chỉ cần chép vùng cắt.
void launch_crop_normalize(const uint8_t* src, int n_frames, size_t frame_pitch,
                           PixelType dst_type, ChannelOrder order, void* dst,
                           cudaStream_t stream);

// Nguồn kích thước bất kỳ: nội suy kiểu Pillow sang 256x256 rồi cắt 224x224.
// Bảng trọng số truyền vào dưới dạng con trỏ DEVICE, do phía gọi cache lại —
// tránh cudaMalloc/cudaFree mỗi lần chạy. `scratch` phải có ít nhất
// resize_scratch_bytes(n_frames, src_h) byte.
void launch_resize_pil(const uint8_t* src, int n_frames, size_t frame_pitch,
                       int src_w, int src_h, const int* d_off_x,
                       const float* d_wt_x, int ksize_x, const int* d_off_y,
                       const float* d_wt_y, int ksize_y, uint8_t* scratch,
                       PixelType dst_type, ChannelOrder order, void* dst,
                       cudaStream_t stream);

// Số byte cần cho bảng trọng số trên device (offset + weight của một trục).
size_t taps_device_bytes(const ResizeTaps& t);

// Chép riêng vùng cắt 224x224 từ ảnh 256x256 (host) sang vùng staging (device).
// Dùng cudaMemcpy2DAsync nên chỉ truyền 150528 B/frame thay vì 196608 B.
void copy_crop_256_to_staging(const uint8_t* host_src, size_t host_pitch_bytes,
                              uint8_t* dev_staging, cudaStream_t stream);

}  // namespace uav
