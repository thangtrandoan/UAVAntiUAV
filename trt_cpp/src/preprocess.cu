#include "uav_reid/preprocess.hpp"

#include <algorithm>
#include <cmath>

#include <cuda_fp16.h>

#include "uav_reid/cuda_check.hpp"

namespace uav {
namespace {

// ImageNet, đúng thứ tự kênh đầu ra (0=R, 1=G, 2=B).
// Để dạng literal trong hàm, không để mảng constexpr: mảng trong anonymous
// namespace không đọc được từ device code.
__device__ __forceinline__ float imnet_mean(int c) {
  return (c == 0) ? 0.485f : (c == 1) ? 0.456f : 0.406f;
}

__device__ __forceinline__ float imnet_std(int c) {
  return (c == 0) ? 0.229f : (c == 1) ? 0.224f : 0.225f;
}

// Gộp /255 và chuẩn hoá thành một phép FMA: out = byte * scale + bias.
__device__ __forceinline__ float norm(unsigned char v, int c) {
  const float s = imnet_std(c);
  return __fmaf_rn((float)v, 1.0f / (255.0f * s), -imnet_mean(c) / s);
}

// `v` đã ở thang 0..255 (tổng trọng số = 1) nên công thức giống hệt norm().
__device__ __forceinline__ float norm_f(float v, int c) {
  const float s = imnet_std(c);
  return v / (255.0f * s) - imnet_mean(c) / s;
}

__device__ __forceinline__ void store4(float* p, float a, float b, float c,
                                       float d) {
  *reinterpret_cast<float4*>(p) = make_float4(a, b, c, d);
}

__device__ __forceinline__ void store4(__half* p, float a, float b, float c,
                                       float d) {
  const __half2 lo = __floats2half2_rn(a, b);
  const __half2 hi = __floats2half2_rn(c, d);
  uint2 packed;
  packed.x = *reinterpret_cast<const unsigned*>(&lo);
  packed.y = *reinterpret_cast<const unsigned*>(&hi);
  *reinterpret_cast<uint2*>(p) = packed;
}

// ---------------------------------------------------------------------------
// Đường nhanh: nguồn đã là 224x224x3 liên tục. Mỗi luồng xử lý 4 pixel.
// 4 pixel x 3 kênh = 12 byte = 3 x uchar4, cửa hàng là float4 hoặc half2x2.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void crop_normalize_kernel(const uint8_t* __restrict__ src,
                                      size_t frame_pitch, int src_is_bgr,
                                      T* __restrict__ dst) {
  const int g = (blockIdx.x * blockDim.x + threadIdx.x);
  const int p = g * 4;  // pixel đầu tiên của nhóm 4
  if (p >= kDstPixels) return;

  const uint8_t* s = src + (size_t)blockIdx.y * frame_pitch + (size_t)p * 3;
  const uchar4 a = *reinterpret_cast<const uchar4*>(s);
  const uchar4 b = *reinterpret_cast<const uchar4*>(s + 4);
  const uchar4 c = *reinterpret_cast<const uchar4*>(s + 8);

  // Ánh xạ kênh theo thứ tự nguồn: RGB thì R={a.x,a.w,b.z,c.y};
  // BGR thì R nằm ở vị trí thứ ba của mỗi pixel.
  const bool bgr = (src_is_bgr != 0);
  const unsigned char r0 = bgr ? a.z : a.x, r1 = bgr ? b.y : a.w,
                      r2 = bgr ? c.x : b.z, r3 = bgr ? c.w : c.y;
  const unsigned char g0 = a.y, g1 = b.x, g2 = b.w, g3 = c.z;
  const unsigned char b0 = bgr ? a.x : a.z, b1 = bgr ? a.w : b.y,
                      b2 = bgr ? b.z : c.x, b3 = bgr ? c.y : c.w;

  T* out = dst + (size_t)blockIdx.y * (kDstPixels * 3);
  store4(out + 0 * kDstPixels + p, norm(r0, 0), norm(r1, 0), norm(r2, 0),
         norm(r3, 0));
  store4(out + 1 * kDstPixels + p, norm(g0, 1), norm(g1, 1), norm(g2, 1),
         norm(g3, 1));
  store4(out + 2 * kDstPixels + p, norm(b0, 2), norm(b1, 2), norm(b2, 2),
         norm(b3, 2));
}

// ---------------------------------------------------------------------------
// Đường Pillow, lượt ngang: (src_w) -> 224 cột của cửa sổ cắt.
// Ghi 0..255 float, HWC, để lượt dọc đọc theo cột liền mạch.
// ---------------------------------------------------------------------------
__global__ void resample_h_kernel(const uint8_t* __restrict__ src,
                                  size_t frame_pitch, int src_w, int src_h,
                                  const int* __restrict__ off,
                                  const float* __restrict__ wt, int ksize,
                                  int bgr, uint8_t* __restrict__ tmp) {
  const int xo = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (xo >= kDstSize || y >= src_h) return;

  const uint8_t* row =
      src + (size_t)blockIdx.z * frame_pitch + (size_t)y * (size_t)src_w * 3;
  const int base = off[xo];
  const float* k = wt + (size_t)xo * ksize;

  float acc[3] = {0.0f, 0.0f, 0.0f};
  for (int i = 0; i < ksize; ++i) {
    const float w = k[i];
    if (w == 0.0f) continue;
    const uint8_t* p = row + (size_t)(base + i) * 3;
    // Nguồn RGB -> ra R,G,B; nguồn BGR -> đảo.
    if (bgr) {
      acc[0] += w * (float)p[2];
      acc[1] += w * (float)p[1];
      acc[2] += w * (float)p[0];
    } else {
      acc[0] += w * (float)p[0];
      acc[1] += w * (float)p[1];
      acc[2] += w * (float)p[2];
    }
  }

  // Pillow ghi vùng tạm dạng 8-bit giữa hai lượt; làm tròn + kẹp để khớp.
  uint8_t* o = tmp + (((size_t)blockIdx.z * (size_t)src_h + (size_t)y) *
                          (size_t)kDstSize +
                      (size_t)xo) *
                     3;
  o[0] = (uint8_t)fminf(fmaxf(floorf(acc[0] + 0.5f), 0.0f), 255.0f);
  o[1] = (uint8_t)fminf(fmaxf(floorf(acc[1] + 0.5f), 0.0f), 255.0f);
  o[2] = (uint8_t)fminf(fmaxf(floorf(acc[2] + 0.5f), 0.0f), 255.0f);
}

// ---------------------------------------------------------------------------
// Đường Pillow, lượt dọc: (src_h) -> 224 hàng, rồi chuẩn hoá + đổi sang CHW.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void resample_v_norm_kernel(const uint8_t* __restrict__ tmp,
                                       int src_h,
                                       const int* __restrict__ off,
                                       const float* __restrict__ wt, int ksize,
                                       T* __restrict__ dst) {
  const int xo = blockIdx.x * blockDim.x + threadIdx.x;
  const int yo = blockIdx.y * blockDim.y + threadIdx.y;
  if (xo >= kDstSize || yo >= kDstSize) return;

  const int base = off[yo];
  const float* k = wt + (size_t)yo * ksize;
  const size_t row_stride = (size_t)kDstSize * 3;

  float acc[3] = {0.0f, 0.0f, 0.0f};
  for (int i = 0; i < ksize; ++i) {
    const float w = k[i];
    if (w == 0.0f) continue;
    const uint8_t* r = tmp + (((size_t)blockIdx.z * (size_t)src_h +
                               (size_t)(base + i)) *
                                  (size_t)kDstSize +
                              (size_t)xo) *
                             3;
    acc[0] += w * r[0];
    acc[1] += w * r[1];
    acc[2] += w * r[2];
  }
  (void)row_stride;

  T* out = dst + (size_t)blockIdx.z * (kDstPixels * 3) +
           (size_t)yo * kDstSize + xo;
  out[0 * kDstPixels] = (T)norm_f(acc[0], 0);
  out[1 * kDstPixels] = (T)norm_f(acc[1], 1);
  out[2 * kDstPixels] = (T)norm_f(acc[2], 2);
}

template <typename T>
void dispatch_crop(const uint8_t* src, int n_frames, size_t frame_pitch,
                   int bgr, void* dst, cudaStream_t stream) {
  constexpr int kThreads = 256;
  const int groups = kDstPixels / 4;  // 12544, chia hết
  const dim3 grid((unsigned)(groups / kThreads), (unsigned)n_frames);
  crop_normalize_kernel<T><<<grid, kThreads, 0, stream>>>(
      src, frame_pitch, bgr, reinterpret_cast<T*>(dst));
  UAV_CUDA_CHECK_KERNEL();
}

template <typename T>
void dispatch_resize(const uint8_t* src, int n_frames, size_t frame_pitch,
                     int src_w, int src_h, const int* d_off_x,
                     const float* d_wt_x, int ksize_x, const int* d_off_y,
                     const float* d_wt_y, int ksize_y, uint8_t* scratch,
                     int bgr, void* dst, cudaStream_t stream) {
  const dim3 block(32, 8);
  const dim3 grid_h((kDstSize + block.x - 1) / block.x,
                    (unsigned)((src_h + block.y - 1) / block.y),
                    (unsigned)n_frames);
  resample_h_kernel<<<grid_h, block, 0, stream>>>(
      src, frame_pitch, src_w, src_h, d_off_x, d_wt_x, ksize_x, bgr, scratch);
  UAV_CUDA_CHECK_KERNEL();

  const dim3 grid_v((kDstSize + block.x - 1) / block.x,
                    (kDstSize + block.y - 1) / block.y, (unsigned)n_frames);
  resample_v_norm_kernel<T><<<grid_v, block, 0, stream>>>(
      scratch, src_h, d_off_y, d_wt_y, ksize_y, reinterpret_cast<T*>(dst));
  UAV_CUDA_CHECK_KERNEL();
}

}  // namespace

ResizeTaps build_pil_taps(int in_size) {
  if (in_size <= 0) {
    std::fprintf(stderr, "[resize] in_size phải > 0\n");
    std::exit(EXIT_FAILURE);
  }
  const double scale = (double)in_size / (double)kSrcSize;
  const double filterscale = std::max(1.0, scale);
  const double support = 1.0 * filterscale;
  const int ksize = (int)std::ceil(support) * 2 + 1;

  ResizeTaps t;
  t.ksize = ksize;
  t.offset.assign(kDstSize, 0);
  t.weights.assign((size_t)kDstSize * (size_t)ksize, 0.0f);

  const double ss = 1.0 / filterscale;
  for (int j = 0; j < kDstSize; ++j) {
    // Chỉ số đầu ra trong không gian 256x256 sau khi đã trừ phần cắt.
    const int xx = kCropOrigin + j;
    const double center = ((double)xx + 0.5) * scale;

    int xmin = (int)(center - support + 0.5);
    if (xmin < 0) xmin = 0;
    int xmax = (int)(center + support + 0.5);
    if (xmax > in_size) xmax = in_size;
    const int n = xmax - xmin;

    float* k = &t.weights[(size_t)j * (size_t)ksize];
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
      const double x = ((double)(i + xmin) - center + 0.5) * ss;
      const double ax = std::fabs(x);
      const double w = (ax < 1.0) ? (1.0 - ax) : 0.0;
      k[i] = (float)w;
      sum += w;
    }
    // Chuẩn hoá để tổng trọng số = 1 (giữ nguyên thang 0..255 của nguồn).
    if (sum > 0.0) {
      for (int i = 0; i < n; ++i) k[i] = (float)((double)k[i] / sum);
    }
    t.offset[j] = xmin;
  }
  return t;
}

size_t resize_scratch_bytes(int n_frames, int src_h) {
  return (size_t)n_frames * (size_t)src_h * (size_t)kDstSize * 3;
}

void launch_crop_normalize(const uint8_t* src, int n_frames, size_t frame_pitch,
                           PixelType dst_type, ChannelOrder order, void* dst,
                           cudaStream_t stream) {
  const int bgr = (order == ChannelOrder::kBGR) ? 1 : 0;
  if (dst_type == PixelType::kFloat32) {
    dispatch_crop<float>(src, n_frames, frame_pitch, bgr, dst, stream);
  } else {
    dispatch_crop<__half>(src, n_frames, frame_pitch, bgr, dst, stream);
  }
}

void launch_resize_pil(const uint8_t* src, int n_frames, size_t frame_pitch,
                       int src_w, int src_h, const int* d_off_x,
                       const float* d_wt_x, int ksize_x, const int* d_off_y,
                       const float* d_wt_y, int ksize_y, uint8_t* scratch,
                       PixelType dst_type, ChannelOrder order, void* dst,
                       cudaStream_t stream) {
  const int bgr = (order == ChannelOrder::kBGR) ? 1 : 0;
  if (dst_type == PixelType::kFloat32) {
    dispatch_resize<float>(src, n_frames, frame_pitch, src_w, src_h, d_off_x,
                           d_wt_x, ksize_x, d_off_y, d_wt_y, ksize_y, scratch,
                           bgr, dst, stream);
  } else {
    dispatch_resize<__half>(src, n_frames, frame_pitch, src_w, src_h, d_off_x,
                            d_wt_x, ksize_x, d_off_y, d_wt_y, ksize_y, scratch,
                            bgr, dst, stream);
  }
}

size_t taps_device_bytes(const ResizeTaps& t) {
  return sizeof(int) * kDstSize +
         sizeof(float) * (size_t)kDstSize * (size_t)t.ksize;
}

void copy_crop_256_to_staging(const uint8_t* host_src, size_t host_pitch_bytes,
                              uint8_t* dev_staging, cudaStream_t stream) {
  // Bỏ 16 hàng trên và 16 cột mỗi bên; mỗi hàng còn 224*3 = 672 byte liên tục.
  const uint8_t* start = host_src + (size_t)kCropOrigin * host_pitch_bytes +
                         (size_t)kCropOrigin * 3;
  UAV_CUDA_CHECK(cudaMemcpy2DAsync(dev_staging, (size_t)kDstSize * 3, start,
                                   host_pitch_bytes, (size_t)kDstSize * 3,
                                   (size_t)kDstSize, cudaMemcpyHostToDevice,
                                   stream));
}

}  // namespace uav
