#include "uav_reid/pipeline.hpp"

#include <cstdio>
#include <cstring>
#include <initializer_list>

#include "uav_reid/cuda_check.hpp"

namespace uav {
namespace {

const char* kBackboneIn = "input_img";
const char* kBackboneOut = "cnn_feat";
const char* kTemporalIn = "seq_feats";
const char* kTemporalOut = "reid_embed";

nvinfer1::Dims make_dims(std::initializer_list<int> v) {
  nvinfer1::Dims d{};
  d.nbDims = static_cast<int>(v.size());
  int i = 0;
  for (int x : v) d.d[i++] = x;
  return d;
}

size_t dtype_bytes(nvinfer1::DataType t) {
  return t == nvinfer1::DataType::kHALF ? 2 : 4;
}

PixelType to_pixel_type(nvinfer1::DataType t) {
  return t == nvinfer1::DataType::kHALF ? PixelType::kFloat16
                                        : PixelType::kFloat32;
}

}  // namespace

Pipeline::Pipeline(const Config& cfg) : cfg_(cfg) {
  // `num_frames == 0` hợp lệ: lấy N từ engine temporal (xem prepare_engines).
  // Số âm thì sai.
  if (cfg_.num_frames < 0) {
    std::fprintf(stderr, "[Pipeline] num_frames phải >= 0 (0 = lấy từ engine)\n");
    std::exit(EXIT_FAILURE);
  }
  if (cfg_.max_clips <= 0) {
    std::fprintf(stderr, "[Pipeline] max_clips phải > 0\n");
    std::exit(EXIT_FAILURE);
  }
  num_frames_ = cfg_.num_frames;
  max_clips_ = cfg_.max_clips;
  order_ = cfg_.order;

  UAV_CUDA_CHECK(cudaSetDevice(cfg_.device));
  UAV_CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
  UAV_CUDA_CHECK(cudaEventCreate(&ev_a_));
  UAV_CUDA_CHECK(cudaEventCreate(&ev_b_));

  prepare_engines();
  if (num_frames_ <= 0) {
    std::fprintf(stderr,
                 "[Pipeline] không xác định được số frame cho cửa sổ temporal\n");
    std::exit(EXIT_FAILURE);
  }
  prepare_buffers();
  if (cfg_.mode == Mode::kStream) prepare_stream();
}

Pipeline::~Pipeline() {
  if (ring_ != nullptr) cudaFree(ring_);
  if (intermediate_ != nullptr) cudaFree(intermediate_);
  if (device_staging_ != nullptr) cudaFree(device_staging_);
  if (resize_scratch_ != nullptr) cudaFree(resize_scratch_);
  if (d_off_x_ != nullptr) cudaFree(d_off_x_);
  if (d_wt_x_ != nullptr) cudaFree(d_wt_x_);
  if (d_off_y_ != nullptr) cudaFree(d_off_y_);
  if (d_wt_y_ != nullptr) cudaFree(d_wt_y_);
  if (ev_a_ != nullptr) cudaEventDestroy(ev_a_);
  if (ev_b_ != nullptr) cudaEventDestroy(ev_b_);
  if (stream_ != nullptr) cudaStreamDestroy(stream_);
}

void Pipeline::prepare_engines() {
  backbone_.set_logger_verbose(cfg_.verbose_trt);
  temporal_.set_logger_verbose(cfg_.verbose_trt);

  backbone_.load(cfg_.backbone_plan, cfg_.device, stream_);
  temporal_.load(cfg_.temporal_plan, cfg_.device, stream_);

  if (!backbone_.has(kBackboneIn) || !backbone_.has(kBackboneOut)) {
    std::fprintf(stderr, "[Pipeline] %s phải có '%s' và '%s'\n",
                 cfg_.backbone_plan.c_str(), kBackboneIn, kBackboneOut);
    std::exit(EXIT_FAILURE);
  }
  if (!temporal_.has(kTemporalIn) || !temporal_.has(kTemporalOut)) {
    std::fprintf(stderr, "[Pipeline] %s phải có '%s' và '%s'\n",
                 cfg_.temporal_plan.c_str(), kTemporalIn, kTemporalOut);
    std::exit(EXIT_FAILURE);
  }

  // Chiều đặc trưng lấy từ chiều cuối của cnn_feat. Chiều này tĩnh trong engine
  // (chỉ batch là động), nên đọc được trước khi gọi allocate().
  const nvinfer1::Dims bb_out = backbone_.at(kBackboneOut).dims;
  if (bb_out.nbDims != 2) {
    std::fprintf(stderr, "[Pipeline] '%s' phải là 2 chiều, đang %d chiều\n",
                 kBackboneOut, bb_out.nbDims);
    std::exit(EXIT_FAILURE);
  }
  visual_dim_ = bb_out.d[1];
  if (visual_dim_ <= 0) {
    std::fprintf(stderr, "[Pipeline] '%s' có chiều đặc trưng động (%d)\n",
                 kBackboneOut, visual_dim_);
    std::exit(EXIT_FAILURE);
  }

  // `num_frames <= 0` = lấy N từ chính engine temporal. Cần thiết vì shape được
  // chốt lúc allocate(): nếu config ghi 12 mà engine build với N=16 thì
  // choose_profile() không tìm được profile nào và dừng ngay, trước khi tầng
  // trên kịp so sánh và báo lỗi rõ ràng.
  if (num_frames_ <= 0) {
    nvinfer1::Dims d = temporal_.profile_shape(kTemporalIn, 0,
                                               nvinfer1::OptProfileSelector::kOPT);
    if (d.nbDims != 3 || d.d[1] <= 0) {
      d = temporal_.profile_shape(kTemporalIn, 0,
                                  nvinfer1::OptProfileSelector::kMIN);
    }
    if (d.nbDims != 3 || d.d[1] <= 0) {
      std::fprintf(stderr,
                   "[Pipeline] không đọc được N của engine temporal từ profile "
                   "của '%s'; hãy truyền num_frames tường minh\n", kTemporalIn);
      std::exit(EXIT_FAILURE);
    }
    num_frames_ = static_cast<int>(d.d[1]);
  }

  // Cố định shape một lần. Mọi lần chạy sau dùng đúng shape này.
  backbone_.set_input_shape(kBackboneIn,
                            make_dims({backbone_batch(), 3, kDstSize, kDstSize}));
  if (cfg_.mode == Mode::kStream) {
    // Một frame mỗi bước: batch 1, temporal đọc cửa sổ N frame.
    temporal_.set_input_shape(kTemporalIn,
                              make_dims({1, num_frames_, visual_dim_}));
  } else {
    temporal_.set_input_shape(kTemporalIn,
                              make_dims({max_clips_, num_frames_, visual_dim_}));
  }

  if (cfg_.mode == Mode::kClip) {
    // Zero-copy: cnn_feat (B*N, D) và seq_feats (K, N, D) cùng layout bộ nhớ.
    intermediate_bytes_ = static_cast<size_t>(backbone_batch()) *
                          static_cast<size_t>(visual_dim_) *
                          dtype_bytes(backbone_.at(kBackboneOut).dtype);
    UAV_CUDA_CHECK(cudaMalloc(&intermediate_, intermediate_bytes_));
    backbone_.bind_external(kBackboneOut, intermediate_);
    temporal_.bind_external(kTemporalIn, intermediate_);
  }

  backbone_.allocate(stream_);
  temporal_.allocate(stream_);

  if (cfg_.mode == Mode::kClip) {
    if (backbone_.bytes(kBackboneOut) != temporal_.bytes(kTemporalIn)) {
      std::fprintf(
          stderr, "[Pipeline] không zero-copy được: '%s'=%zu B nhưng '%s'=%zu B\n",
          kBackboneOut, backbone_.bytes(kBackboneOut), kTemporalIn,
          temporal_.bytes(kTemporalIn));
      std::exit(EXIT_FAILURE);
    }
  } else {
    // kStream: backbone ra (1,D), temporal nhận (1,N,D) — khác kích thước nên
    // KHÔNG zero-copy được; chỉ cần chiều đặc trưng khớp.
    const nvinfer1::Dims ti = temporal_.at(kTemporalIn).dims;
    if (ti.nbDims != 3 || ti.d[2] != visual_dim_) {
      std::fprintf(stderr,
                   "[Pipeline] '%s' phải là (1, %d, %d), engine cho %d chiều\n",
                   kTemporalIn, num_frames_, visual_dim_, ti.nbDims);
      std::exit(EXIT_FAILURE);
    }
    if (ti.d[1] != num_frames_) {
      std::fprintf(stderr,
                   "[Pipeline] '%s' có N=%d nhưng --num-frames=%d\n", kTemporalIn,
                   (int)ti.d[1], num_frames_);
      std::exit(EXIT_FAILURE);
    }
  }

  const nvinfer1::Dims tp_out = temporal_.at(kTemporalOut).dims;
  if (tp_out.nbDims != 2) {
    std::fprintf(stderr, "[Pipeline] '%s' phải là 2 chiều\n", kTemporalOut);
    std::exit(EXIT_FAILURE);
  }
  embed_dim_ = tp_out.d[1];
  pixel_type_ = to_pixel_type(backbone_.at(kBackboneIn).dtype);
}

void Pipeline::prepare_buffers() { ensure_staging((size_t)kDstPixels * 3); }

void Pipeline::ensure_taps(int src_w, int src_h) {
  if (src_w == taps_w_ && src_h == taps_h_) return;
  taps_x_ = build_pil_taps(src_w);
  taps_y_ = build_pil_taps(src_h);
  taps_w_ = src_w;
  taps_h_ = src_h;

  // Nạp bảng trọng số lên device MỘT lần cho mỗi kích thước nguồn.
  auto upload = [](const ResizeTaps& t, int** off, float** wt) {
    const size_t off_bytes = sizeof(int) * (size_t)kDstSize;
    const size_t wt_bytes =
        sizeof(float) * (size_t)kDstSize * (size_t)t.ksize;
    if (*off != nullptr) UAV_CUDA_CHECK(cudaFree(*off));
    if (*wt != nullptr) UAV_CUDA_CHECK(cudaFree(*wt));
    UAV_CUDA_CHECK(cudaMalloc(off, off_bytes));
    UAV_CUDA_CHECK(cudaMalloc(wt, wt_bytes));
    UAV_CUDA_CHECK(cudaMemcpy(*off, t.offset.data(), off_bytes,
                              cudaMemcpyHostToDevice));
    UAV_CUDA_CHECK(cudaMemcpy(*wt, t.weights.data(), wt_bytes,
                              cudaMemcpyHostToDevice));
  };
  upload(taps_x_, &d_off_x_, &d_wt_x_);
  upload(taps_y_, &d_off_y_, &d_wt_y_);
  ksize_x_ = taps_x_.ksize;
  ksize_y_ = taps_y_.ksize;

  const size_t need = resize_scratch_bytes(backbone_batch(), src_h);
  if (need > resize_scratch_bytes_) {
    if (resize_scratch_ != nullptr) UAV_CUDA_CHECK(cudaFree(resize_scratch_));
    UAV_CUDA_CHECK(cudaMalloc(&resize_scratch_, need));
    resize_scratch_bytes_ = need;
  }
}

void Pipeline::ensure_staging(size_t bytes_per_frame) {
  const size_t bytes =
      bytes_per_frame * static_cast<size_t>(backbone_batch());
  if (bytes <= staging_bytes_) return;
  if (device_staging_ != nullptr) UAV_CUDA_CHECK(cudaFree(device_staging_));
  UAV_CUDA_CHECK(cudaMalloc(&device_staging_, bytes));
  staging_bytes_ = bytes;
}

void Pipeline::run(const std::vector<HostFrame>& frames, int clips,
                   float* out_host, bool measure) {
  if (cfg_.mode == Mode::kStream) {
    std::fprintf(stderr,
                 "[Pipeline] đang ở chế độ kStream: dùng push_frame()/embed()\n");
    std::exit(EXIT_FAILURE);
  }
  if (clips <= 0 || clips > max_clips_) {
    std::fprintf(stderr, "[Pipeline] clips=%d ngoài khoảng 1..%d\n", clips,
                 max_clips_);
    std::exit(EXIT_FAILURE);
  }
  const int need = clips * num_frames_;
  if (static_cast<int>(frames.size()) < need) {
    std::fprintf(stderr, "[Pipeline] cần %d frame, nhận %zu\n", need,
                 frames.size());
    std::exit(EXIT_FAILURE);
  }

  const int batch = backbone_batch();
  const size_t fast_frame = (size_t)kDstPixels * 3;

  auto t_start = [&]() {
    if (measure) UAV_CUDA_CHECK(cudaEventRecord(ev_a_, stream_));
  };
  auto t_stop = [&](float* dst) {
    if (!measure) return;
    UAV_CUDA_CHECK(cudaEventRecord(ev_b_, stream_));
    UAV_CUDA_CHECK(cudaEventSynchronize(ev_b_));
    UAV_CUDA_CHECK(cudaEventElapsedTime(dst, ev_a_, ev_b_));
  };

  // Đường nhanh khi mọi frame đúng 256x256 (Resize của torchvision là no-op).
  bool fast = true;
  size_t max_bytes = fast_frame;
  for (int i = 0; i < need; ++i) {
    const HostFrame& f = frames[(size_t)i];
    if (f.data == nullptr) {
      std::fprintf(stderr, "[Pipeline] frame %d có data null\n", i);
      std::exit(EXIT_FAILURE);
    }
    if (f.width != kSrcSize || f.height != kSrcSize) fast = false;
    const size_t b = (size_t)f.width * (size_t)f.height * 3;
    if (b > max_bytes) max_bytes = b;
  }

  if (!fast) {
    // Đường tổng quát dùng chung một kích thước nguồn cho cả batch.
    for (int i = 1; i < need; ++i) {
      if (frames[(size_t)i].width != frames[0].width ||
          frames[(size_t)i].height != frames[0].height) {
        std::fprintf(stderr,
                     "[Pipeline] đường resize yêu cầu mọi frame cùng kích thước "
                     "(frame 0 = %dx%d, frame %d = %dx%d)\n",
                     frames[0].width, frames[0].height, i,
                     frames[(size_t)i].width, frames[(size_t)i].height);
        std::exit(EXIT_FAILURE);
      }
    }
  }

  const size_t per_frame = fast ? fast_frame : max_bytes;
  ensure_staging(per_frame);

  t_start();

  // 1. Upload (và cắt luôn 224x224 ở đường nhanh) rồi tiền xử lý trên GPU.
  for (int i = 0; i < need; ++i) {
    const HostFrame& f = frames[(size_t)i];
    const size_t pitch = f.stride > 0 ? (size_t)f.stride : (size_t)f.width * 3;
    uint8_t* dst = static_cast<uint8_t*>(device_staging_) + (size_t)i * per_frame;
    if (fast) {
      copy_crop_256_to_staging(f.data, pitch, dst, stream_);
    } else {
      UAV_CUDA_CHECK(cudaMemcpy2DAsync(dst, (size_t)f.width * 3, f.data, pitch,
                                       (size_t)f.width * 3, (size_t)f.height,
                                       cudaMemcpyHostToDevice, stream_));
    }
  }

  // Slot dư lặp frame cuối để giữ shape cố định khi clips < max_clips.
  for (int i = need; i < batch; ++i) {
    UAV_CUDA_CHECK(cudaMemcpyAsync(
        static_cast<uint8_t*>(device_staging_) + (size_t)i * per_frame,
        static_cast<uint8_t*>(device_staging_) + (size_t)(need - 1) * per_frame,
        per_frame, cudaMemcpyDeviceToDevice, stream_));
  }

  void* bb_in = backbone_.ptr(kBackboneIn);
  if (fast) {
    launch_crop_normalize(static_cast<const uint8_t*>(device_staging_), batch,
                          fast_frame, pixel_type_, order_, bb_in, stream_);
  } else {
    ensure_taps(frames[0].width, frames[0].height);
    launch_resize_pil(static_cast<const uint8_t*>(device_staging_), batch,
                      per_frame, frames[0].width, frames[0].height, d_off_x_,
                      d_wt_x_, ksize_x_, d_off_y_, d_wt_y_, ksize_y_,
                      resize_scratch_, pixel_type_, order_, bb_in, stream_);
  }

  t_stop(&timings_.upload_preprocess_ms);

  // 2. Backbone.
  t_start();
  backbone_.infer(stream_);
  t_stop(&timings_.backbone_ms);

  // 3. Temporal + head + L2 norm, đọc thẳng vùng chia sẻ với backbone.
  t_start();
  temporal_.infer(stream_);
  t_stop(&timings_.temporal_ms);

  timings_.total_ms = measure ? timings_.upload_preprocess_ms +
                                   timings_.backbone_ms + timings_.temporal_ms
                             : 0.0f;

  // 4. Về host, chỉ phần clip thật.
  const size_t row = static_cast<size_t>(embed_dim_) * sizeof(float);
  UAV_CUDA_CHECK(cudaMemcpyAsync(out_host, temporal_.ptr(kTemporalOut),
                                 row * static_cast<size_t>(clips),
                                 cudaMemcpyDeviceToHost, stream_));
  UAV_CUDA_CHECK(cudaStreamSynchronize(stream_));
}


// ===========================================================================
// kStream: bộ đệm vòng feature.
// ===========================================================================
void Pipeline::prepare_stream() {
  if (temporal_.at(kTemporalOut).dtype != nvinfer1::DataType::kFLOAT) {
    std::fprintf(stderr, "[Pipeline] kStream cần '%s' kiểu FLOAT để chép về host\n",
                 kTemporalOut);
    std::exit(EXIT_FAILURE);
  }
  // Chép byte thô vào bộ đệm nên kiểu của cnn_feat và seq_feats phải trùng nhau.
  if (dtype_bytes(temporal_.at(kTemporalIn).dtype) !=
      dtype_bytes(backbone_.at(kBackboneOut).dtype)) {
    std::fprintf(stderr,
                 "[Pipeline] kStream cần '%s' và '%s' cùng kiểu dữ liệu\n",
                 kTemporalIn, kBackboneOut);
    std::exit(EXIT_FAILURE);
  }

  // N thật của engine temporal. Có thể khác `num_frames_` (config) — chênh lệch
  // này làm embedding lệch so với PyTorch, nên phải phơi ra cho tầng trên kiểm.
  const nvinfer1::Dims t_in = temporal_.dims(kTemporalIn);
  temporal_alloc_n_ = (t_in.nbDims == 3) ? static_cast<int>(t_in.d[1]) : num_frames_;

  ring_row_bytes_ = static_cast<size_t>(visual_dim_) *
                    dtype_bytes(backbone_.at(kBackboneOut).dtype);
  ring_bytes_ = 2 * static_cast<size_t>(num_frames_) * ring_row_bytes_;
  UAV_CUDA_CHECK(cudaMalloc(&ring_, ring_bytes_));
  reset_stream();
}

void Pipeline::reset_stream() {
  stream_count_ = 0;
  stride_counter_ = 0;
  if (ring_ != nullptr) {
    UAV_CUDA_CHECK(cudaMemsetAsync(ring_, 0, ring_bytes_, stream_));
  }
}

bool Pipeline::push_frame(const HostFrame& frame, bool measure,
                          float* out_feature) {
  if (cfg_.mode != Mode::kStream) {
    std::fprintf(stderr, "[Pipeline] push_frame() chỉ dùng ở chế độ kStream\n");
    std::exit(EXIT_FAILURE);
  }
  // Giống SlidingWindowBuffer.should_extract(): chỉ lấy mỗi `frame_stride` frame.
  const int stride = cfg_.frame_stride > 0 ? cfg_.frame_stride : 1;
  const bool take = (stride_counter_ % static_cast<size_t>(stride)) == 0;
  ++stride_counter_;
  if (!take) return false;
  measure_stream_ = measure;
  push_single(frame);
  if (out_feature != nullptr) {
    // Lấy lại đúng vector vừa ghi vào slot mới nhất.
    const size_t slot = (stream_count_ - 1) % static_cast<size_t>(num_frames_);
    UAV_CUDA_CHECK(cudaMemcpyAsync(
        out_feature,
        static_cast<uint8_t*>(ring_) + slot * ring_row_bytes_,
        static_cast<size_t>(visual_dim_) * sizeof(float),
        cudaMemcpyDeviceToHost, stream_));
    UAV_CUDA_CHECK(cudaStreamSynchronize(stream_));
  }
  return true;
}

// Tiền xử lý MỘT frame vào đầu vào backbone (batch 1). Dùng chung cho push_single
// và backbone_feature.
void Pipeline::preprocess_one(const HostFrame& f) {
  if (f.data == nullptr) {
    std::fprintf(stderr, "[Pipeline] frame có data null\n");
    std::exit(EXIT_FAILURE);
  }
  if (cfg_.mode != Mode::kStream) {
    std::fprintf(stderr, "[Pipeline] preprocess_one() chỉ dùng ở chế độ kStream\n");
    std::exit(EXIT_FAILURE);
  }
  const bool fast = (f.width == kSrcSize && f.height == kSrcSize);
  const size_t per_frame = fast ? (size_t)kDstPixels * 3
                                : (size_t)f.width * (size_t)f.height * 3;
  ensure_staging(per_frame);

  const size_t pitch = f.stride > 0 ? (size_t)f.stride : (size_t)f.width * 3;
  uint8_t* dst = static_cast<uint8_t*>(device_staging_);

  const bool measure = measure_stream_;
  if (measure) UAV_CUDA_CHECK(cudaEventRecord(ev_a_, stream_));

  if (fast) {
    copy_crop_256_to_staging(f.data, pitch, dst, stream_);
  } else {
    UAV_CUDA_CHECK(cudaMemcpy2DAsync(dst, (size_t)f.width * 3, f.data, pitch,
                                     (size_t)f.width * 3, (size_t)f.height,
                                     cudaMemcpyHostToDevice, stream_));
  }

  void* bb_in = backbone_.ptr(kBackboneIn);
  if (fast) {
    launch_crop_normalize(dst, 1, per_frame, pixel_type_, order_, bb_in, stream_);
  } else {
    ensure_taps(f.width, f.height);
    launch_resize_pil(dst, 1, per_frame, f.width, f.height, d_off_x_, d_wt_x_,
                      ksize_x_, d_off_y_, d_wt_y_, ksize_y_, resize_scratch_,
                      pixel_type_, order_, bb_in, stream_);
  }
  if (measure) {
    UAV_CUDA_CHECK(cudaEventRecord(ev_b_, stream_));
    UAV_CUDA_CHECK(cudaEventSynchronize(ev_b_));
    UAV_CUDA_CHECK(cudaEventElapsedTime(&timings_.upload_preprocess_ms, ev_a_,
                                        ev_b_));
    UAV_CUDA_CHECK(cudaEventRecord(ev_a_, stream_));
  }
}

void Pipeline::push_single(const HostFrame& f) {
  preprocess_one(f);
  const bool measure = measure_stream_;

  backbone_.infer(stream_);

  if (measure) {
    UAV_CUDA_CHECK(cudaEventRecord(ev_b_, stream_));
    UAV_CUDA_CHECK(cudaEventSynchronize(ev_b_));
    UAV_CUDA_CHECK(cudaEventElapsedTime(&timings_.backbone_ms, ev_a_, ev_b_));
  }

  // Ghi vào slot (p % N) và bản sao (p % N + N) để cửa sổ luôn liền mạch.
  const size_t slot = stream_count_ % static_cast<size_t>(num_frames_);
  uint8_t* base = static_cast<uint8_t*>(ring_);
  const void* feat = backbone_.ptr(kBackboneOut);
  UAV_CUDA_CHECK(cudaMemcpyAsync(base + slot * ring_row_bytes_, feat,
                                 ring_row_bytes_, cudaMemcpyDeviceToDevice,
                                 stream_));
  UAV_CUDA_CHECK(cudaMemcpyAsync(
      base + (slot + (size_t)num_frames_) * ring_row_bytes_, feat,
      ring_row_bytes_, cudaMemcpyDeviceToDevice, stream_));
  ++stream_count_;
}

void Pipeline::embed(float* out_host, bool measure) {
  if (cfg_.mode != Mode::kStream) {
    std::fprintf(stderr, "[Pipeline] embed() chỉ dùng ở chế độ kStream\n");
    std::exit(EXIT_FAILURE);
  }
  if (!ready()) {
    std::fprintf(stderr, "[Pipeline] embed() khi mới có %zu/%d frame\n",
                 stream_count_, num_frames_);
    std::exit(EXIT_FAILURE);
  }
  measure_stream_ = measure;

  // Cửa sổ N frame gần nhất bắt đầu tại (p - N) % N trong bộ đệm nhân đôi.
  const size_t w =
      (stream_count_ - static_cast<size_t>(num_frames_)) %
      static_cast<size_t>(num_frames_);
  temporal_.rebind(kTemporalIn,
                   static_cast<uint8_t*>(ring_) + w * ring_row_bytes_);

  if (measure) UAV_CUDA_CHECK(cudaEventRecord(ev_a_, stream_));
  temporal_.infer(stream_);
  if (measure) {
    UAV_CUDA_CHECK(cudaEventRecord(ev_b_, stream_));
    UAV_CUDA_CHECK(cudaEventSynchronize(ev_b_));
    UAV_CUDA_CHECK(cudaEventElapsedTime(&timings_.temporal_ms, ev_a_, ev_b_));
  }

  UAV_CUDA_CHECK(cudaMemcpyAsync(out_host, temporal_.ptr(kTemporalOut),
                                 static_cast<size_t>(embed_dim_) * sizeof(float),
                                 cudaMemcpyDeviceToHost, stream_));
  UAV_CUDA_CHECK(cudaStreamSynchronize(stream_));

  timings_.total_ms = measure ? timings_.upload_preprocess_ms +
                                    timings_.backbone_ms + timings_.temporal_ms
                              : 0.0f;
}

void Pipeline::backbone_feature(const HostFrame& f, float* out_host) {
  if (backbone_.at(kBackboneOut).dtype != nvinfer1::DataType::kFLOAT) {
    std::fprintf(stderr,
                 "[Pipeline] backbone_feature cần '%s' kiểu FLOAT để chép về host\n",
                 kBackboneOut);
    std::exit(EXIT_FAILURE);
  }
  preprocess_one(f);
  backbone_.infer(stream_);
  UAV_CUDA_CHECK(cudaMemcpyAsync(out_host, backbone_.ptr(kBackboneOut),
                                 static_cast<size_t>(visual_dim_) * sizeof(float),
                                 cudaMemcpyDeviceToHost, stream_));
  UAV_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void Pipeline::temporal_forward(const float* seq_host, int n, float* out_host,
                                bool measure) {
  if (cfg_.mode != Mode::kStream) {
    std::fprintf(stderr, "[Pipeline] temporal_forward() chỉ dùng ở chế độ kStream\n");
    std::exit(EXIT_FAILURE);
  }
  const int n_eng = temporal_alloc_n_ > 0 ? temporal_alloc_n_ : num_frames_;
  if (n <= 0 || n > n_eng) {
    std::fprintf(stderr, "[Pipeline] temporal_forward(): n=%d ngoài 1..%d\n", n,
                 n_eng);
    std::exit(EXIT_FAILURE);
  }
  if (n < n_eng && !warned_pad_) {
    warned_pad_ = true;
    std::fprintf(stderr,
                 "[Pipeline] CẢNH BÁO: engine temporal nhận N=%d nhưng chỉ có %d "
                 "frame -> đệm bằng cách lặp hàng cuối. Embedding KHÁC với "
                 "PyTorch chạy N=%d, ngưỡng đã calibrate sẽ không còn đúng. "
                 "Hãy build lại engine với N=%d.\n",
                 n_eng, n, n, n);
  }
  // Đệm bằng cách LẶP HÀNG CUỐI (bước thời gian = 0).
  const size_t row = static_cast<size_t>(visual_dim_);
  seq_pad_.resize(row * static_cast<size_t>(n_eng));
  for (int i = 0; i < n_eng; ++i) {
    const int src = (i < n) ? i : (n - 1);
    std::memcpy(&seq_pad_[(size_t)i * row], seq_host + (size_t)src * row,
                row * sizeof(float));
  }

  // Sau embed() con trỏ của seq_feats đang trỏ vào bộ đệm vòng, nên phải trả nó
  // về vùng nhớ riêng của engine trước khi ghi.
  void* arena_in = temporal_.ptr(kTemporalIn);
  temporal_.rebind(kTemporalIn, arena_in);
  UAV_CUDA_CHECK(cudaMemcpyAsync(arena_in, seq_pad_.data(),
                                 row * (size_t)n_eng * sizeof(float),
                                 cudaMemcpyHostToDevice, stream_));

  if (measure) UAV_CUDA_CHECK(cudaEventRecord(ev_a_, stream_));
  temporal_.infer(stream_);
  if (measure) {
    UAV_CUDA_CHECK(cudaEventRecord(ev_b_, stream_));
    UAV_CUDA_CHECK(cudaEventSynchronize(ev_b_));
    UAV_CUDA_CHECK(cudaEventElapsedTime(&timings_.temporal_ms, ev_a_, ev_b_));
  }

  UAV_CUDA_CHECK(cudaMemcpyAsync(out_host, temporal_.ptr(kTemporalOut),
                                 static_cast<size_t>(embed_dim_) * sizeof(float),
                                 cudaMemcpyDeviceToHost, stream_));
  UAV_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void Pipeline::synchronize() { UAV_CUDA_CHECK(cudaStreamSynchronize(stream_)); }

void Pipeline::print_info() const {
  auto dump = [](const TrtEngine& e, const char* tag) {
    std::printf("  [%s] %s\n", tag, e.plan_path().c_str());
    for (const auto& t : e.tensors()) {
      const char* dir =
          t.mode == nvinfer1::TensorIOMode::kINPUT ? "in " : "out";
      std::printf("    %s %-14s dims=[", dir, t.name.c_str());
      for (int i = 0; i < t.dims.nbDims; ++i) {
        std::printf("%s%d", i ? "," : "", (int)t.dims.d[i]);
      }
      std::printf("] %zu B%s\n", t.bytes,
                  t.external != nullptr ? "  (external, zero-copy)" : "");
    }
    std::printf("    arena=%zu B  profile=%d/%d\n", e.arena_bytes(),
                e.profile_index(), e.profile_count());
  };
  std::printf(
      "Pipeline: N=%d  max_clips=%d  backbone_batch=%d  visual_dim=%d  "
      "embed_dim=%d\n",
      num_frames_, max_clips_, backbone_batch(), visual_dim_, embed_dim_);
  dump(backbone_, "backbone");
  dump(temporal_, "temporal");
  std::printf("  intermediate (shared)=%zu B  staging=%zu B\n",
              intermediate_bytes_, staging_bytes_);
}

}  // namespace uav
