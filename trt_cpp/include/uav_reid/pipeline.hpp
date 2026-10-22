// Nối backbone.engine -> temporal.engine thành một đường trích embedding.
//
// Ba điểm tối ưu chính so với chạy Python:
//   1. Tiền xử lý chạy trên GPU (kernel), không dùng PIL/cv2 trên CPU.
//   2. Cố định shape một lần: backbone nhận (max_clips*N, 3, 224, 224) và
//      temporal nhận (max_clips, N, D). Không gọi lại setInputShape mỗi lần chạy.
//   3. Zero-copy giữa hai engine: cnn_feat của backbone và seq_feats của temporal
//      cùng layout bộ nhớ (B*N hàng x D) nên trỏ vào CÙNG một vùng device.
//
// Hai chế độ:
//   kClip   (mặc định) xử lý N frame một lượt, backbone batch = clips*N. Dùng cho
//           đánh giá offline và benchmark — thông lượng/frame tốt nhất.
//   kStream đẩy TỪNG frame qua backbone (batch 1) vào bộ đệm vòng N slot trên
//           device, temporal đọc cửa sổ N gần nhất. Khớp đường online của
//           infer.py (SlidingWindowBuffer): có embedding mới ngay mỗi frame
//           thay vì phải chờ đủ N frame.
#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "uav_reid/engine.hpp"
#include "uav_reid/preprocess.hpp"

namespace uav {

struct HostFrame {
  const uint8_t* data = nullptr;  // ảnh liên tục, 8-bit, 3 kênh
  int width = 0;
  int height = 0;
  int stride = 0;  // byte mỗi hàng; <=0 thì lấy width*3
};

// Chế độ chạy.
enum class Mode {
  kClip,   // N frame một lượt (offline / benchmark)
  kStream  // 1 frame mỗi bước, bộ đệm vòng N slot (online)
};

class Pipeline {
 public:
  struct Config {
    std::string backbone_plan;
    std::string temporal_plan;
    int num_frames = 12;
    int max_clips = 1;
    int device = 0;
    ChannelOrder order = ChannelOrder::kBGR;
    bool verbose_trt = false;
    Mode mode = Mode::kClip;
    // kStream: chỉ đẩy mỗi `frame_stride` frame vào bộ đệm, giống
    // SlidingWindowBuffer.should_extract() bên infer.py. 1 = mọi frame.
    int frame_stride = 1;
  };

  struct Timings {
    float upload_preprocess_ms = 0.0f;
    float backbone_ms = 0.0f;
    float temporal_ms = 0.0f;
    float total_ms = 0.0f;
  };

  explicit Pipeline(const Config& cfg);
  ~Pipeline();
  Pipeline(const Pipeline&) = delete;
  Pipeline& operator=(const Pipeline&) = delete;

  int num_frames() const { return num_frames_; }
  // N mà engine temporal đã được cấp phát (đọc từ engine, không phải từ config).
  int temporal_n() const { return temporal_alloc_n_; }
  int max_clips() const { return max_clips_; }
  int backbone_batch() const { return (cfg_.mode == Mode::kStream)
                                          ? 1
                                          : max_clips_ * num_frames_; }
  Mode mode() const { return cfg_.mode; }
  int visual_dim() const { return visual_dim_; }
  int embed_dim() const { return embed_dim_; }
  const Timings& last_timings() const { return timings_; }

  // `frames` xếp theo [clip][frame], kích thước clips*num_frames.
  // `out_host` nhận clips*embed_dim float, mỗi hàng đã L2-chuẩn-hoá (engine lo).
  // Nếu clips < max_clips thì các slot dư lặp lại clip cuối, kết quả dư bị bỏ.
  //
  // `measure=false` (mặc định) chỉ chạy một lần đồng bộ ở cuối — nhanh nhất.
  // `measure=true` chèn mốc thời gian CUDA giữa 3 chặng, tốn thêm 3 lần
  // đồng bộ mỗi vòng; chỉ dùng khi benchmark.
  void run(const std::vector<HostFrame>& frames, int clips, float* out_host,
           bool measure = false);

  // --- chế độ kStream -----------------------------------------------------
  // Đẩy một frame (hoặc bỏ qua nếu chưa tới nhịp `frame_stride`).
  // Trả về true nếu frame thực sự được đưa vào bộ đệm.
  // `out_feature` (khác null) nhận vector ĐẶC TRƯNG của frame đó (visual_dim float).
  bool push_frame(const HostFrame& frame, bool measure = false,
                  float* out_feature = nullptr);

  // Đủ N slot để chạy temporal chưa.
  bool ready() const { return stream_count_ >= (size_t)num_frames_; }

  // Chạy temporal trên cửa sổ N gần nhất, ghi embed_dim float vào `out_host`.
  // Yêu cầu ready(); chưa đủ thì ném lỗi ra stderr và dừng.
  void embed(float* out_host, bool measure = false);

  // Xoá bộ đệm (bắt đầu track mới).
  void reset_stream();

  // Số frame đã đẩy vào bộ đệm kể từ reset.
  size_t stream_count() const { return stream_count_; }

  // --- dùng cho binding Python -------------------------------------------
  // Trích feature MỘT frame, không đụng bộ đệm vòng.
  void backbone_feature(const HostFrame& frame, float* out_host);

  // Chạy temporal trên cửa sổ n hàng (n <= N), tự đệm hàng cuối cho đủ N.
  void temporal_forward(const float* seq_host, int n, float* out_host,
                        bool measure = false);

  void synchronize();
  void print_info() const;

 private:
  void prepare_engines();
  void prepare_buffers();
  void ensure_staging(size_t bytes_per_frame);
  void ensure_taps(int src_w, int src_h);
  void prepare_stream();
  void preprocess_one(const HostFrame& frame);
  void push_single(const HostFrame& frame);

  Config cfg_;
  TrtEngine backbone_;
  TrtEngine temporal_;

  int num_frames_ = 0;
  int max_clips_ = 0;
  int visual_dim_ = 0;
  int embed_dim_ = 0;

  PixelType pixel_type_ = PixelType::kFloat32;
  ChannelOrder order_ = ChannelOrder::kBGR;

  cudaStream_t stream_ = nullptr;
  cudaEvent_t ev_a_ = nullptr;
  cudaEvent_t ev_b_ = nullptr;

  void* intermediate_ = nullptr;   // chia sẻ: cnn_feat <-> seq_feats
  size_t intermediate_bytes_ = 0;

  void* device_staging_ = nullptr;  // ảnh uint8 sau khi upload
  size_t staging_bytes_ = 0;

  // Bảng trọng số Pillow cho đường resize, dựng lại khi đổi kích thước nguồn.
  ResizeTaps taps_x_;
  ResizeTaps taps_y_;
  int taps_w_ = -1;
  int taps_h_ = -1;
  int* d_off_x_ = nullptr;
  float* d_wt_x_ = nullptr;
  int* d_off_y_ = nullptr;
  float* d_wt_y_ = nullptr;
  int ksize_x_ = 0;
  int ksize_y_ = 0;
  uint8_t* resize_scratch_ = nullptr;
  size_t resize_scratch_bytes_ = 0;

  // --- bộ đệm vòng (kStream) ---
  // Cấp 2*N slot: frame thứ p ghi vào slot (p % N) và (p % N + N). Nhờ bản sao
  // thứ hai, cửa sổ N frame gần nhất LUÔN nằm liền mạch tại
  // [(p - N + 1) % N, +N) nên temporal đọc thẳng, không phải chép đảo thứ tự.
  void* ring_ = nullptr;
  size_t ring_bytes_ = 0;
  size_t ring_row_bytes_ = 0;   // D * sizeof(dtype)
  size_t stream_count_ = 0;     // tổng số frame đã đẩy (tăng vô hạn)
  size_t stride_counter_ = 0;
  bool measure_stream_ = false;
  std::vector<float> seq_pad_;  // đệm cửa sổ lên đúng N hàng cho temporal_forward
  int temporal_alloc_n_ = 0;    // N của engine temporal (sau allocate)
  bool warned_pad_ = false;     // chỉ cảnh báo đệm một lần

  Timings timings_;
};

}  // namespace uav
