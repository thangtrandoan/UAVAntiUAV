// C ABI cho Python (nạp bằng ctypes trong uav_reid_engine.py).
//
// Vì sao không dùng pybind11: máy đích không có pybind11, torch bản này cũng
// không kèm header của nó; còn ctypes là thư viện chuẩn của Python. Cùng một
// kiến trúc (engine C++ gọi từ infer.py), không thêm phụ thuộc, không phải khớp
// ABI Python.
//
// LƯU Ý: lõi (engine.cpp/pipeline.cpp) gọi std::exit khi cấu hình sai (engine
// không khớp shape, thiếu tensor, sai N...). Các lỗi đó là lỗi CẤU HÌNH lúc mở,
// không phải lỗi dữ liệu, nên sẽ dừng tiến trình kèm thông báo rõ ràng. Hàm nào
// cũng trả void/int nhưng không có mã lỗi vì lý do đó.
#include <vector>

#include "uav_reid/pipeline.hpp"

namespace {

struct UavEngine {
  explicit UavEngine(const uav::Pipeline::Config& c) : pipe(c) {}
  uav::Pipeline pipe;
};

inline UavEngine* eng(void* h) { return static_cast<UavEngine*>(h); }

inline uav::HostFrame frame(const unsigned char* img, int w, int h, int stride) {
  uav::HostFrame f;
  f.data = img;
  f.width = w;
  f.height = h;
  f.stride = stride > 0 ? stride : w * 3;
  return f;
}

}  // namespace

extern "C" {

// Mở engine. `bgr` = 1 nếu ảnh nguồn là BGR (cv2.imread / cv2 VideoCapture).
void* uav_create(const char* backbone_plan, const char* temporal_plan,
                 int num_frames, int device, int bgr) {
  uav::Pipeline::Config cfg;
  cfg.backbone_plan = backbone_plan;
  cfg.temporal_plan = temporal_plan;
  cfg.num_frames = num_frames;
  cfg.max_clips = 1;
  cfg.device = device;
  cfg.order = (bgr != 0) ? uav::ChannelOrder::kBGR : uav::ChannelOrder::kRGB;
  // Binding luôn dùng chế độ kStream: backbone batch 1 + bộ đệm vòng N slot.
  cfg.mode = uav::Mode::kStream;
  return new UavEngine(cfg);
}

void uav_destroy(void* h) { delete eng(h); }

int uav_visual_dim(void* h) { return eng(h)->pipe.visual_dim(); }
int uav_embed_dim(void* h) { return eng(h)->pipe.embed_dim(); }
int uav_num_frames(void* h) { return eng(h)->pipe.num_frames(); }

// N mà engine temporal đã cấp phát. Khác uav_num_frames() nghĩa là engine được
// build với N khác config -> embedding sẽ lệch, tầng Python phải kiểm.
int uav_temporal_n(void* h) { return eng(h)->pipe.temporal_n(); }

// Trích feature một frame: ảnh HWC uint8 -> visual_dim float32.
void uav_backbone(void* h, const unsigned char* img, int width, int height,
                  int stride, float* out) {
  eng(h)->pipe.backbone_feature(frame(img, width, height, stride), out);
}

// Chạy temporal trên cửa sổ `n` hàng (n <= N). Tự đệm hàng cuối cho đủ N.
void uav_temporal(void* h, const float* seq, int n, float* out) {
  eng(h)->pipe.temporal_forward(seq, n, out);
}

// Đẩy một frame vào bộ đệm vòng và trả luôn feature của frame đó.
// Trả 1 nếu frame được nhận, 0 nếu bị bỏ do chưa tới nhịp stride.
int uav_push(void* h, const unsigned char* img, int width, int height,
             int stride, float* out_feature) {
  return eng(h)->pipe.push_frame(frame(img, width, height, stride), false,
                                 out_feature)
             ? 1
             : 0;
}

// Chạy temporal trên cửa sổ N frame gần nhất trong bộ đệm vòng.
void uav_embed(void* h, float* out) { eng(h)->pipe.embed(out); }

int uav_ready(void* h) { return eng(h)->pipe.ready() ? 1 : 0; }

void uav_reset(void* h) { eng(h)->pipe.reset_stream(); }

}  // extern "C"
