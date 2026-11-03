// Bản C++ của infer.py: video vào -> video ra (có overlay) + file txt ghi chi tiết.
//
// Mục đích: đo tốc độ TRIỂN KHAI THẬT. Toàn bộ đường chạy nằm trong C++, không gọi
// sang Python, nên số FPS ở đây là số của thiết bị.
//
// Phần port nguyên từ infer.py:
//   SlidingWindowBuffer  -> Window            (infer.py:58)
//   TwoTierMemoryBank    -> Bank              (infer.py:149)
//   compute_fused_vector -> fused_bundle()    (infer.py:122)
//   SeqReIDPipeline      -> Runner            (infer.py:244)
//
// KHÁC infer.py (vì engine TensorRT chỉ xuất `reid_embed`):
//   - Không có `temporal_token` và `raw_feat` (đầu vào bnneck) -> không có cột PRE-BN,
//     không có `temporal_score`, không dùng được `fine_space: pre_bn`. Chỉ `fused`.
//     Muốn có: export_onnx.py phải xuất thêm 2 output rồi build engine mới.
//   - Cửa sổ chưa đủ N hàng (cập nhật bank "last-moment" trước khi LOST) phải đệm
//     bằng cách lặp hàng cuối, vì engine đã unroll đúng N hàng. PyTorch cắt được
//     pos_embed nên không cần đệm -> hai bên lệch nhau đúng ở trường hợp này.
//
// File ghi ra <out-dir>/:
//   output.mp4     video overlay: state, bank, soft/hard fill, điểm, bbox, frame
//   frames.csv     mọi thông số từng frame (thứ không vừa trên video)
//   events.txt     chuyển trạng thái + quyết định, kèm frame và lý do
//   summary.txt    tổng hợp: latency, false alarm, FPS, thời gian từng chặng
//   config_used.txt  tham số thực tế đã dùng
//
// Chạy:
//   uav_reid_infer --seq-dir data/UAV-Anti-UAV/Test/UAV-Anti-UAV_Test_000059
//       --backbone v1_backbone.engine --temporal v1_n8_temporal.engine
//       --stride 4 --num-frames 8 --reid-thr 0.75

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "uav_reid/pipeline.hpp"

namespace {

using Clock = std::chrono::steady_clock;
using Vec = std::vector<float>;

double ms_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

Vec l2(const Vec& v) {
  double s = 0.0;
  for (float x : v) s += static_cast<double>(x) * x;
  const float n = s > 0 ? static_cast<float>(std::sqrt(s)) : 0.0f;
  Vec out(v.size());
  if (n > 0) {
    for (size_t i = 0; i < v.size(); ++i) out[i] = v[i] / n;
  }
  return out;
}

float dot(const Vec& a, const Vec& b) {
  double s = 0.0;
  const size_t n = std::min(a.size(), b.size());
  for (size_t i = 0; i < n; ++i) s += static_cast<double>(a[i]) * b[i];
  return static_cast<float>(s);
}

std::string f3(float v) {
  if (!std::isfinite(v)) return "";
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.3f", v);
  return buf;
}

// ---------------------------------------------------------------------------
// Tham số: config (nếu có) làm nền, CLI đè lên.
// ---------------------------------------------------------------------------
struct Args {
  std::string seq_dir, config, backbone, temporal, out_dir;
  std::string fine_space = "fused";
  int num_frames = 0;  // 0 = lấy N từ engine
  int stride = 4;
  int device = 0;
  int limit = 0;
  int warmup = 0;
  int max_anchor = 10;
  int max_recent = 30;
  int hijack_check_count = 5;
  int gap_tolerance = 2;
  float soft_lock_threshold = 0.50f;
  float reid_threshold = 0.75f;
  float hijack_threshold = 0.40f;
  double update_interval = 2.0;
  double padding = 0.2;
  bool dry_run = false;
  bool no_video_out = false;
  bool quiet = false;
};

void usage() {
  std::printf(
      "uav_reid_infer --seq-dir DIR --backbone PATH --temporal PATH [tuỳ chọn]\n"
      "\n"
      "  --seq-dir DIR        thư mục sequence (mp4 + groundtruth_rect.txt + absent.txt)\n"
      "  --config PATH        đọc các khoá trong block `infer:` của file yaml\n"
      "  --backbone PATH      engine backbone\n"
      "  --temporal PATH      engine temporal (N phải khớp --num-frames)\n"
      "  --out-dir DIR        mặc định infer_cpp/<ten sequence>\n"
      "  --num-frames N       0 = lấy N từ engine\n"
      "  --stride K           bước lấy mẫu cho HARD LOCK (mặc định 4)\n"
      "  --padding F          bbox_padding (mặc định 0.2)\n"
      "  --soft-lock-thr F    ngưỡng soft lock (mặc định 0.50)\n"
      "  --reid-thr F         ngưỡng HARD LOCK (mặc định 0.75)\n"
      "  --hijack-thr F       ngưỡng anti-hijack (mặc định 0.40)\n"
      "  --hijack-count N     số lần kiểm anti-hijack sau khi lock (mặc định 5)\n"
      "  --gap-tolerance N    số frame vắng được bỏ qua trong T2_SEARCH (mặc định 2)\n"
      "  --update-interval S  giây video giữa hai lần cập nhật bank (mặc định 2.0)\n"
      "  --max-anchor N       (mặc định 10)\n"
      "  --max-recent N       (mặc định 30)\n"
      "  --limit N            chỉ xử lý N frame\n"
      "  --warmup N           bỏ N frame đầu khỏi thống kê thời gian\n"
      "  --device D           GPU index (mặc định 0)\n"
      "  --no-video-out       không ghi mp4, chỉ ghi txt\n"
      "  --dry-run            chỉ kiểm dữ liệu, không dựng engine\n"
      "  --quiet              không in từng frame ra màn hình\n");
}

// YAML tối giản: chỉ lấy khoá phẳng nằm trong block `infer:`.
void load_config(Args& a, const std::string& path) {
  std::ifstream f(path);
  if (!f) {
    std::fprintf(stderr, "[infer] không đọc được config: %s\n", path.c_str());
    std::exit(EXIT_FAILURE);
  }
  std::string line, section;
  while (std::getline(f, line)) {
    const size_t hash = line.find('#');
    if (hash != std::string::npos) line = line.substr(0, hash);
    if (line.find_first_not_of(" \t\r") == std::string::npos) continue;

    const bool top_level = line[0] != ' ' && line[0] != '\t';
    const size_t colon = line.find(':');
    if (colon == std::string::npos) continue;

    std::string key = line.substr(0, colon);
    key.erase(std::remove_if(key.begin(), key.end(),
                             [](unsigned char c) { return std::isspace(c) != 0; }),
              key.end());
    // Dòng tiêu đề section ("infer:") có giá trị RỖNG, nên phải ghi nhận section
    // TRƯỚC khi kiểm tra giá trị rỗng — nếu không sẽ bỏ qua toàn bộ section.
    if (top_level) {
      section = key;
      continue;
    }
    if (section != "infer") continue;

    std::string val = line.substr(colon + 1);
    val.erase(std::remove_if(val.begin(), val.end(),
                             [](unsigned char c) { return c == '"' || c == '\'' || c == '\r'; }),
              val.end());
    const size_t first = val.find_first_not_of(" \t");
    if (first == std::string::npos) continue;
    val = val.substr(first);
    while (!val.empty() && (val.back() == ' ' || val.back() == '\t')) val.pop_back();

    const double num = std::atof(val.c_str());
    if (key == "num_frames") a.num_frames = std::atoi(val.c_str());
    else if (key == "stride") a.stride = std::atoi(val.c_str());
    else if (key == "bbox_padding") a.padding = num;
    else if (key == "soft_lock_threshold") a.soft_lock_threshold = static_cast<float>(num);
    else if (key == "reid_threshold") a.reid_threshold = static_cast<float>(num);
    else if (key == "hijack_threshold") a.hijack_threshold = static_cast<float>(num);
    else if (key == "hijack_check_count") a.hijack_check_count = std::atoi(val.c_str());
    else if (key == "update_interval_sec") a.update_interval = num;
    else if (key == "t2_search_gap_tolerance") a.gap_tolerance = std::atoi(val.c_str());
    else if (key == "max_anchor_size") a.max_anchor = std::atoi(val.c_str());
    else if (key == "max_recent_size") a.max_recent = std::atoi(val.c_str());
    else if (key == "fine_space") a.fine_space = val;
  }
}

const char* next_arg(int argc, char** argv, int& i, const std::string& flag) {
  if (i + 1 >= argc) {
    std::fprintf(stderr, "[infer] thiếu giá trị cho %s\n", flag.c_str());
    std::exit(EXIT_FAILURE);
  }
  return argv[++i];
}

Args parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {  // lượt 1: config làm nền
    if (std::string(argv[i]) == "--config") {
      a.config = next_arg(argc, argv, i, "--config");
      load_config(a, a.config);
    }
  }
  for (int i = 1; i < argc; ++i) {  // lượt 2: CLI đè
    const std::string k = argv[i];
    if (k == "-h" || k == "--help") {
      usage();
      std::exit(EXIT_SUCCESS);
    } else if (k == "--config") {
      ++i;
    } else if (k == "--seq-dir") a.seq_dir = next_arg(argc, argv, i, k);
    else if (k == "--backbone") a.backbone = next_arg(argc, argv, i, k);
    else if (k == "--temporal") a.temporal = next_arg(argc, argv, i, k);
    else if (k == "--out-dir") a.out_dir = next_arg(argc, argv, i, k);
    else if (k == "--num-frames") a.num_frames = std::atoi(next_arg(argc, argv, i, k));
    else if (k == "--stride") a.stride = std::max(1, std::atoi(next_arg(argc, argv, i, k)));
    else if (k == "--device") a.device = std::atoi(next_arg(argc, argv, i, k));
    else if (k == "--limit") a.limit = std::atoi(next_arg(argc, argv, i, k));
    else if (k == "--warmup") a.warmup = std::max(0, std::atoi(next_arg(argc, argv, i, k)));
    else if (k == "--max-anchor") a.max_anchor = std::atoi(next_arg(argc, argv, i, k));
    else if (k == "--max-recent") a.max_recent = std::atoi(next_arg(argc, argv, i, k));
    else if (k == "--hijack-count") a.hijack_check_count = std::atoi(next_arg(argc, argv, i, k));
    else if (k == "--gap-tolerance") a.gap_tolerance = std::atoi(next_arg(argc, argv, i, k));
    else if (k == "--padding") a.padding = std::atof(next_arg(argc, argv, i, k));
    else if (k == "--soft-lock-thr") a.soft_lock_threshold = static_cast<float>(std::atof(next_arg(argc, argv, i, k)));
    else if (k == "--reid-thr") a.reid_threshold = static_cast<float>(std::atof(next_arg(argc, argv, i, k)));
    else if (k == "--hijack-thr") a.hijack_threshold = static_cast<float>(std::atof(next_arg(argc, argv, i, k)));
    else if (k == "--update-interval") a.update_interval = std::atof(next_arg(argc, argv, i, k));
    else if (k == "--no-video-out") a.no_video_out = true;
    else if (k == "--dry-run") a.dry_run = true;
    else if (k == "--quiet") a.quiet = true;
    else {
      std::fprintf(stderr, "[infer] cờ lạ: %s (xem --help)\n", k.c_str());
      std::exit(EXIT_FAILURE);
    }
  }
  if (a.seq_dir.empty()) {
    std::fprintf(stderr, "[infer] cần --seq-dir\n");
    std::exit(EXIT_FAILURE);
  }
  if (!a.dry_run && (a.backbone.empty() || a.temporal.empty())) {
    std::fprintf(stderr, "[infer] cần --backbone và --temporal\n");
    std::exit(EXIT_FAILURE);
  }
  return a;
}

// ---------------------------------------------------------------------------
// Dữ liệu đầu vào
// ---------------------------------------------------------------------------
struct BBox {
  int x = 0, y = 0, w = 0, h = 0;
  bool valid() const { return w > 0 && h > 0; }
};

std::vector<BBox> load_gt(const std::string& path) {
  std::vector<BBox> out;
  std::ifstream f(path);
  if (!f) return out;
  std::string line;
  while (std::getline(f, line)) {
    for (char& c : line) {
      if (!std::isdigit(static_cast<unsigned char>(c)) && c != '-') c = ' ';
    }
    std::istringstream ss(line);
    BBox b;
    if (ss >> b.x >> b.y >> b.w >> b.h) out.push_back(b);
    else out.push_back(BBox{});
  }
  return out;
}

std::vector<int> load_absent(const std::string& path) {
  std::vector<int> out;
  std::ifstream f(path);
  if (!f) return out;
  std::string line;
  while (std::getline(f, line)) {
    const size_t first = line.find_first_not_of(" \t\r");
    if (first == std::string::npos) continue;
    const std::string t = line.substr(first);
    const size_t last = t.find_last_not_of(" \t\r");
    const std::string v = t.substr(0, last + 1);
    if (v.find_first_not_of("0123456789") == std::string::npos) out.push_back(std::atoi(v.c_str()));
  }
  return out;
}

// Y hệt crop_and_pad của infer.py:230.
cv::Rect crop_rect(int W, int H, const BBox& b, double padding) {
  const int pad_w = static_cast<int>(b.w * padding);
  const int pad_h = static_cast<int>(b.h * padding);
  const int x1 = std::max(0, b.x - pad_w);
  const int y1 = std::max(0, b.y - pad_h);
  const int x2 = std::min(W, b.x + b.w + pad_w);
  const int y2 = std::min(H, b.y + b.h + pad_h);
  if (x2 <= x1 || y2 <= y1) return cv::Rect();
  return cv::Rect(x1, y1, x2 - x1, y2 - y1);
}

// Y hệt compute_sharpness của infer.py:54.
double sharpness(const cv::Mat& bgr) {
  cv::Mat gray, lap;
  cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
  cv::Laplacian(gray, lap, CV_64F);
  cv::Scalar mean, sd;
  cv::meanStdDev(lap, mean, sd);
  return sd[0] * sd[0];
}

// ---------------------------------------------------------------------------
// Cửa sổ trượt — port SlidingWindowBuffer (infer.py:58)
// ---------------------------------------------------------------------------
struct Window {
  int window_size = 0;
  int stride = 1;
  int counter = 0;
  std::deque<Vec> feats;
  std::deque<float> sharp;

  Window() = default;
  Window(int n, int s) : window_size(n), stride(s) {}

  bool should_extract() {
    const bool r = (counter % stride == 0);
    ++counter;
    return r;
  }
  void add(const Vec& f, float s) {
    feats.push_back(f);
    sharp.push_back(s);
    while (static_cast<int>(feats.size()) > window_size) {
      feats.pop_front();
      sharp.pop_front();
    }
  }
  bool ready() const { return static_cast<int>(feats.size()) >= window_size; }
  int size() const { return static_cast<int>(feats.size()); }
};

// ---------------------------------------------------------------------------
// Memory Bank 2 tầng — port TwoTierMemoryBank (infer.py:149)
// ---------------------------------------------------------------------------
struct Bundle {
  Vec visual_mean;   // trung bình có trọng số sharpness (coarse score)
  Vec visual_plain;  // trung bình thường (input của head lúc train)
  Vec fused;         // đầu ra engine, đã L2 sẵn
};

struct Entry {
  Vec visual, visual_plain, fused;
};

struct Bank {
  int max_anchor = 10;
  int max_recent = 30;
  std::deque<Entry> anchor, recent;

  Bank() = default;
  Bank(int a, int r) : max_anchor(a), max_recent(r) {}

  void add(const Bundle& b) {
    Entry e{l2(b.visual_mean), l2(b.visual_plain), l2(b.fused)};
    if (static_cast<int>(anchor.size()) < max_anchor) {
      anchor.push_back(std::move(e));
      return;
    }
    recent.push_back(std::move(e));
    while (static_cast<int>(recent.size()) > max_recent) recent.pop_front();
  }

  // which: 0 = visual (coarse), 1 = visual_plain, 2 = fused (fine)
  float max_sim(const Vec& query, int which) const {
    const Vec q = l2(query);
    float best = 0.0f;
    for (const auto* bank : {&anchor, &recent}) {
      for (const Entry& e : *bank) {
        const Vec& ref = which == 0 ? e.visual : (which == 1 ? e.visual_plain : e.fused);
        if (ref.size() != q.size()) continue;
        best = std::max(best, dot(q, ref));
      }
    }
    return best;
  }
  bool empty() const { return anchor.empty() && recent.empty(); }
  std::string size_info() const {
    return "Anchor: " + std::to_string(anchor.size()) + "/" + std::to_string(max_anchor) +
           " | Recent: " + std::to_string(recent.size()) + "/" + std::to_string(max_recent);
  }
};

enum class State { kT0Init, kT1Lost, kT2Search, kT3Verified };

const char* state_name(State s) {
  switch (s) {
    case State::kT0Init: return "T0_INIT";
    case State::kT1Lost: return "T1_LOST";
    case State::kT2Search: return "T2_SEARCH";
    case State::kT3Verified: return "T3_VERIFIED";
  }
  return "?";
}

// ---------------------------------------------------------------------------
// Đường chạy — port SeqReIDPipeline.process_frame (infer.py:376)
// ---------------------------------------------------------------------------
class Runner {
 public:
  Runner(const Args& a, uav::Pipeline& pipe, std::ofstream& frames, std::ofstream& events)
      : a_(a),
        pipe_(pipe),
        frames_(frames),
        events_(events),
        n_(pipe.num_frames()),
        dim_(pipe.visual_dim()),
        edim_(pipe.embed_dim()),
        bank_(a.max_anchor, a.max_recent),
        sliding_(n_, a.stride),
        soft_(n_, 1),
        hard_(n_, a.stride),
        seq_(static_cast<size_t>(n_) * static_cast<size_t>(dim_)),
        feat_(static_cast<size_t>(dim_)),
        emb_(static_cast<size_t>(edim_)) {
  }

  void process(const cv::Mat& frame, const BBox& bbox, bool is_absent, int idx,
               double video_time);

  State state() const { return state_; }
  const Bank& bank() const { return bank_; }
  int soft_size() const { return soft_.size(); }
  int hard_size() const { return hard_.size(); }
  int hijack_remaining() const { return hijack_remaining_; }
  int false_alarms() const { return false_alarms_; }
  int hard_lock_count() const { return hard_lock_count_; }
  int num_frames() const { return n_; }
  int hard_lock_capacity() const { return (n_ - 1) * a_.stride + 1; }
  const std::vector<double>& latencies() const { return latency_; }
  const std::vector<double>& backbone_ms() const { return backbone_ms_; }
  const std::vector<double>& temporal_ms() const { return temporal_ms_; }
  std::vector<double> frame_ms;  // thời gian xử lý mỗi frame (không gồm ghi video)

  // Giá trị của frame vừa xử lý, để main() vẽ overlay.
  float last_coarse = NAN, last_soft = NAN, last_fine = NAN;

 private:
  Bundle fused_bundle(Window& w);
  void log_event(int idx, const std::string& what);
  void to_lost(int idx, const char* why);
  void write_row(int idx, double video_time, const BBox& bbox, bool present, int crop_w,
                 int crop_h, double sharp, float coarse, float soft, float fine,
                 double ms_bb, double ms_tp, const std::string& ev);

  const Args& a_;
  uav::Pipeline& pipe_;
  std::ofstream& frames_;
  std::ofstream& events_;
  int n_;
  int dim_;
  int edim_;
  Bank bank_;
  Window sliding_, soft_, hard_;

  State state_ = State::kT0Init;
  bool soft_announced_ = false;
  bool soft_passed_ = false;
  int absent_streak_ = 0;
  int hijack_remaining_ = 0;
  int reappeared_idx_ = -1;
  int false_alarms_ = 0;
  int hard_lock_count_ = 0;
  double last_update_ = 0.0;
  std::vector<double> latency_;
  std::vector<double> backbone_ms_, temporal_ms_;

  Vec seq_, feat_, emb_;
};

Bundle Runner::fused_bundle(Window& w) {
  const int n = w.size();
  std::fill(seq_.begin(), seq_.end(), 0.0f);
  for (int i = 0; i < n; ++i) {
    std::memcpy(&seq_[static_cast<size_t>(i) * dim_], w.feats[i].data(),
                sizeof(float) * static_cast<size_t>(dim_));
  }

  Bundle b;
  b.visual_plain.assign(static_cast<size_t>(dim_), 0.0f);
  b.visual_mean.assign(static_cast<size_t>(dim_), 0.0f);

  double wsum = 0.0;
  for (int i = 0; i < n; ++i) wsum += w.sharp[i];
  for (int i = 0; i < n; ++i) {
    const float ww =
        wsum > 0 ? static_cast<float>(w.sharp[i] / wsum) : 1.0f / static_cast<float>(n);
    const Vec& f = w.feats[i];
    for (int d = 0; d < dim_; ++d) {
      b.visual_plain[d] += f[d] / static_cast<float>(n);
      b.visual_mean[d] += ww * f[d];
    }
  }

  const auto t0 = Clock::now();
  pipe_.temporal_forward(seq_.data(), n, emb_.data(), false);
  temporal_ms_.push_back(ms_since(t0));
  b.fused = emb_;
  return b;
}

void Runner::log_event(int idx, const std::string& what) {
  events_ << "[f " << idx << "] " << what << "\n";
  events_.flush();
  if (!a_.quiet) std::printf("[%d] %s\n", idx, what.c_str());
}

void Runner::to_lost(int idx, const char* why) {
  const std::string from = state_name(state_);
  state_ = State::kT1Lost;
  hijack_remaining_ = 0;
  log_event(idx, std::string(from) + " -> T1_LOST (" + why + ")");
}

void Runner::write_row(int idx, double video_time, const BBox& bbox, bool present, int crop_w,
                       int crop_h, double sharp, float coarse, float soft, float fine,
                       double ms_bb, double ms_tp, const std::string& ev) {
  frames_ << idx << ',' << std::fixed << video_time << ',' << state_name(state_) << ','
          << (present ? 1 : 0) << ',' << bbox.x << ',' << bbox.y << ',' << bbox.w << ','
          << bbox.h << ',' << crop_w << ',' << crop_h << ',';
  if (std::isfinite(sharp)) frames_ << sharp;
  frames_ << ',' << f3(coarse) << ',' << f3(soft) << ',' << f3(fine) << ','
          << soft_.size() << ',' << hard_.size() << ',' << bank_.anchor.size() << ','
          << bank_.recent.size() << ',' << hijack_remaining_ << ',';
  if (std::isfinite(ms_bb)) frames_ << ms_bb;
  frames_ << ',';
  if (std::isfinite(ms_tp)) frames_ << ms_tp;
  frames_ << ',' << ev << "\n";
}

void Runner::process(const cv::Mat& frame, const BBox& bbox, bool is_absent, int idx,
                     double video_time) {
  const bool valid = bbox.valid();
  const auto t_frame = Clock::now();

  int crop_w = 0, crop_h = 0;
  double sharp = NAN, ms_bb = NAN, ms_tp = NAN;
  float coarse = NAN, soft_s = NAN, fine = NAN;
  std::string ev;
  last_coarse = last_soft = last_fine = NAN;

  do {
    // ---------------- đang TRACKING ----------------
    if (state_ == State::kT0Init || state_ == State::kT3Verified) {
      if (is_absent || !valid) {
        if (sliding_.size() > 0) {
          const Bundle b = fused_bundle(sliding_);
          bank_.add(b);
          ev = "last-moment bank update before LOST";
          log_event(idx, "Last-moment Memory Bank update before LOST. Bank: " + bank_.size_info());
        }
        to_lost(idx, "target vang mat");
        break;
      }

      const cv::Rect rc = crop_rect(frame.cols, frame.rows, bbox, a_.padding);
      if (rc.width > 0 && sliding_.should_extract()) {
        const cv::Mat crop = frame(rc);
        crop_w = crop.cols;
        crop_h = crop.rows;
        sharp = sharpness(crop);
        const uav::HostFrame hf{crop.data, crop.cols, crop.rows, static_cast<int>(crop.step)};
        const auto t0 = Clock::now();
        pipe_.push_frame(hf, false, feat_.data());
        ms_bb = ms_since(t0);
        backbone_ms_.push_back(ms_bb);
        sliding_.add(feat_, static_cast<float>(sharp));
      }

      if (sliding_.ready() && (video_time - last_update_) >= a_.update_interval) {
        const Bundle b = fused_bundle(sliding_);
        ms_tp = temporal_ms_.back();

        if (state_ == State::kT3Verified && hijack_remaining_ > 0) {
          const float hs = bank_.max_sim(b.fused, 2);
          --hijack_remaining_;
          log_event(idx, "Anti-Hijack check #" +
                             std::to_string(a_.hijack_check_count - hijack_remaining_) +
                             ": fused=" + f3(hs));
          if (hs < a_.hijack_threshold) {
            to_lost(idx, "HIJACK: fused duoi nguong");
            ev = "hijack detected";
            break;
          }
        }

        bank_.add(b);
        last_update_ = video_time;
        log_event(idx, "bank update. " + bank_.size_info());
        ev = "bank update";
      }
      break;
    }

    // ---------------- vừa mất target ----------------
    if (state_ == State::kT1Lost) {
      if (!is_absent && valid) {
        state_ = State::kT2Search;
        reappeared_idx_ = idx;
        soft_ = Window(n_, 1);
        hard_ = Window(n_, a_.stride);
        soft_announced_ = false;
        soft_passed_ = false;
        absent_streak_ = 0;
        log_event(idx, "T1_LOST -> T2_SEARCH (UAV tai xuat theo GT)");
        ev = "reappear";
      }
      break;
    }

    // ---------------- đang tìm lại ----------------
    if (state_ == State::kT2Search) {
      if (is_absent || !valid) {
        ++absent_streak_;
        if (absent_streak_ <= a_.gap_tolerance) {
          log_event(idx, "T2_SEARCH: vang " + std::to_string(absent_streak_) + "/" +
                             std::to_string(a_.gap_tolerance) + " -> BO QUA, giu " +
                             std::to_string(soft_.size()) + "/" + std::to_string(n_) + " feature");
          ev = "gap skip";
          break;
        }
        to_lost(idx, "vang qua gap_tolerance");
        ev = "gap exceeded";
        break;
      }
      absent_streak_ = 0;

      const cv::Rect rc = crop_rect(frame.cols, frame.rows, bbox, a_.padding);
      if (rc.width <= 0) break;

      const cv::Mat crop = frame(rc);
      crop_w = crop.cols;
      crop_h = crop.rows;
      sharp = sharpness(crop);
      const uav::HostFrame hf{crop.data, crop.cols, crop.rows, static_cast<int>(crop.step)};
      const auto t0 = Clock::now();
      pipe_.push_frame(hf, false, feat_.data());
      ms_bb = ms_since(t0);
      backbone_ms_.push_back(ms_bb);

      // Thu SOFT LOCK: `n_` frame LIÊN TỤC.
      if (soft_announced_) {
        // cửa sổ soft đã xong việc, không thu nữa
      } else if (soft_.size() > 0) {
        soft_.add(feat_, static_cast<float>(sharp));
        log_event(idx, "Soft Lock collecting: " + std::to_string(soft_.size()) + "/" +
                           std::to_string(n_));
      } else {
        coarse = bank_.empty() ? 1.0f : bank_.max_sim(feat_, 0);
        if (coarse >= a_.soft_lock_threshold) {
          soft_.add(feat_, static_cast<float>(sharp));
          log_event(idx, "Soft Lock collecting: 1/" + std::to_string(n_) + " (coarse=" +
                             f3(coarse) + ")");
        } else {
          log_event(idx, "Coarse FAILED (coarse=" + f3(coarse) + " < " +
                             f3(a_.soft_lock_threshold) + ")");
          ev = "coarse fail";
        }
      }

      // Chấm điểm SOFT LOCK: trung bình thường `n_` frame liên tục.
      if (soft_.ready() && !soft_announced_) {
        soft_announced_ = true;
        Vec mean_visual(static_cast<size_t>(dim_), 0.0f);
        for (const Vec& f : soft_.feats) {
          for (int d = 0; d < dim_; ++d) mean_visual[d] += f[d] / static_cast<float>(n_);
        }
        soft_s = bank_.max_sim(mean_visual, 0);
        last_soft = soft_s;
        if (soft_s >= a_.soft_lock_threshold) {
          soft_passed_ = true;
          log_event(idx, "SOFT LOCK PASS (soft=" + f3(soft_s) + " >= " +
                             f3(a_.soft_lock_threshold) + ") -> duoc tinh HARD LOCK");
          ev = "soft pass";
        } else {
          log_event(idx, "SOFT LOCK FAIL (soft=" + f3(soft_s) + " < " +
                             f3(a_.soft_lock_threshold) + ") -> ve T1_LOST");
          ev = "soft fail";
          to_lost(idx, "soft lock fail");
          break;
        }
      }

      // Thu HARD LOCK: `n_` MẪU cách nhau `stride` frame.
      bool new_hard = false;
      if (soft_passed_) {
        new_hard = hard_.should_extract();
        if (new_hard) {
          hard_.add(feat_, static_cast<float>(sharp));
          log_event(idx, "HARD LOCK collecting: " + std::to_string(hard_.size()) + "/" +
                             std::to_string(n_) + " (moi " + std::to_string(a_.stride) +
                             " frame)");
        }
      }

      if (soft_passed_ && new_hard && hard_.ready()) {
        const Bundle b = fused_bundle(hard_);
        ms_tp = temporal_ms_.back();
        fine = bank_.empty() ? 1.0f : bank_.max_sim(b.fused, 2);
        last_fine = fine;
        if (fine >= a_.reid_threshold) {
          const int lat = idx - reappeared_idx_;
          latency_.push_back(lat);
          ++hard_lock_count_;
          log_event(idx, "HARD LOCK! (fine=" + f3(fine) + " >= " + f3(a_.reid_threshold) +
                             ") latency=" + std::to_string(lat) + " frames");
          ev = "HARD LOCK";
          state_ = State::kT3Verified;
          hijack_remaining_ = a_.hijack_check_count;
          last_update_ = video_time;
          bank_.add(b);
          // Nạp `sliding_window` từ đúng cửa sổ HARD LOCK vừa dùng.
          sliding_ = Window(n_, a_.stride);
          for (int i = 0; i < hard_.size(); ++i) sliding_.add(hard_.feats[i], hard_.sharp[i]);
          soft_ = Window(n_, 1);
          hard_ = Window(n_, a_.stride);
          soft_announced_ = false;
          soft_passed_ = false;
        } else {
          ++false_alarms_;
          log_event(idx, "Fine FAILED (fine=" + f3(fine) + " < " + f3(a_.reid_threshold) +
                             ") -> rolling window");
          ev = "false alarm";
          // rolling window: hard_lock_buffer tự đẩy mẫu cũ ra ở lần thu kế tiếp
        }
      }
      break;
    }
  } while (false);

  frame_ms.push_back(ms_since(t_frame));
  write_row(idx, video_time, bbox, !is_absent && valid, crop_w, crop_h, sharp, coarse, soft_s,
            fine, ms_bb, ms_tp, ev);
}

// ---------------------------------------------------------------------------
// Overlay lên video — tương đương draw_ui (infer.py:620), thêm điểm và bộ đệm
// ---------------------------------------------------------------------------
void draw_ui(cv::Mat& img, const BBox& bbox, const Runner& r, int idx, double fps_out,
             double video_time) {
  cv::Scalar color(0, 0, 255);
  std::string label = "LOST";
  if (r.state() == State::kT0Init || r.state() == State::kT3Verified) {
    color = cv::Scalar(0, 255, 0);
    label = "TRACKING (HARD LOCK)";
  } else if (r.state() == State::kT2Search) {
    color = cv::Scalar(255, 255, 0);
    label = "SEARCHING (soft " + std::to_string(r.soft_size()) + "/" +
            std::to_string(r.num_frames()) + " | hard " + std::to_string(r.hard_size()) + "/" +
            std::to_string(r.num_frames()) + ")";
  }

  const int font = cv::FONT_HERSHEY_SIMPLEX;
  cv::putText(img, std::string("State: ") + state_name(r.state()), cv::Point(20, 40), font, 0.8,
              color, 2);
  cv::putText(img, "Bank: " + r.bank().size_info(), cv::Point(20, 70), font, 0.6,
              cv::Scalar(255, 255, 255), 2);
  cv::putText(img, "f " + std::to_string(idx) + "  " + std::to_string(video_time).substr(0, 5) +
                       "s  " + std::to_string(static_cast<int>(fps_out)) + " FPS",
              cv::Point(20, 100), font, 0.6, cv::Scalar(200, 200, 200), 2);
  if (!std::isnan(r.last_coarse) || !std::isnan(r.last_fine)) {
    cv::putText(img, "coarse " + f3(r.last_coarse) + "  soft " + f3(r.last_soft) + "  fine " +
                         f3(r.last_fine),
                cv::Point(20, 130), font, 0.6, cv::Scalar(200, 200, 200), 2);
  }

  if (bbox.valid() && r.state() != State::kT1Lost) {
    cv::rectangle(img, cv::Point(bbox.x, bbox.y), cv::Point(bbox.x + bbox.w, bbox.y + bbox.h),
                  color, 2);
    cv::putText(img, label, cv::Point(bbox.x, std::max(12, bbox.y - 10)), font, 0.6, color, 2);
  }
}

std::string derive_out_dir(const Args& a, const std::string& seq_name) {
  if (!a.out_dir.empty()) return a.out_dir;
  return "infer_cpp/" + seq_name;
}

}  // namespace

int main(int argc, char** argv) {
  const Args a = parse(argc, argv);

  // Kiểm trước khi nạp engine: PRE-BN cần `raw_feat` mà engine không xuất.
  if (a.fine_space != "fused") {
    std::fprintf(stderr,
                 "[infer] fine_space='%s' cần `raw_feat` (PRE-BN), mà engine hiện tại chỉ\n"
                 "        xuất `reid_embed`. Dùng 'fused' hoặc export lại engine.\n",
                 a.fine_space.c_str());
    return 1;
  }

  namespace fs = std::filesystem;
  std::string seq_dir = a.seq_dir;
  while (seq_dir.size() > 1 && seq_dir.back() == '/') seq_dir.pop_back();
  const std::string seq_name = fs::path(seq_dir).filename().string();
  const std::string video_path = seq_dir + "/" + seq_name + ".mp4";
  const std::string gt_path = seq_dir + "/groundtruth_rect.txt";
  const std::string absent_path = seq_dir + "/absent.txt";

  if (a.dry_run) {
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
      std::fprintf(stderr, "[infer] không mở được video: %s\n", video_path.c_str());
      return 1;
    }
    const std::vector<BBox> gt = load_gt(gt_path);
    const std::vector<int> absent = load_absent(absent_path);
    long n_absent = 0;
    for (int v : absent) {
      if (v != 0) ++n_absent;
    }
    std::printf("=== DRY RUN (không dựng engine) ===\n");
    std::printf("  video   : %s\n", video_path.c_str());
    std::printf("            %dx%d, %.2f fps, %ld frame\n",
                static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)),
                static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT)),
                cap.get(cv::CAP_PROP_FPS),
                static_cast<long>(cap.get(cv::CAP_PROP_FRAME_COUNT)));
    std::printf("  gt      : %zu dòng\n", gt.size());
    std::printf("  absent  : %zu dòng, %ld frame vắng\n", absent.size(), n_absent);
    std::printf("  N=%d stride=%d -> can %d frame de day HARD LOCK\n", a.num_frames, a.stride,
                a.num_frames > 0 ? (a.num_frames - 1) * a.stride + 1 : 0);
    std::printf("  padding=%.2f  soft=%.2f  reid=%.2f  hijack=%.2f  update=%.1fs\n", a.padding,
                a.soft_lock_threshold, a.reid_threshold, a.hijack_threshold, a.update_interval);
    return 0;
  }

  const std::string out_dir = derive_out_dir(a, seq_name);
  std::error_code ec;
  fs::create_directories(out_dir, ec);
  if (ec) {
    std::fprintf(stderr, "[infer] không tạo được thư mục %s: %s\n", out_dir.c_str(),
                 ec.message().c_str());
    return 1;
  }

  std::ofstream frames(out_dir + "/frames.csv");
  std::ofstream events(out_dir + "/events.txt");
  std::ofstream summary(out_dir + "/summary.txt");
  std::ofstream cfg_out(out_dir + "/config_used.txt");
  if (!frames || !events || !summary || !cfg_out) {
    std::fprintf(stderr, "[infer] không ghi được file trong %s\n", out_dir.c_str());
    return 1;
  }

  cfg_out << "seq_dir            : " << seq_dir << "\n"
          << "backbone_engine    : " << a.backbone << "\n"
          << "temporal_engine    : " << a.temporal << "\n"
          << "num_frames         : " << a.num_frames << " (0 = tu engine)\n"
          << "stride             : " << a.stride << "\n"
          << "bbox_padding       : " << a.padding << "\n"
          << "soft_lock_threshold: " << a.soft_lock_threshold << "\n"
          << "reid_threshold     : " << a.reid_threshold << "\n"
          << "hijack_threshold   : " << a.hijack_threshold << "\n"
          << "hijack_check_count : " << a.hijack_check_count << "\n"
          << "gap_tolerance      : " << a.gap_tolerance << "\n"
          << "update_interval_sec: " << a.update_interval << "\n"
          << "max_anchor/recent  : " << a.max_anchor << " / " << a.max_recent << "\n"
          << "fine_space         : " << a.fine_space << "\n";
  cfg_out.close();

  uav::Pipeline::Config cfg;
  cfg.backbone_plan = a.backbone;
  cfg.temporal_plan = a.temporal;
  cfg.num_frames = a.num_frames;
  cfg.max_clips = 1;
  cfg.device = a.device;
  cfg.order = uav::ChannelOrder::kBGR;
  cfg.mode = uav::Mode::kStream;
  cfg.frame_stride = 1;

  cv::VideoCapture cap(video_path);
  if (!cap.isOpened()) {
    std::fprintf(stderr, "[infer] không mở được video: %s\n", video_path.c_str());
    return 1;
  }
  const int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
  const int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
  double fps_video = cap.get(cv::CAP_PROP_FPS);
  if (!(fps_video > 1.0)) fps_video = 30.0;

  const std::vector<BBox> bboxes = load_gt(gt_path);
  const std::vector<int> absent = load_absent(absent_path);

  uav::Pipeline pipe(cfg);

  std::printf("==================================================================\n");
  std::printf("  %s  (%dx%d, %.1f fps, %d frame)\n", seq_name.c_str(), width, height, fps_video,
              static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT)));
  std::printf("  engine backbone %s\n", fs::path(a.backbone).filename().string().c_str());
  std::printf("  engine temporal %s\n", fs::path(a.temporal).filename().string().c_str());
  std::printf("  N=%d (engine=%d) stride=%d -> HARD LOCK can %d frame\n", pipe.num_frames(),
              pipe.temporal_n(), a.stride, (pipe.num_frames() - 1) * a.stride + 1);
  std::printf("  visual_dim=%d embed_dim=%d\n", pipe.visual_dim(), pipe.embed_dim());
  std::printf("  nguong: soft=%.2f reid=%.2f hijack=%.2f | update=%.1fs\n",
              a.soft_lock_threshold, a.reid_threshold, a.hijack_threshold, a.update_interval);
  std::printf("==================================================================\n");

  if (!absent.empty() && absent.size() < bboxes.size()) {
    std::printf(" CANH BAO: absent.txt co %zu dong < GT %zu frame; frame thieu coi la PRESENT\n",
                absent.size(), bboxes.size());
  }

  frames << "frame,video_time_s,state,present,bbox_x,bbox_y,bbox_w,bbox_h,crop_w,crop_h,"
            "sharpness,coarse,soft,fine,soft_fill,hard_fill,anchor,recent,hijack_left,"
            "ms_backbone,ms_temporal,event\n";

  cv::VideoWriter writer;
  if (!a.no_video_out) {
    writer.open(out_dir + "/output.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps_video,
                cv::Size(width, height));
    if (!writer.isOpened()) {
      std::fprintf(stderr, "[infer] không mở được VideoWriter -> chỉ ghi txt\n");
    }
  }

  Runner runner(a, pipe, frames, events);

  double total_model = 0.0, total_write = 0.0;
  int frame_idx = 0;
  cv::Mat frame;

  while (cap.read(frame) && !frame.empty()) {
    if (a.limit > 0 && frame_idx >= a.limit) break;

    const bool is_absent = frame_idx < static_cast<int>(absent.size()) && absent[frame_idx] == 1;
    const BBox bbox = frame_idx < static_cast<int>(bboxes.size()) ? bboxes[frame_idx] : BBox{};
    const double video_time = frame_idx / fps_video;

    const auto t0 = Clock::now();
    cv::Mat display = frame.clone();
    runner.process(frame, bbox, is_absent, frame_idx, video_time);

    const double elapsed = ms_since(t0);
    if (frame_idx >= a.warmup) total_model += elapsed;

    const double fps_now = elapsed > 0 ? 1000.0 / elapsed : 0.0;
    draw_ui(display, bbox, runner, frame_idx, fps_now, video_time);

    const auto tw = Clock::now();
    if (writer.isOpened()) writer.write(display);
    if (frame_idx >= a.warmup) total_write += ms_since(tw);

    ++frame_idx;
  }
  cap.release();
  if (writer.isOpened()) writer.release();
  frames.close();
  events.close();

  // ---------------- summary ----------------
  auto mean = [](const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    double s = 0;
    for (double x : v) s += x;
    return s / static_cast<double>(v.size());
  };

  const int counted = std::max(1, frame_idx - a.warmup);
  const double fps_model = total_model > 0 ? 1000.0 * counted / total_model : 0.0;
  const double fps_all = (total_model + total_write) > 0
                             ? 1000.0 * counted / (total_model + total_write)
                             : 0.0;

  summary << std::fixed;
  summary << "=== TONG HOP " << seq_name << " ===\n"
          << "frame xu ly            : " << frame_idx << " (bo " << a.warmup << " warmup)\n"
          << "N / stride             : " << pipe.num_frames() << " / " << a.stride << "\n"
          << "HARD LOCK can          : " << (pipe.num_frames() - 1) * a.stride + 1 << " frame\n"
          << "\n--- THOI GIAN ---\n"
          << "backbone (prep+TRT)    : " << mean(runner.backbone_ms()) << " ms  (n="
          << runner.backbone_ms().size() << ")\n"
          << "temporal (H2D+TRT)     : " << mean(runner.temporal_ms()) << " ms  (n="
          << runner.temporal_ms().size() << ")\n"
          << "throughput (model+UI)  : " << fps_model << " FPS\n"
          << "throughput (+ghi video): " << fps_all << " FPS  (ghi "
          << (counted > 0 ? total_write / counted : 0.0) << " ms/frame)\n"
          << "\n--- KET QUA REID ---\n"
          << "HARD LOCK              : " << runner.hard_lock_count() << " lan\n"
          << "False alarm (fine fail): " << runner.false_alarms() << "\n";
  if (runner.latencies().empty()) {
    summary << "Re-acquisition latency : N/A\n";
  } else {
    double sum = 0, mn = 1e18, mx = 0;
    for (double x : runner.latencies()) {
      sum += x;
      mn = std::min(mn, x);
      mx = std::max(mx, x);
    }
    summary << "Re-acquisition latency : mean " << sum / runner.latencies().size() << " frame"
            << "  (min " << mn << ", max " << mx << ", n=" << runner.latencies().size() << ")\n";
  }
  summary << "\n--- GIOI HAN ---\n"
          << "Khong co cot PRE-BN vi engine chi xuat `reid_embed`.\n"
          << "Cua so chua du N hang phai dem bang cach lap hang cuoi (engine unroll dung N).\n";
  summary.close();

  std::printf("\n--- TONG HOP ---\n");
  std::printf("  backbone  : %.2f ms  (n=%zu)\n", mean(runner.backbone_ms()),
              runner.backbone_ms().size());
  std::printf("  temporal  : %.2f ms  (n=%zu)\n", mean(runner.temporal_ms()),
              runner.temporal_ms().size());
  std::printf("  throughput: %.1f FPS (model+UI) | %.1f FPS (+ghi video)\n", fps_model, fps_all);
  std::printf("  HARD LOCK : %d lan | False alarm: %d | latency: %s\n", runner.hard_lock_count(),
              runner.false_alarms(),
              runner.latencies().empty() ? "N/A" : std::to_string(static_cast<int>(
                                                            mean(runner.latencies()))).c_str());
  std::printf("  ghi ra    : %s/{output.mp4, frames.csv, events.txt, summary.txt, "
              "config_used.txt}\n",
              out_dir.c_str());
  return 0;
}
