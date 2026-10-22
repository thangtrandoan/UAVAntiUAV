// uav_reid_trt — chạy ReID bằng hai engine TensorRT.
//
// Chế độ:
//   --info                 in cấu hình engine rồi thoát
//   --bench N              đo tốc độ với N vòng (sau --warmup vòng)
//   --clip DIR             trích embedding của một clip trong thư mục
//   --match DIR_A DIR_B    cosine giữa hai clip
//   --manifest FILE        mỗi dòng: tên<TAB>path0,path1,... -> in embedding
//
// Ví dụ:
//   ./uav_reid_trt --backbone v1_backbone.engine --temporal v1_temporal.engine
//                  --num-frames 12 --bench 200
//   ./uav_reid_trt --backbone v1_backbone.engine --temporal v1_temporal.engine
//                  --num-frames 12 --match clipA clipB
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "uav_reid/pipeline.hpp"

namespace {

struct Args {
  std::string backbone;
  std::string temporal;
  int num_frames = 12;
  int max_clips = 1;
  int device = 0;
  uav::ChannelOrder order = uav::ChannelOrder::kBGR;
  bool verbose = false;
  bool take_last = false;

  bool info = false;
  int bench = 0;
  int warmup = 5;
  std::string clip;
  std::string match_a;
  std::string match_b;
  std::string manifest;
  std::string stream;
  int stride = 1;
  std::string dump;
};

void usage() {
  std::printf(
      "uav_reid_trt --backbone PATH --temporal PATH [tuỳ chọn]\n\n"
      "  --num-frames N     số frame mỗi clip (mặc định 12)\n"
      "  --max-clips K      số clip chạy song song trong một batch (mặc định 1)\n"
      "  --device D         GPU index (mặc định 0)\n"
      "  --order bgr|rgb    thứ tự kênh của ảnh nguồn (mặc định bgr)\n"
      "  --take-last        lấy N frame CUỐI thay vì N frame đầu\n"
      "  --info             in cấu hình engine rồi thoát\n"
      "  --bench N          đo tốc độ, N vòng\n"
      "  --warmup N         số vòng khởi động trước khi đo (mặc định 5)\n"
      "  --clip DIR         trích embedding một clip\n"
      "  --match A B        cosine giữa hai clip\n"
      "  --manifest FILE    xử lý hàng loạt\n"
      "  --stream DIR       chế độ bộ đệm vòng: đẩy TỪNG frame, temporal đọc\n"
      "                     cửa sổ N frame (giống SlidingWindowBuffer của infer.py)\n"
      "  --stride K         --stream: chỉ đẩy mỗi K frame (mặc định 1)\n"
      "  --dump FILE        ghi embedding float32 thô (count x dim)\n"
      "  --verbose          in log TensorRT\n");
}

bool parse(int argc, char** argv, Args* a) {
  auto need = [&](int& i) -> const char* {
    if (i + 1 >= argc) {
      std::fprintf(stderr, "thiếu giá trị cho %s\n", argv[i]);
      std::exit(EXIT_FAILURE);
    }
    return argv[++i];
  };
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    if (k == "--backbone") a->backbone = need(i);
    else if (k == "--temporal") a->temporal = need(i);
    else if (k == "--num-frames") a->num_frames = std::atoi(need(i));
    else if (k == "--max-clips") a->max_clips = std::atoi(need(i));
    else if (k == "--device") a->device = std::atoi(need(i));
    else if (k == "--order") {
      const std::string v = need(i);
      if (v == "rgb") a->order = uav::ChannelOrder::kRGB;
      else if (v == "bgr") a->order = uav::ChannelOrder::kBGR;
      else { std::fprintf(stderr, "--order phải là rgb hoặc bgr\n"); return false; }
    }
    else if (k == "--verbose") a->verbose = true;
    else if (k == "--take-last") a->take_last = true;
    else if (k == "--info") a->info = true;
    else if (k == "--bench") a->bench = std::atoi(need(i));
    else if (k == "--warmup") a->warmup = std::atoi(need(i));
    else if (k == "--clip") a->clip = need(i);
    else if (k == "--match") { a->match_a = need(i); a->match_b = need(i); }
    else if (k == "--manifest") a->manifest = need(i);
    else if (k == "--stream") a->stream = need(i);
    else if (k == "--stride") a->stride = std::atoi(need(i));
    else if (k == "--dump") a->dump = need(i);
    else if (k == "-h" || k == "--help") { usage(); std::exit(EXIT_SUCCESS); }
    else {
      std::fprintf(stderr, "tham số lạ: %s\n", k.c_str());
      return false;
    }
  }
  if (a->backbone.empty() || a->temporal.empty()) {
    std::fprintf(stderr, "phải có --backbone và --temporal\n");
    return false;
  }
  return true;
}

bool has_image_ext(const std::string& s) {
  auto ends = [&](const char* e) {
    const size_t n = std::strlen(e);
    return s.size() >= n && s.compare(s.size() - n, n, e) == 0;
  };
  return ends(".jpg") || ends(".jpeg") || ends(".png") || ends(".bmp");
}

std::vector<std::string> list_images(const std::string& dir) {
  std::vector<std::string> out;
  cv::glob(dir + "/*", out, false);
  out.erase(std::remove_if(out.begin(), out.end(),
                           [](const std::string& p) { return !has_image_ext(p); }),
            out.end());
  std::sort(out.begin(), out.end());
  return out;
}

// Giữ đúng ngữ nghĩa EvalDataset._load_clip: cắt về N frame, thiếu thì lặp frame cuối.
std::vector<std::string> pick_frames(std::vector<std::string> paths, int n,
                                     bool take_last) {
  if ((int)paths.size() > n) {
    paths = take_last ? std::vector<std::string>(paths.end() - n, paths.end())
                      : std::vector<std::string>(paths.begin(), paths.begin() + n);
  } else if (paths.empty()) {
    std::fprintf(stderr, "không có ảnh nào\n");
    std::exit(EXIT_FAILURE);
  } else {
    while ((int)paths.size() < n) paths.push_back(paths.back());
  }
  return paths;
}

struct LoadedFrames {
  std::vector<cv::Mat> mats;  // BGR 8UC3
  std::vector<uav::HostFrame> frames;
};

LoadedFrames load_clip(const std::string& dir, int n, bool take_last) {
  const std::vector<std::string> paths =
      pick_frames(list_images(dir), n, take_last);
  LoadedFrames lf;
  lf.mats.reserve(paths.size());
  for (const auto& p : paths) {
    cv::Mat m = cv::imread(p, cv::IMREAD_COLOR);
    if (m.empty()) {
      std::fprintf(stderr, "đọc ảnh thất bại: %s\n", p.c_str());
      std::exit(EXIT_FAILURE);
    }
    if (!m.isContinuous()) m = m.clone();
    lf.mats.push_back(m);
  }
  // Dựng HostFrame sau khi đã có đủ mats, để con trỏ không bị treo.
  lf.frames.reserve(lf.mats.size());
  for (const auto& m : lf.mats) {
    uav::HostFrame f;
    f.data = m.data;
    f.width = m.cols;
    f.height = m.rows;
    f.stride = (int)m.step;
    lf.frames.push_back(f);
  }
  return lf;
}

void print_embedding(const float* e, int dim, const char* tag) {
  double ss = 0.0;
  for (int i = 0; i < dim; ++i) ss += (double)e[i] * (double)e[i];
  std::printf("%s dim=%d L2=%.6f  đầu=[", tag, dim, std::sqrt(ss));
  for (int i = 0; i < std::min(dim, 6); ++i) {
    std::printf("%s%.4f", i ? ", " : "", e[i]);
  }
  std::printf(" ...]\n");
}

double cosine(const float* a, const float* b, int dim) {
  double s = 0.0;
  for (int i = 0; i < dim; ++i) s += (double)a[i] * (double)b[i];
  return s;
}

int cmd_bench(uav::Pipeline& pipe, const Args& a) {
  // Frame tổng hợp 256x256 BGR. Cấp phát thường là đủ cho phép đo;
  // đường H2D không phải nút thắt ở đây.
  const int total = a.max_clips * a.num_frames;
  std::vector<cv::Mat> mats;
  mats.reserve((size_t)total);
  for (int i = 0; i < total; ++i) {
    cv::Mat m(256, 256, CV_8UC3);
    cv::randu(m, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
    mats.push_back(m);
  }
  std::vector<uav::HostFrame> frames;
  frames.reserve((size_t)total);
  for (const auto& m : mats) {
    uav::HostFrame f;
    f.data = m.data;
    f.width = m.cols;
    f.height = m.rows;
    f.stride = (int)m.step;
    frames.push_back(f);
  }

  std::vector<float> emb((size_t)a.max_clips * pipe.embed_dim());

  for (int i = 0; i < a.warmup; ++i) pipe.run(frames, a.max_clips, emb.data());

  std::vector<double> total_ms, bb_ms, tp_ms, pre_ms;
  total_ms.reserve((size_t)a.bench);
  bb_ms.reserve((size_t)a.bench);
  tp_ms.reserve((size_t)a.bench);
  pre_ms.reserve((size_t)a.bench);

  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < a.bench; ++i) {
    pipe.run(frames, a.max_clips, emb.data(), /*measure=*/true);
    const auto& t = pipe.last_timings();
    total_ms.push_back(t.total_ms);
    bb_ms.push_back(t.backbone_ms);
    tp_ms.push_back(t.temporal_ms);
    pre_ms.push_back(t.upload_preprocess_ms);
  }
  const auto t1 = std::chrono::steady_clock::now();
  const double wall_s =
      std::chrono::duration<double>(t1 - t0).count();

  auto stat = [](std::vector<double>& v, const char* name) {
    std::vector<double> s = v;
    std::sort(s.begin(), s.end());
    const double mean = std::accumulate(s.begin(), s.end(), 0.0) / (double)s.size();
    auto pct = [&](double p) {
      const size_t i = std::min(s.size() - 1, (size_t)(p * (double)(s.size() - 1)));
      return s[i];
    };
    std::printf("  %-22s mean=%7.3f ms  median=%7.3f  p95=%7.3f  p99=%7.3f\n",
                name, mean, pct(0.50), pct(0.95), pct(0.99));
  };

  std::printf("\n=== BENCH  N=%d  max_clips=%d  reps=%d ===\n", a.num_frames,
              a.max_clips, a.bench);
  stat(pre_ms, "upload+preprocess");
  stat(bb_ms, "backbone");
  stat(tp_ms, "temporal+head");
  stat(total_ms, "TOTAL (GPU)");

  const double mean_total = std::accumulate(total_ms.begin(), total_ms.end(), 0.0) /
                            (double)total_ms.size();
  std::printf("  wall                 %.3f s tong, %.3f ms/vong\n",
              wall_s, wall_s * 1000.0 / (double)a.bench);
  std::printf("  throughput           %.2f clip/s  |  %.2f frame/s\n",
              mean_total > 0 ? 1000.0 / mean_total * a.max_clips : 0.0,
              mean_total > 0
                  ? 1000.0 / mean_total * a.max_clips * a.num_frames
                  : 0.0);
  return 0;
}

}  // namespace

// ---------------------------------------------------------------------------
// kStream: chế độ bộ đệm vòng.
//   Đẩy từng frame qua backbone (batch 1), lưu feature vào bộ đệm N slot, chạy
//   temporal trên cửa sổ N gần nhất. Đây là đường ONLINE: có embedding mới ngay
//   mỗi frame thay vì chờ đủ N frame như kClip.
// ---------------------------------------------------------------------------
int cmd_stream(uav::Pipeline& pipe, const Args& a) {
  const std::vector<std::string> paths = list_images(a.stream);
  if (paths.empty()) {
    std::fprintf(stderr, "không có ảnh nào trong %s\n", a.stream.c_str());
    return EXIT_FAILURE;
  }
  // n = tổng số ảnh nên pick_frames không cắt và không đệm: giữ đúng thứ tự.
  const LoadedFrames lf = load_clip(a.stream, static_cast<int>(paths.size()), false);
  const int n = static_cast<int>(lf.frames.size());
  const int dim = pipe.embed_dim();

  std::printf("bộ đệm vòng: %d frame nguồn, cửa sổ N=%d, stride=%d\n", n,
              pipe.num_frames(), a.stride);
  if (n < pipe.num_frames()) {
    std::printf(
        "  LƯU Ý: chỉ có %d frame < N=%d nên cửa sổ phải lặp lại frame; số đo "
        "vẫn hợp lệ nhưng không phải %d frame khác nhau.\n",
        n, pipe.num_frames(), pipe.num_frames());
  }

  std::vector<float> emb(static_cast<size_t>(dim), 0.0f);
  std::vector<std::vector<float>> history;

  // Không có --bench thì chạy đúng một lượt qua dữ liệu; cần ít nhất N lần đẩy
  // mới đầy cửa sổ nên vòng lặp quay lại đầu danh sách nếu thiếu frame.
  const int rounds =
      a.bench > 0 ? a.bench : std::max(n, pipe.num_frames());
  const int warmup = a.bench > 0 ? a.warmup : 0;
  const int total = rounds + warmup;

  // Tach rieng tien xu ly va backbone: gop lai thi khong biet 8 ms nam o dau.
  std::vector<double> pre_ms;
  std::vector<double> bb_ms;
  std::vector<double> tpl_ms;
  int measured = 0;

  for (int i = 0; i < total; ++i) {
    const bool meas = (i >= warmup);
    pipe.push_frame(lf.frames[static_cast<size_t>(i % n)], meas);
    if (!pipe.ready()) continue;
    pipe.embed(emb.data(), meas);
    if (meas) {
      const uav::Pipeline::Timings& t = pipe.last_timings();
      pre_ms.push_back(t.upload_preprocess_ms);
      bb_ms.push_back(t.backbone_ms);
      tpl_ms.push_back(t.temporal_ms);
      ++measured;
    } else {
      history.push_back(emb);
    }
  }

  if (measured > 0) {
    auto mean = [](const std::vector<double>& v) {
      if (v.empty()) return 0.0;
      return std::accumulate(v.begin(), v.end(), 0.0) / (double)v.size();
    };
    auto pct = [](std::vector<double> v, double p) {
      if (v.empty()) return 0.0;
      std::sort(v.begin(), v.end());
      return v[(size_t)(p * (double)(v.size() - 1))];
    };
    std::vector<double> tot;
    tot.reserve((size_t)measured);
    for (int i = 0; i < measured; ++i) {
      tot.push_back(pre_ms[(size_t)i] + bb_ms[(size_t)i] + tpl_ms[(size_t)i]);
    }

    std::printf("\n  %d bước đo (mỗi bước = 1 frame mới)\n", measured);
    std::printf("  upload + tiền xử lý     %8.3f ms\n", mean(pre_ms));
    std::printf("  backbone (batch 1)      %8.3f ms\n", mean(bb_ms));
    std::printf("  temporal (cửa sổ N=%d)  %8.3f ms\n", pipe.num_frames(),
                mean(tpl_ms));
    std::printf("  --------------------------------------------------\n");
    std::printf("  mỗi frame               %8.3f ms   p95 %.3f\n", mean(tot),
                pct(tot, 0.95));
    std::printf("  -> %.1f FPS\n",
                mean(tot) > 0.0 ? 1000.0 / mean(tot) : 0.0);
  }

  if (!history.empty()) {
    // Đã L2-chuẩn-hoá nên tích vô hướng chính là cosine. Hai embedding liền kề
    // phải gần nhau; nhảy bất thường là dấu hiệu bộ đệm trượt sai.
    double worst = 1.0;
    for (size_t i = 1; i < history.size(); ++i) {
      worst = std::min(worst,
                       cosine(history[i].data(), history[i - 1].data(), dim));
    }
    std::printf("  cosine liền kề thấp nhất: %.6f\n", worst);
    print_embedding(history.back().data(), dim, "embedding cuối");

    if (!a.dump.empty()) {
      std::FILE* fp = std::fopen(a.dump.c_str(), "wb");
      if (fp == nullptr) {
        std::fprintf(stderr, "không mở được %s để ghi\n", a.dump.c_str());
        return EXIT_FAILURE;
      }
      for (const auto& e : history) {
        if (std::fwrite(e.data(), sizeof(float), e.size(), fp) != e.size()) {
          std::fprintf(stderr, "ghi %s thất bại\n", a.dump.c_str());
          std::fclose(fp);
          return EXIT_FAILURE;
        }
      }
      std::fclose(fp);
      std::printf("  đã ghi %zu x %d float -> %s\n", history.size(), dim,
                  a.dump.c_str());
    }
  }
  return EXIT_SUCCESS;
}

// ---------------------------------------------------------------------------
// --info: chẩn đoán hai engine mà KHÔNG cần allocate().
// Nhờ vậy vẫn xem được profile khi cặp engine không khớp shape (ví dụ N=16 mà
// backbone chỉ cho batch toi da 12).
// ---------------------------------------------------------------------------
int cmd_info(const Args& a) {
  uav::TrtEngine bb;
  uav::TrtEngine tp;
  bb.set_logger_verbose(a.verbose);
  tp.set_logger_verbose(a.verbose);
  bb.load(a.backbone, a.device, nullptr);
  tp.load(a.temporal, a.device, nullptr);

  std::printf("BACKBONE\n");
  bb.describe();
  std::printf("\nTEMPORAL\n");
  tp.describe();

  // Doi chieu hai engine: chieu dac trung phai khop.
  std::printf("\nDOI CHIEU\n");
  int verdict = EXIT_SUCCESS;
  if (!bb.has("cnn_feat") || !tp.has("seq_feats")) {
    std::fprintf(stderr, "  thieu 'cnn_feat' hoac 'seq_feats'\n");
    return EXIT_FAILURE;
  }
  const nvinfer1::Dims bo = bb.at("cnn_feat").dims;
  const nvinfer1::Dims ti = tp.at("seq_feats").dims;
  const int d_backbone = bo.nbDims >= 2 ? (int)bo.d[bo.nbDims - 1] : -1;
  const int d_temporal = ti.nbDims >= 3 ? (int)ti.d[ti.nbDims - 1] : -1;
  const int n_frames = ti.nbDims >= 3 ? (int)ti.d[ti.nbDims - 2] : -1;
  std::printf("  chieu dac trung: backbone=%d  temporal=%d  %s\n", d_backbone,
              d_temporal, d_backbone == d_temporal ? "KHOP" : "LECH -> dung lai engine");
  if (d_backbone != d_temporal) verdict = EXIT_FAILURE;

  std::printf("  cua so temporal N=%d  -> phai chay --num-frames %d\n", n_frames,
              n_frames);
  if (a.num_frames != n_frames) {
    std::printf("  CANH BAO: --num-frames=%d nhung engine la N=%d\n",
                a.num_frames, n_frames);
  }

  // kClip can backbone batch = clips*N; bao truoc neu vuot profile.
  const nvinfer1::Dims mx =
      bb.profile_shape("input_img", 0, nvinfer1::OptProfileSelector::kMAX);
  const int max_batch = mx.nbDims > 0 ? (int)mx.d[0] : -1;
  std::printf("  backbone batch toi da=%d\n", max_batch);
  if (max_batch >= 0 && n_frames > 0 && max_batch < n_frames) {
    std::printf(
        "  -> kClip KHONG chay duoc (can batch %d > %d). Dung --stream, hoac\n"
        "     export lai backbone voi maxShapes lon hon (boi so cua %d).\n",
        n_frames, max_batch, n_frames);
  } else if (max_batch >= 0 && n_frames > 0) {
    std::printf("  -> kClip chay duoc voi --num-frames %d --max-clips <= %d\n",
                n_frames, max_batch / n_frames);
  }
  return verdict;
}

int main(int argc, char** argv) {
  Args a;
  if (!parse(argc, argv, &a)) {
    usage();
    return EXIT_FAILURE;
  }

  if (a.info) return cmd_info(a);

  uav::Pipeline::Config cfg;
  cfg.backbone_plan = a.backbone;
  cfg.temporal_plan = a.temporal;
  cfg.num_frames = a.num_frames;
  cfg.max_clips = a.max_clips;
  cfg.device = a.device;
  cfg.order = a.order;
  cfg.verbose_trt = a.verbose;
  // --stream chọn chế độ bộ đệm vòng; ngược lại giữ chế độ theo cụm.
  cfg.mode = a.stream.empty() ? uav::Mode::kClip : uav::Mode::kStream;
  cfg.frame_stride = a.stride > 0 ? a.stride : 1;

  uav::Pipeline pipe(cfg);
  if (a.bench == 0 && a.clip.empty() && a.manifest.empty() &&
      a.match_a.empty() && a.stream.empty()) {
    pipe.print_info();
  }

  if (!a.stream.empty()) return cmd_stream(pipe, a);
  if (a.bench > 0) return cmd_bench(pipe, a);

  const int dim = pipe.embed_dim();

  if (!a.clip.empty()) {
    auto lf = load_clip(a.clip, a.num_frames, a.take_last);
    std::vector<float> emb((size_t)dim);
    pipe.run(lf.frames, 1, emb.data());
    print_embedding(emb.data(), dim, a.clip.c_str());
    if (!a.dump.empty()) {
      std::ofstream f(a.dump, std::ios::binary);
      f.write(reinterpret_cast<const char*>(emb.data()),
              (std::streamsize)emb.size() * 4);
      std::printf("  đã ghi %s (%d float32)\n", a.dump.c_str(), dim);
    }
    return EXIT_SUCCESS;
  }

  if (!a.match_a.empty()) {
    auto la = load_clip(a.match_a, a.num_frames, a.take_last);
    std::vector<float> ea((size_t)dim);
    pipe.run(la.frames, 1, ea.data());
    auto lb = load_clip(a.match_b, a.num_frames, a.take_last);
    std::vector<float> eb((size_t)dim);
    pipe.run(lb.frames, 1, eb.data());
    print_embedding(ea.data(), dim, "A");
    print_embedding(eb.data(), dim, "B");
    std::printf("  cosine(A,B) = %.6f\n", cosine(ea.data(), eb.data(), dim));
    return EXIT_SUCCESS;
  }

  if (!a.manifest.empty()) {
    std::ifstream in(a.manifest);
    if (!in) {
      std::fprintf(stderr, "không mở được manifest: %s\n", a.manifest.c_str());
      return EXIT_FAILURE;
    }
    std::vector<float> all;
    std::vector<std::string> names;
    std::string line;
    int line_no = 0;
    while (std::getline(in, line)) {
      ++line_no;
      if (line.empty()) continue;
      const size_t tab = line.find('\t');
      if (tab == std::string::npos) {
        std::fprintf(stderr, "dòng %d thiếu TAB: %s\n", line_no, line.c_str());
        return EXIT_FAILURE;
      }
      const std::string name = line.substr(0, tab);
      std::string rest = line.substr(tab + 1);
      std::vector<std::string> paths;
      size_t pos = 0;
      while ((pos = rest.find(',')) != std::string::npos) {
        paths.push_back(rest.substr(0, pos));
        rest.erase(0, pos + 1);
      }
      if (!rest.empty()) paths.push_back(rest);
      paths = pick_frames(paths, a.num_frames, a.take_last);

      std::vector<cv::Mat> mats;
      mats.reserve(paths.size());
      for (const auto& p : paths) {
        cv::Mat m = cv::imread(p, cv::IMREAD_COLOR);
        if (m.empty()) {
          std::fprintf(stderr, "đọc ảnh thất bại: %s\n", p.c_str());
          return EXIT_FAILURE;
        }
        if (!m.isContinuous()) m = m.clone();
        mats.push_back(m);
      }
      std::vector<uav::HostFrame> frames;
      frames.reserve(mats.size());
      for (const auto& m : mats) {
        uav::HostFrame f;
        f.data = m.data;
        f.width = m.cols;
        f.height = m.rows;
        f.stride = (int)m.step;
        frames.push_back(f);
      }
      std::vector<float> emb((size_t)dim);
      pipe.run(frames, 1, emb.data());
      print_embedding(emb.data(), dim, name.c_str());
      names.push_back(name);
      all.insert(all.end(), emb.begin(), emb.end());
    }
    if (!a.dump.empty()) {
      std::ofstream f(a.dump, std::ios::binary);
      f.write(reinterpret_cast<const char*>(all.data()),
              (std::streamsize)all.size() * 4);
      std::printf("  đã ghi %s (%zu x %d float32)\n", a.dump.c_str(),
                  names.size(), dim);
    }
    return EXIT_SUCCESS;
  }

  return EXIT_SUCCESS;
}
