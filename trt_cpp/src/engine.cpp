#include "uav_reid/engine.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>

#include "uav_reid/cuda_check.hpp"

namespace uav {
namespace {

// Mỗi tensor bắt đầu ở bội số của giá trị này. Đủ cho mọi vector load
// (float4 = 16 B, nhưng 256 B cũng tránh false sharing giữa các tensor liền nhau).
constexpr size_t kAlign = 256;

size_t align_up(size_t v, size_t a) { return (v + a - 1) / a * a; }

// `--fp16` của trtexec chỉ đặt cờ builder cho các lớp BÊN TRONG; tensor I/O vẫn
// giữ kiểu khai trong ONNX (thường FLOAT). In ra để kiểm đúng điều đó.
const char* dtype_str(nvinfer1::DataType t) {
  switch (t) {
    case nvinfer1::DataType::kFLOAT:
      return "FLOAT";
    case nvinfer1::DataType::kHALF:
      return "HALF";
    case nvinfer1::DataType::kINT32:
      return "INT32";
    case nvinfer1::DataType::kINT64:
      return "INT64";
    case nvinfer1::DataType::kBOOL:
      return "BOOL";
    default:
      return "KHAC";
  }
}

}  // namespace

void TrtLogger::log(Severity sev, const char* msg) noexcept {
  if (!verbose_ && (sev == Severity::kINFO || sev == Severity::kVERBOSE)) return;
  const char* tag = "INFO";
  switch (sev) {
    case Severity::kINTERNAL_ERROR: tag = "INTERNAL_ERROR"; break;
    case Severity::kERROR: tag = "ERROR"; break;
    case Severity::kWARNING: tag = "WARN"; break;
    case Severity::kINFO: tag = "INFO"; break;
    default: tag = "VERBOSE"; break;
  }
  std::fprintf(stderr, "[TRT %s] %s\n", tag, msg);
}

size_t TensorInfo::elem_count() const {
  if (dims.nbDims < 0) return 0;
  size_t n = 1;
  for (int i = 0; i < dims.nbDims; ++i) {
    if (dims.d[i] < 0) return 0;
    n *= static_cast<size_t>(dims.d[i]);
  }
  return n;
}

TrtEngine::~TrtEngine() {
  // Thứ tự huỷ quan trọng: context -> engine -> runtime -> bộ nhớ.
  context_.reset();
  engine_.reset();
  runtime_.reset();
  if (arena_ != nullptr) {
    cudaFree(arena_);
    arena_ = nullptr;
  }
}

size_t TrtEngine::dtype_size(nvinfer1::DataType t) {
  switch (t) {
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kINT64: return 8;
    case nvinfer1::DataType::kBOOL: return 1;
    case nvinfer1::DataType::kUINT8: return 1;
    default: return 0;
  }
}

size_t TrtEngine::volume(const nvinfer1::Dims& d) {
  if (d.nbDims < 0) return 0;
  size_t n = 1;
  for (int i = 0; i < d.nbDims; ++i) {
    if (d.d[i] < 0) return 0;
    n *= static_cast<size_t>(d.d[i]);
  }
  return n;
}

bool TrtEngine::dims_static(const nvinfer1::Dims& d) {
  for (int i = 0; i < d.nbDims; ++i) {
    if (d.d[i] < 0) return false;
  }
  return true;
}

std::string TrtEngine::dims_str(const nvinfer1::Dims& d) {
  std::ostringstream os;
  os << "[";
  for (int i = 0; i < d.nbDims; ++i) {
    if (i) os << ",";
    os << d.d[i];
  }
  os << "]";
  return os.str();
}

void TrtEngine::load(const std::string& plan_path, int device_id,
                     cudaStream_t stream) {
  plan_path_ = plan_path;
  UAV_CUDA_CHECK(cudaSetDevice(device_id));

  std::ifstream f(plan_path, std::ios::binary | std::ios::ate);
  if (!f) {
    std::fprintf(stderr, "[TRT] không mở được plan: %s\n", plan_path.c_str());
    std::exit(EXIT_FAILURE);
  }
  const std::streamsize size = f.tellg();
  if (size <= 0) {
    std::fprintf(stderr, "[TRT] plan rỗng: %s\n", plan_path.c_str());
    std::exit(EXIT_FAILURE);
  }
  f.seekg(0, std::ios::beg);
  std::vector<char> blob(static_cast<size_t>(size));
  if (!f.read(blob.data(), size)) {
    std::fprintf(stderr, "[TRT] đọc plan thất bại: %s\n", plan_path.c_str());
    std::exit(EXIT_FAILURE);
  }
  f.close();

  runtime_.reset(nvinfer1::createInferRuntime(logger_));
  UAV_TRT_CHECK(runtime_ != nullptr, "createInferRuntime trả null");

  engine_.reset(runtime_->deserializeCudaEngine(blob.data(), blob.size()));
  if (!engine_) {
    std::fprintf(stderr,
                 "[TRT] deserialize thất bại: %s\n"
                 "      thường do engine build cho GPU/phiên bản TensorRT khác.\n",
                 plan_path.c_str());
    std::exit(EXIT_FAILURE);
  }

  context_.reset(engine_->createExecutionContext());
  UAV_TRT_CHECK(context_ != nullptr, "createExecutionContext trả null");

  index_io_tensors();
  choose_profile(stream);
}

void TrtEngine::index_io_tensors() {
  const int n = engine_->getNbIOTensors();
  UAV_TRT_CHECK(n > 0, "engine không có tensor I/O nào");

  tensors_.clear();
  index_.clear();
  tensors_.reserve(static_cast<size_t>(n));
  any_dynamic_ = false;

  for (int i = 0; i < n; ++i) {
    const char* name = engine_->getIOTensorName(i);
    UAV_TRT_CHECK(name != nullptr, "getIOTensorName trả null");

    TensorInfo t;
    t.name = name;
    t.mode = engine_->getTensorIOMode(name);
    t.dtype = engine_->getTensorDataType(name);
    t.dims = engine_->getTensorShape(name);
    t.was_dynamic = !dims_static(t.dims);
    if (t.was_dynamic) any_dynamic_ = true;

    if (dtype_size(t.dtype) == 0) {
      std::fprintf(stderr, "[TRT] dtype không hỗ trợ cho tensor '%s'\n",
                   t.name.c_str());
      std::exit(EXIT_FAILURE);
    }
    index_[t.name] = static_cast<int>(tensors_.size());
    tensors_.push_back(std::move(t));
  }
}

void TrtEngine::choose_profile(cudaStream_t stream) {
  profile_count_ = engine_->getNbOptimizationProfiles();
  profile_index_ = 0;
  if (!any_dynamic_ || requested_shapes_.empty()) return;

  // Engine động: chọn profile đầu tiên mà min <= shape yêu cầu <= max
  // cho MỌI đầu vào động. Thường chỉ có 1 profile nên vòng lặp chạy 1 lần.
  auto fits = [&](int p) {
    for (const auto& kv : requested_shapes_) {
      const TensorInfo* t = find(kv.first);
      if (t == nullptr || !t->was_dynamic) continue;
      const nvinfer1::Dims mn =
          engine_->getProfileShape(kv.first.c_str(), p,
                                   nvinfer1::OptProfileSelector::kMIN);
      const nvinfer1::Dims mx =
          engine_->getProfileShape(kv.first.c_str(), p,
                                   nvinfer1::OptProfileSelector::kMAX);
      const nvinfer1::Dims& d = kv.second;
      if (d.nbDims != mn.nbDims) return false;
      for (int i = 0; i < d.nbDims; ++i) {
        if (d.d[i] < mn.d[i] || d.d[i] > mx.d[i]) return false;
      }
    }
    return true;
  };

  bool found = false;
  for (int p = 0; p < profile_count_; ++p) {
    if (fits(p)) {
      profile_index_ = p;
      found = true;
      break;
    }
  }

  if (!found) {
    std::fprintf(stderr,
                 "[TRT] không có optimization profile nào chứa shape yêu cầu "
                 "(%s)\n", plan_path_.c_str());
    for (const auto& kv : requested_shapes_) {
      std::fprintf(stderr, "      yêu cầu %s = %s\n", kv.first.c_str(),
                   dims_str(kv.second).c_str());
    }
    for (int p = 0; p < profile_count_; ++p) {
      for (const auto& t : tensors_) {
        if (!t.was_dynamic || t.mode != nvinfer1::TensorIOMode::kINPUT) continue;
        const nvinfer1::Dims mn = engine_->getProfileShape(
            t.name.c_str(), p, nvinfer1::OptProfileSelector::kMIN);
        const nvinfer1::Dims mx = engine_->getProfileShape(
            t.name.c_str(), p, nvinfer1::OptProfileSelector::kMAX);
        std::fprintf(stderr, "      profile %d: %s min=%s max=%s\n", p,
                     t.name.c_str(), dims_str(mn).c_str(),
                     dims_str(mx).c_str());
      }
    }
    std::exit(EXIT_FAILURE);
  }

  if (!context_->setOptimizationProfileAsync(profile_index_, stream)) {
    std::fprintf(stderr, "[TRT] setOptimizationProfileAsync(%d) thất bại\n",
                 profile_index_);
    std::exit(EXIT_FAILURE);
  }
  UAV_CUDA_CHECK(cudaStreamSynchronize(stream));
}

void TrtEngine::set_input_shape(const std::string& name,
                               const nvinfer1::Dims& d) {
  const TensorInfo* t = find(name);
  if (t == nullptr) {
    std::fprintf(stderr, "[TRT] không có tensor '%s' (shape)\n", name.c_str());
    std::exit(EXIT_FAILURE);
  }
  if (t->mode != nvinfer1::TensorIOMode::kINPUT) {
    std::fprintf(stderr, "[TRT] '%s' không phải đầu vào (shape)\n", name.c_str());
    std::exit(EXIT_FAILURE);
  }
  // Cache: gọi lại cùng shape là no-op, tránh TRT phải làm lại shape inference.
  auto it = requested_shapes_.find(name);
  if (it != requested_shapes_.end()) {
    if (it->second.nbDims == d.nbDims &&
        std::memcmp(it->second.d, d.d, sizeof(d.d[0]) * (size_t)d.nbDims) == 0) {
      return;
    }
  }
  requested_shapes_[name] = d;
  allocated_ = false;
}

void TrtEngine::bind_external(const std::string& name, void* device_ptr) {
  const int i = index_.at(name);
  tensors_[(size_t)i].external = device_ptr;
  allocated_ = false;
}

void TrtEngine::describe() const {
  std::printf("  %s\n", plan_path_.c_str());
  for (const auto& t : tensors_) {
    const char* dir = t.mode == nvinfer1::TensorIOMode::kINPUT ? "vao" : "ra ";
    std::printf("    %s %-14s %-6s dims=[%s] %zu B%s\n", dir, t.name.c_str(),
                dtype_str(t.dtype), dims_str(t.dims).c_str(), t.bytes,
                t.was_dynamic ? "  dong" : "");
  }
  std::printf("    arena=%zu B  so profile=%d\n", arena_bytes_, profile_count_);
  for (int p = 0; p < profile_count_; ++p) {
    for (const auto& t : tensors_) {
      if (!t.was_dynamic || t.mode != nvinfer1::TensorIOMode::kINPUT) continue;
      const nvinfer1::Dims mn = engine_->getProfileShape(
          t.name.c_str(), p, nvinfer1::OptProfileSelector::kMIN);
      const nvinfer1::Dims op = engine_->getProfileShape(
          t.name.c_str(), p, nvinfer1::OptProfileSelector::kOPT);
      const nvinfer1::Dims mx = engine_->getProfileShape(
          t.name.c_str(), p, nvinfer1::OptProfileSelector::kMAX);
      std::printf("    profile %d  %-12s min=%s opt=%s max=%s\n", p,
                  t.name.c_str(), dims_str(mn).c_str(), dims_str(op).c_str(),
                  dims_str(mx).c_str());
    }
  }
}

nvinfer1::Dims TrtEngine::profile_shape(
    const std::string& name, int profile,
    nvinfer1::OptProfileSelector which) const {
  if (engine_ == nullptr || !has(name)) return nvinfer1::Dims{};
  return engine_->getProfileShape(name.c_str(), profile, which);
}

void TrtEngine::rebind(const std::string& name, void* device_ptr) {
  if (!allocated_) {
    std::fprintf(stderr, "[TRT] gọi rebind() trước allocate() (%s)\n",
                 plan_path_.c_str());
    std::exit(EXIT_FAILURE);
  }
  if (!context_->setTensorAddress(name.c_str(), device_ptr)) {
    std::fprintf(stderr, "[TRT] rebind('%s') thất bại\n", name.c_str());
    std::exit(EXIT_FAILURE);
  }
}

void TrtEngine::resolve_shapes_and_sizes() {
  // Ép TRT suy ra shape đầu ra từ shape đầu vào hiện tại.
  //
  // CẨN THẬN: inferShapes() trả int32_t, KHÔNG phải bool (NvInferRuntime.h:4123):
  //     0   = thành công
  //     n>0 = còn n đầu vào chưa được đặt shape (tên ghi vào tensorNames)
  //     -1  = lỗi khác (hoặc nbMaxNames < 0)
  // Viết `if (!inferShapes(...))` sẽ báo lỗi đúng vào lúc THÀNH CÔNG.
  if (any_dynamic_) {
    std::vector<const char*> missing(8, nullptr);
    const int32_t r = context_->inferShapes(
        static_cast<int32_t>(missing.size()), missing.data());
    if (r != 0) {
      if (r > 0) {
        const int shown = std::min<int>(static_cast<int>(r),
                                        static_cast<int>(missing.size()));
        std::fprintf(stderr,
                     "[TRT] inferShapes: %d đầu vào chưa đặt shape (%s)\n",
                     static_cast<int>(r), plan_path_.c_str());
        for (int i = 0; i < shown; ++i) {
          std::fprintf(stderr, "      thiếu: %s\n",
                       missing[(size_t)i] != nullptr ? missing[(size_t)i] : "?");
        }
      } else {
        std::fprintf(stderr,
                     "[TRT] inferShapes trả -1: shape không hợp lệ (%s)\n",
                     plan_path_.c_str());
      }
      std::exit(EXIT_FAILURE);
    }
  }

  size_t need = 0;
  for (auto& t : tensors_) {
    nvinfer1::Dims d = context_->getTensorShape(t.name.c_str());
    if (!dims_static(d)) {
      std::fprintf(stderr,
                   "[TRT] tensor '%s' vẫn còn chiều động %s sau khi chốt shape "
                   "(%s)\n", t.name.c_str(), dims_str(d).c_str(),
                   plan_path_.c_str());
      std::exit(EXIT_FAILURE);
    }
    t.dims = d;
    t.bytes = volume(d) * dtype_size(t.dtype);
    if (t.external != nullptr) {
      t.offset = 0;
      continue;
    }
    need = align_up(need, kAlign);
    t.offset = need;
    need += align_up(t.bytes, kAlign);
  }
  arena_bytes_ = need;
}

void TrtEngine::bind_addresses() {
  for (const auto& t : tensors_) {
    void* p = t.ptr(arena_);
    if (!context_->setTensorAddress(t.name.c_str(), p)) {
      std::fprintf(stderr, "[TRT] setTensorAddress('%s') thất bại\n",
                   t.name.c_str());
      std::exit(EXIT_FAILURE);
    }
  }
}

void TrtEngine::allocate(cudaStream_t stream) {
  // Thứ tự bắt buộc của TensorRT:
  //   1. setOptimizationProfileAsync   (chỉ engine động)
  //   2. setInputShape                 (đẩy shape đã yêu cầu vào context)
  //   3. inferShapes                   (để TRT suy ra shape đầu ra)
  if (any_dynamic_ && !requested_shapes_.empty()) {
    choose_profile(stream);
  }

  for (const auto& t : tensors_) {
    if (!t.was_dynamic || t.mode != nvinfer1::TensorIOMode::kINPUT) continue;
    auto it = requested_shapes_.find(t.name);
    if (it == requested_shapes_.end()) {
      std::fprintf(stderr,
                   "[TRT] đầu vào động '%s' chưa được đặt shape (%s)\n",
                   t.name.c_str(), plan_path_.c_str());
      std::exit(EXIT_FAILURE);
    }
    if (!context_->setInputShape(t.name.c_str(), it->second)) {
      std::fprintf(stderr, "[TRT] setInputShape('%s') thất bại\n",
                   t.name.c_str());
      std::exit(EXIT_FAILURE);
    }
  }

  resolve_shapes_and_sizes();

  if (arena_bytes_ > 0) {
    if (arena_ != nullptr) UAV_CUDA_CHECK(cudaFree(arena_));
    UAV_CUDA_CHECK(cudaMalloc(&arena_, arena_bytes_));
    UAV_CUDA_CHECK(cudaMemsetAsync(arena_, 0, arena_bytes_, stream));
  }

  bind_addresses();
  allocated_ = true;
}

const TensorInfo* TrtEngine::find(const std::string& name) const {
  auto it = index_.find(name);
  if (it == index_.end()) return nullptr;
  return &tensors_[(size_t)it->second];
}

bool TrtEngine::has(const std::string& name) const {
  return index_.find(name) != index_.end();
}

const TensorInfo& TrtEngine::at(const std::string& name) const {
  const TensorInfo* t = find(name);
  if (t == nullptr) {
    std::fprintf(stderr, "[TRT] không có tensor '%s'\n", name.c_str());
    std::exit(EXIT_FAILURE);
  }
  return *t;
}

const TensorInfo& TrtEngine::only_input() const {
  const TensorInfo* found = nullptr;
  int count = 0;
  for (const auto& t : tensors_) {
    if (t.mode == nvinfer1::TensorIOMode::kINPUT) {
      ++count;
      found = &t;
    }
  }
  if (count != 1) {
    std::fprintf(stderr, "[TRT] engine có %d đầu vào, cần đúng 1 (%s)\n", count,
                 plan_path_.c_str());
    std::exit(EXIT_FAILURE);
  }
  return *found;
}

const TensorInfo& TrtEngine::only_output() const {
  const TensorInfo* found = nullptr;
  int count = 0;
  for (const auto& t : tensors_) {
    if (t.mode == nvinfer1::TensorIOMode::kOUTPUT) {
      ++count;
      found = &t;
    }
  }
  if (count != 1) {
    std::fprintf(stderr, "[TRT] engine có %d đầu ra, cần đúng 1 (%s)\n", count,
                 plan_path_.c_str());
    std::exit(EXIT_FAILURE);
  }
  return *found;
}

void* TrtEngine::ptr(const std::string& name) const {
  const TensorInfo& t = at(name);
  return t.ptr(arena_);
}

size_t TrtEngine::bytes(const std::string& name) const { return at(name).bytes; }

nvinfer1::Dims TrtEngine::dims(const std::string& name) const {
  return at(name).dims;
}

void TrtEngine::infer(cudaStream_t stream) {
  if (!allocated_) {
    std::fprintf(stderr, "[TRT] gọi infer() trước allocate() (%s)\n",
                 plan_path_.c_str());
    std::exit(EXIT_FAILURE);
  }
  if (!context_->enqueueV3(stream)) {
    std::fprintf(stderr, "[TRT] enqueueV3 thất bại (%s)\n", plan_path_.c_str());
    std::exit(EXIT_FAILURE);
  }
}

}  // namespace uav
