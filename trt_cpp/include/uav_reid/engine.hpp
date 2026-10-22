// Bọc một TensorRT engine đã serialize: nạp, khai báo I/O, cấp phát một vùng nhớ
// liên tục cho mọi tensor, và chạy bất đồng bộ trên stream.
//
// Chỉ dùng API V3 (getNbIOTensors / setTensorAddress / enqueueV3) nên cần
// TensorRT >= 8.5. JetPack 5 (TRT 8.5) và JetPack 6 (TRT 10.x) đều chạy được.
#pragma once

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace uav {

class TrtLogger : public nvinfer1::ILogger {
 public:
  explicit TrtLogger(bool verbose = false) : verbose_(verbose) {}
  void set_verbose(bool v) { verbose_ = v; }
  void log(Severity sev, const char* msg) noexcept override;

 private:
  bool verbose_;
};

struct TensorInfo {
  std::string name;
  nvinfer1::TensorIOMode mode{nvinfer1::TensorIOMode::kNONE};
  nvinfer1::DataType dtype{nvinfer1::DataType::kFLOAT};
  nvinfer1::Dims dims{};      // dims cụ thể sau khi đã chốt shape
  bool was_dynamic{false};    // lúc nạp có chiều -1
  size_t bytes{0};
  size_t offset{0};           // vị trí trong arena
  void* external{nullptr};    // khác null thì dùng con trỏ ngoài, không cấp trong arena

  size_t elem_count() const;
  void* ptr(void* arena) const { return external ? external : (char*)arena + offset; }
};

// Một engine + một execution context.
//
// Trình tự dùng:
//   load(...)                  nạp plan, liệt kê I/O
//   set_input_shape(...)       chỉ với engine động; gọi trước allocate()
//   bind_external(...)         tuỳ chọn, để chia sẻ bộ nhớ giữa hai engine
//   allocate(stream)           chốt shape ra, cấp phát, gán địa chỉ tensor
//   infer(stream)              chạy
//
// Shape được giữ cố định sau allocate(). Đổi shape phải gọi lại allocate().
class TrtEngine {
 public:
  TrtEngine() = default;
  ~TrtEngine();
  TrtEngine(const TrtEngine&) = delete;
  TrtEngine& operator=(const TrtEngine&) = delete;

  void load(const std::string& plan_path, int device_id, cudaStream_t stream);
  void set_logger_verbose(bool v) { logger_.set_verbose(v); }

  const std::vector<TensorInfo>& tensors() const { return tensors_; }
  const TensorInfo* find(const std::string& name) const;
  bool has(const std::string& name) const;
  const TensorInfo& at(const std::string& name) const;
  const TensorInfo& only_input() const;
  const TensorInfo& only_output() const;

  bool any_dynamic() const { return any_dynamic_; }
  int profile_count() const { return profile_count_; }
  int profile_index() const { return profile_index_; }

  // Ghi shape đầu vào mong muốn. Có cache: gọi lại cùng shape là no-op.
  void set_input_shape(const std::string& name, const nvinfer1::Dims& d);

  // Dùng con trỏ device bên ngoài cho tensor `name` (ví dụ chia sẻ giữa hai engine).
  void bind_external(const std::string& name, void* device_ptr);

  // Gán LẠI địa chỉ cho một tensor sau allocate(). Dùng cho bộ đệm vòng, nơi
  // cửa sổ đọc dịch theo từng frame nên con trỏ đổi mỗi lần chạy.
  void rebind(const std::string& name, void* device_ptr);

  // In I/O (tên, chiều, dtype, số byte) và min/opt/max của mọi profile.
  // Gọi được ngay sau load(), KHÔNG cần allocate() — nhờ vậy chẩn đoán được cả
  // engine không khớp shape mà không phải dựng cả pipeline.
  void describe() const;

  // min/opt/max của `name` trong profile `profile`. Dims rỗng nếu tên không có.
  nvinfer1::Dims profile_shape(const std::string& name, int profile,
                               nvinfer1::OptProfileSelector which) const;

  // Chốt shape cụ thể cho mọi tensor, cấp phát một arena duy nhất, gán địa chỉ.
  void allocate(cudaStream_t stream);

  void* ptr(const std::string& name) const;
  size_t bytes(const std::string& name) const;
  nvinfer1::Dims dims(const std::string& name) const;

  void infer(cudaStream_t stream);

  size_t arena_bytes() const { return arena_bytes_; }
  const std::string& plan_path() const { return plan_path_; }

 private:
  void index_io_tensors();
  void choose_profile(cudaStream_t stream);
  void resolve_shapes_and_sizes();
  void bind_addresses();

  static size_t dtype_size(nvinfer1::DataType t);
  static size_t volume(const nvinfer1::Dims& d);
  static bool dims_static(const nvinfer1::Dims& d);
  static std::string dims_str(const nvinfer1::Dims& d);

  TrtLogger logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;

  std::vector<TensorInfo> tensors_;
  std::unordered_map<std::string, int> index_;
  std::unordered_map<std::string, nvinfer1::Dims> requested_shapes_;

  void* arena_ = nullptr;
  size_t arena_bytes_ = 0;
  bool any_dynamic_ = false;
  bool allocated_ = false;
  int profile_count_ = 1;
  int profile_index_ = 0;
  std::string plan_path_;
};

}  // namespace uav
