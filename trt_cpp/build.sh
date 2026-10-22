#!/usr/bin/env bash
# Build uav_reid_trt. Chạy trên thiết bị đích (Jetson) để link đúng TensorRT.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${here}/build"
arch="${CUDA_ARCH:-87}"

cmake -S "${here}" -B "${build_dir}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="${arch}"

cmake --build "${build_dir}" -j"$(nproc)"

echo
echo "xong: ${build_dir}/uav_reid_trt"
