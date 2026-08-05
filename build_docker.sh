#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/.docker_build"

echo "=== Chuẩn bị Docker build context ==="
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"/{gasnet_src,weights,src,configs}

# GASNet source (chỉ 3 file cần thiết)
GASNET="$SCRIPT_DIR/../UAV/gasnet_project"
cp "$GASNET"/{train.py,dataset.py,utils.py} "$BUILD_DIR/gasnet_src/"

# GASNet weights (~242MB)
cp "$GASNET/test/gasnet.best.pth" "$BUILD_DIR/weights/"

# Project source
cp "$SCRIPT_DIR"/{model.py,train_reid.py,evaluate_reid.py,data_pipeline.py,process_rgbt_pipeline.py} "$BUILD_DIR/src/"

# Configs
cp "$SCRIPT_DIR"/{config_jetson.yaml,config_local.yaml,config_h100.yaml} "$BUILD_DIR/configs/"

# Build files
cp "$SCRIPT_DIR"/{Dockerfile,requirements.txt,entrypoint.sh,.dockerignore} "$BUILD_DIR/"

echo "=== Build context size ==="
du -sh "$BUILD_DIR"

echo "=== Building Docker image ==="
docker build -t uav-reid:latest "$BUILD_DIR"

echo ""
echo "=== DONE ==="
echo "Image size:"
docker images uav-reid:latest --format '{{.Size}}'
echo ""
echo "Export: docker save uav-reid:latest | gzip > uav-reid.tar.gz"
