#!/usr/bin/env bash
# Chạy trên server H100
# Cần: nvidia-docker hoặc --gpus all

docker run --gpus all --rm -it \
  --shm-size=16g \
  -v /path/to/processed_data:/data/processed \
  -v $(pwd)/checkpoints:/workspace/UAVAntiUAV/checkpoints \
  -v $(pwd)/logs:/workspace/UAVAntiUAV/logs \
  uav-reid:latest \
  --config config_h100.yaml \
  --mode train
