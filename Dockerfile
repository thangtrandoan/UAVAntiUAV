# ==================== Stage 1: Builder ====================
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime AS builder

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ==================== Stage 2: Runtime ====================
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

LABEL org.opencontainers.image.title="uav-antiiuav-reid" \
      org.opencontainers.image.description="UAV ReID training on H100"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    GASNET_PATH=/workspace/gasnet_project \
    GASNET_WEIGHTS=/workspace/gasnet_project/test/gasnet.best.pth

WORKDIR /workspace/UAVAntiUAV

# Copy pip packages từ builder
COPY --from=builder /install /usr/local

# Copy GASNet source (chỉ file cần thiết)
COPY gasnet_src/train.py gasnet_src/dataset.py gasnet_src/utils.py \
     /workspace/gasnet_project/

# Copy GASNet weights
COPY weights/gasnet.best.pth /workspace/gasnet_project/test/gasnet.best.pth

# Copy project source
COPY src/ ./

# Copy configs
COPY configs/ ./

# Entrypoint
COPY entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh

ENTRYPOINT ["/workspace/entrypoint.sh"]
CMD ["--help"]
