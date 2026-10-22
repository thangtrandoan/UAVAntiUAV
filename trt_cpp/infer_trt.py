"""Chạy `infer.py` với engine TensorRT: video vào, video ra, y nguyên state machine.

KHÔNG sửa `infer.py`. Chỉ thay đúng các điểm nối của nó:

    1. `transform`              (tham số của process_frame) -> ảnh BGR thô
    2. `extract_cnn_feature`                                 -> engine backbone
    3. `compute_fused_vector`                                -> engine temporal
    4. `TwoTierMemoryBank._max_sim` -> chịu được query None (engine chưa xuất
       temporal_token / raw_feat)

Toàn bộ phần còn lại — đọc video, bbox GT, Memory Bank, SOFT/HARD LOCK, vẽ,
`cv2.VideoWriter` — vẫn là code của `infer.py`, nên chỉ có MỘT nguồn sự thật.

Chạy:
    cd trt_cpp
    ../uav_env/bin/python infer_trt.py \\
        --engine-backbone ../v1_backbone.engine \\
        --engine-temporal ../v1_temporal.engine \\
        --config ../configs/config_jetson.yaml \\
        --seq-dir ../data/UAV-Anti-UAV/Test/UAV-Anti-UAV_Test_000169 \\
        --out-dir ../infer_output/trt_000169

Mọi cờ khác được chuyển nguyên cho `infer.py` (`--checkpoint`, `--out-dir`...).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))          # trt_cpp/
PROJ = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, PROJ)

import infer  # noqa: E402
from uav_reid_engine import ReIDEngine  # noqa: E402


class RawCrop:
    """Thay cho `transforms.Compose`.

    `process_frame` gọi `transform(rgb)` rồi `.unsqueeze(0).to(device)` rồi
    `extract_cnn_feature(...)`. Ở đây ảnh gốc được giữ lại trong `last`, còn trả
    về một tensor 1 phần tử (đã nằm sẵn trên GPU) để hai phép sau gần như miễn
    phí. Nhờ vậy bỏ được cả resize/chuẩn hoá PIL trên CPU.
    """

    def __init__(self) -> None:
        self.last: np.ndarray | None = None
        self._dummy: torch.Tensor | None = None

    def __call__(self, rgb: np.ndarray) -> torch.Tensor:
        # engine tạo với bgr=False nên ảnh đi thẳng, không đổi kênh.
        self.last = np.ascontiguousarray(rgb)
        if self._dummy is None:
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            self._dummy = torch.zeros(1, device=dev)
        return self._dummy


TRANSFORM = RawCrop()
ENGINE: ReIDEngine | None = None


def _install(engine: ReIDEngine) -> None:
    global ENGINE
    ENGINE = engine

    # --- 2. backbone ------------------------------------------------------
    def _extract(model, tensor_frame):
        # feat (visual_dim,) float32 trên CPU -> (1, visual_dim) trên đúng device
        # như bản PyTorch, để torch.stack() phía sau không lệch device.
        out = torch.from_numpy(ENGINE.backbone(TRANSFORM.last))
        return out.unsqueeze(0).to(tensor_frame.device)

    infer.extract_cnn_feature = _extract

    # --- 3. temporal + head ----------------------------------------------
    def _fused(model, sliding_window, stride: int = 1):
        seq = sliding_window.get_strided_sequence(stride)      # (1, n, 2560)
        if seq.shape[0] != 1:
            seq = seq[:1]
        visual_mean = sliding_window.get_weighted_visual_mean(stride)
        visual_plain = seq.mean(dim=1)                         # giống training

        arr = np.ascontiguousarray(
            seq.squeeze(0).detach().cpu().numpy(), dtype=np.float32)
        emb = ENGINE.temporal(arr)                             # đã L2-chuẩn-hoá
        fused_feat = torch.from_numpy(emb).unsqueeze(0).to(seq.device)

        # Engine hiện CHỈ xuất reid_embed. Hai vector dưới chỉ dùng cho debug
        # (debug_sim) và cho `fine_space: pre_bn`; xem cảnh báo lúc khởi động.
        return infer.FusedBundle(visual_mean, visual_plain, None, None, fused_feat)

    infer.compute_fused_vector = _fused

    # --- 4. _max_sim chịu None -------------------------------------------
    # temporal_token/raw_feat là None nên F.normalize(None) sẽ nổ; trả 0.0 và
    # phần debug in ra 0.000 (đúng nghĩa "không có số liệu").
    _orig_max_sim = infer.TwoTierMemoryBank._max_sim

    def _max_sim(self, query_feat, key):
        if query_feat is None:
            return 0.0
        return _orig_max_sim(self, query_feat, key)

    infer.TwoTierMemoryBank._max_sim = _max_sim

    # --- 1. transform -----------------------------------------------------
    # `main()` tạo transform rồi truyền vào run_sequence; thay ở đây để không
    # phải sửa `infer.py`.
    _orig_run = infer.run_sequence

    def _run(*args, **kwargs):
        if "transform" in kwargs:
            kwargs["transform"] = TRANSFORM
        else:
            args = list(args)
            if len(args) > 3:
                args[3] = TRANSFORM
            args = tuple(args)
        return _orig_run(*args, **kwargs)

    infer.run_sequence = _run


def main() -> int:
    # Kiểm --help TRƯỚC argparse, vì hai cờ engine là bắt buộc nên argparse sẽ
    # báo lỗi thiếu tham số trước khi kịp in hướng dẫn.
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print(__doc__)
        return 0

    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--engine-backbone", required=True)
    ap.add_argument("--engine-temporal", required=True)
    ap.add_argument("--engine-device", type=int, default=0)
    ap.add_argument("--allow-pad", action="store_true",
                    help="cho phép đệm cửa sổ lên N của engine (điểm sẽ lệch)")
    args, rest = ap.parse_known_args()

    # Đọc num_frames của infer để so với N thật của engine.
    cfg_path = "configs/config_jetson.yaml"
    if "--config" in rest:
        cfg_path = rest[rest.index("--config") + 1]
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except OSError:
        cfg = {}
    inf_cfg = cfg.get("infer", {}) or {}
    want_n = int(inf_cfg.get("num_frames", 16))

    engine = ReIDEngine(args.engine_backbone, args.engine_temporal,
                        num_frames=want_n, device=args.engine_device, bgr=False)
    print("=" * 66)
    print(f"  {engine}")
    print(f"  infer.num_frames = {want_n}")
    if engine.temporal_n != want_n:
        print("=" * 66)
        print(f"  CẢNH BÁO NGHIÊM TRỌNG: engine temporal build với N="
              f"{engine.temporal_n}, còn infer dùng {want_n} frame.")
        print(f"  Đệm {want_n}->{engine.temporal_n} bằng cách lặp frame cuối KHÔNG")
        print("  tương đương PyTorch chạy N=" f"{want_n}, nên cosine sẽ lệch và")
        print("  ngưỡng đã calibrate (0.6667) không còn đúng.")
        print(f"  Build lại engine với N={want_n}:")
        print(f"    trtexec --onnx=v1_temporal.onnx --saveEngine=v1_temporal.engine \\")
        print(f"      --fp16 --minShapes=seq_feats:1x{want_n}x2560 \\")
        print(f"      --optShapes=seq_feats:1x{want_n}x2560 \\")
        print(f"      --maxShapes=seq_feats:12x{want_n}x2560")
        if not args.allow_pad:
            print("  Dừng. Thêm --allow-pad nếu vẫn muốn chạy để xem thử.")
            print("=" * 66)
            return 2
    if str(inf_cfg.get("fine_space", "fused")) != "fused":
        print(f"  CẢNH BÁO: fine_space={inf_cfg.get('fine_space')} cần raw_feat, "
              "mà engine chưa xuất -> gate HARD LOCK sẽ luôn bằng 0.")
    print("=" * 66)

    _install(engine)

    # Chuyển các cờ còn lại cho infer.py.
    sys.argv = [sys.argv[0]] + rest
    infer.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
