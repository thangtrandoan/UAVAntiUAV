"""Ví dụ: chạy engine TensorRT C++ và XUẤT VIDEO, thay cho đoạn PyTorch.

Đây là khuôn mẫu để áp vào `infer.py`, không phải bản sao state machine của nó.
Script cố tình KHÔNG chép logic SOFT/HARD LOCK, Memory Bank hay ngưỡng — những
phần đó phải giữ nguyên trong `infer.py` để chỉ có một nguồn sự thật.

Chạy:
    cd trt_cpp
    python example_swap.py --backbone v1_backbone.engine --temporal v1_temporal.engine \\
        --num-frames 16 --video /duong/dan/video.mp4 --out demo.mp4

Thêm `--bbox x y w h` để chỉ lấy một vùng, hoặc bỏ trống để lấy cả khung hình.
"""

from __future__ import annotations

import argparse
import time
from collections import deque

try:
    import cv2
except ImportError as exc:  # cv2 biên dịch cho numpy 1.x nhưng numpy đang 2.x
    raise SystemExit(
        "không import được cv2 — đây là lỗi MÔI TRƯỜNG, không phải lỗi script.\n"
        "Triệu chứng: 'numpy.core.multiarray failed to import' hoặc\n"
        "  'A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x'.\n"
        "Nguyên nhân: cv2 biên dịch cho numpy 1.x nhưng numpy đang là 2.x (ABI khác).\n"
        "Cách sửa: dùng đúng trình thông dịch của dự án (numpy 1.26 + cv2 4.5):\n"
        "    ../uav_env/bin/python example_swap.py ...\n"
        "uav_reid_engine.py KHÔNG cần cv2 và chạy được trên cả numpy 1.x lẫn 2.x.\n"
    ) from exc

import numpy as np

from uav_reid_engine import ReIDEngine


def crop_and_pad(frame: np.ndarray, bbox, padding: float = 0.0):
    """Cắt vùng bbox, nới `padding` theo tỉ lệ, kẹp trong khung hình."""
    h, w = frame.shape[:2]
    x, y, bw, bh = bbox
    px, py = bw * padding, bh * padding
    x0 = max(0, int(x - px))
    y0 = max(0, int(y - py))
    x1 = min(w, int(x + bw + px))
    y1 = min(h, int(y + bh + py))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return frame[y0:y1, x0:x1]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cả hai đã L2-chuẩn-hoá nên tích vô hướng chính là cosine."""
    return float(np.dot(a, b))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--temporal", required=True)
    ap.add_argument("--num-frames", type=int, default=16)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="demo.mp4")
    ap.add_argument("--bbox", type=int, nargs=4, default=None,
                    metavar=("X", "Y", "W", "H"))
    ap.add_argument("--padding", type=float, default=0.0)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    engine = ReIDEngine(args.backbone, args.temporal, num_frames=args.num_frames,
                        device=args.device, bgr=True)
    print(engine)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"không mở được video: {args.video}")
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    print(f"video {w}x{h} @ {fps:.1f} fps -> {args.out}")

    # Bộ đệm vòng phía Python: giữ N vector feature gần nhất.
    ring: deque[np.ndarray] = deque(maxlen=args.num_frames)
    prev_emb: np.ndarray | None = None
    idx = 0
    t_frame = 0.0
    t_emb = 0.0
    n_emb = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.max_frames and idx >= args.max_frames:
            break

        bbox = args.bbox if args.bbox else (0, 0, w, h)
        crop = crop_and_pad(frame, bbox, args.padding)
        if crop is None:
            writer.write(frame)
            idx += 1
            continue

        # --- thay cho transform(...) + model.backbone(...) của PyTorch ---
        # Dùng đường STATELESS (backbone + temporal): cửa sổ do phía Python giữ,
        # đúng như SlidingWindowBuffer của infer.py. Không trộn với push()/embed()
        # vì đó là bộ đệm vòng bên trong C++.
        t0 = time.perf_counter()
        feat = engine.backbone(crop)          # (2560,) float32
        ring.append(feat)
        t1 = time.perf_counter()
        t_frame += (t1 - t0) * 1000.0

        emb = None
        if len(ring) == args.num_frames:
            seq = np.stack(ring, axis=0)      # (N, 2560)
            t2 = time.perf_counter()
            emb = engine.temporal(seq)        # (3072,) đã chuẩn hoá
            t3 = time.perf_counter()
            t_emb += (t3 - t2) * 1000.0
            n_emb += 1

        # --- vẽ ---
        x, y, bw, bh = bbox
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        cv2.putText(frame, f"frame {idx}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"buf {len(ring)}/{args.num_frames}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        if emb is not None:
            sim = cosine(emb, prev_emb) if prev_emb is not None else 1.0
            prev_emb = emb
            cv2.putText(frame, f"cos {sim:.4f}", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        writer.write(frame)
        idx += 1

    cap.release()
    writer.release()

    print(f"\n  {idx} frame -> {args.out}")
    if idx:
        print(f"  backbone + tiền xử lý : {t_frame / idx:7.3f} ms/frame")
    if n_emb:
        print(f"  temporal              : {t_emb / n_emb:7.3f} ms/embed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
