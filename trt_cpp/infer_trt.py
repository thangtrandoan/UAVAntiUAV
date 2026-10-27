"""Chạy `infer.py` với engine TensorRT: video vào, video ra, y nguyên state machine.

KHÔNG sửa `infer.py`. Chỉ thay đúng các điểm nối của nó:

    1. `transform`              (tham số của process_frame) -> ảnh BGR thô
    2. `extract_cnn_feature`                                 -> engine backbone
    3. `compute_fused_vector`                                -> engine temporal
    4. `TwoTierMemoryBank._max_sim` -> chịu được query None (engine chưa xuất
       temporal_token / raw_feat)

Toàn bộ phần còn lại — đọc video, bbox GT, Memory Bank, SOFT/HARD LOCK, vẽ,
`cv2.VideoWriter` — vẫn là code của `infer.py`, nên chỉ có MỘT nguồn sự thật.

Chạy (mọi thứ khác đọc từ config, gồm cả đường dẫn engine):
    uav_env/bin/python trt_cpp/infer_trt.py \\
        --config configs/config_resnet50_ibn.yaml

Ghi đè khi cần:
    --engine-backbone / --engine-temporal   (mặc định: trt.backbone_engine / trt.temporal_engine)
    --num-frames                            (mặc định: infer.num_frames)

Mọi cờ khác được chuyển nguyên cho `infer.py` (`--seq-dir`, `--checkpoint`,
`--out-dir`...); thiếu thì `infer.py` tự lấy `infer.seq_dir` / `infer.model_path`
/ `infer.out_dir` trong config, nên bình thường không phải gõ.

Quét nhiều N: engine temporal bị khóa cứng theo N lúc export (vòng lặp selective
scan bị unroll), nên mỗi N cần một engine riêng. Đổi N khi chạy bằng:

    --num-frames 8   # ghi đè infer.num_frames, phải khớp N của engine
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

# Engine hiện chỉ xuất `reid_embed`; `temporal_token` / `raw_feat` không có nên
# hai tầng điểm đó KHÔNG đo được. Bật lên khi export temporal kèm đầu ra
# `temporal_token` và bổ sung đường đọc tương ứng ở C++.
_HAS_TOKEN = False


_PATH_FLAGS = ("--config", "--seq-dir", "--out-dir", "--checkpoint")


def _absolutize(argv, base: str):
    """Đổi giá trị của các cờ đường dẫn sang tuyệt đối theo `base`."""
    out = list(argv)
    for i, a in enumerate(out[:-1]):
        if a in _PATH_FLAGS:
            v = out[i + 1]
            if v and not os.path.isabs(v):
                out[i + 1] = os.path.normpath(os.path.join(base, v))
    return out


def _install(engine: ReIDEngine, force_n: int = 0) -> None:
    global ENGINE
    ENGINE = engine

    # `--num-frames` ghi đè N ngay trong `resolve_pipeline` của infer.py, để
    # sweep nhiều N mà không phải sửa config mỗi lần.
    if force_n:
        _orig_resolve = infer.resolve_pipeline

        def _resolve(cfg, section, script="", verbose=True):
            sec = _orig_resolve(cfg, section, script, verbose)
            if section == "infer":
                sec["num_frames"] = force_n
            return sec

        infer.resolve_pipeline = _resolve

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
    # temporal_token/raw_feat là None nên F.normalize(None) sẽ nổ. Trả NaN chứ
    # KHÔNG trả 0.0: 0.0 trông như "điểm thật bằng 0" và dẫn tới kết luận sai,
    # còn NaN in ra là `nan` — thấy ngay là không có số liệu.
    _orig_max_sim = infer.TwoTierMemoryBank._max_sim

    def _max_sim(self, query_feat, key):
        if query_feat is None:
            return float("nan")
        return _orig_max_sim(self, query_feat, key)

    infer.TwoTierMemoryBank._max_sim = _max_sim

    # --- 4b. không để bảng tổng kết kết luận sai --------------------------
    # `format_bn_debug_lines` (infer.py:654) thấy pre_mean = 0.0 < 0.70 nên in
    # "raw cũng thấp sẵn -> lỗi nằm ở feature (temporal/dữ liệu train)". Với
    # engine, 0.0 đó KHÔNG phải phép đo mà là "không có dữ liệu" -> kết luận
    # hoàn toàn sai. Thay bằng báo cáo trung thực.
    _orig_bn_lines = infer.format_bn_debug_lines

    def _bn_lines(pre_scores, post_scores, thr, tag_counts=None,
                  n_false_alarms=None):
        if _HAS_TOKEN:
            return _orig_bn_lines(pre_scores, post_scores, thr, tag_counts,
                                  n_false_alarms)
        post = np.asarray(post_scores or [], dtype=np.float64)
        out = ["Sim PRE-BN  (raw concat)   : KHONG CO SO LIEU — engine chi xuat "
               "reid_embed"]
        if post.size:
            out.append(f"Sim POST-BN (fused/fine)   : {post.mean():.3f} "
                       f"(n={post.size})")
            out.append(f"Cua so vuot nguong {thr:.2f}    : fused "
                       f"{float(np.mean(post >= thr)) * 100.0:.1f}%")
        # Giữ dòng phân bố theo tag (hữu ích, không liên quan tới hai tầng thiếu).
        if tag_counts:
            n_re = int(tag_counts.get("re-acquire", 0))
            n_ah = int(tag_counts.get("anti-hijack", 0))
            extra = (f"Phan bo theo tag           : re-acquire {n_re} | "
                     f"anti-hijack {n_ah}")
            if n_false_alarms is not None and n_re > 0:
                n_lock = max(0, n_re - int(n_false_alarms))
                extra += (f"   -> HARD LOCK {n_lock}/{n_re} = "
                          f"{100.0 * n_lock / n_re:.1f}% (chi tinh re-acquire)")
            out.append(extra)
        return out

    infer.format_bn_debug_lines = _bn_lines

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
    ap.add_argument("--engine-backbone", default=None,
                    help="mặc định: trt.backbone_engine trong config")
    ap.add_argument("--engine-temporal", default=None,
                    help="mặc định: trt.temporal_engine trong config")
    ap.add_argument("--engine-device", type=int, default=None,
                    help="mặc định: trt.device trong config")
    ap.add_argument("--num-frames", type=int, default=0,
                    help="ghi đè infer.num_frames (phải khớp N của engine)")
    ap.add_argument("--allow-pad", action="store_true",
                    help="cho phép đệm cửa sổ lên N của engine (điểm sẽ lệch)")
    args, rest = ap.parse_known_args()

    # Giữ CWD lúc gọi: đường dẫn người dùng gõ trên dòng lệnh hiểu theo chỗ họ
    # đứng, còn đường dẫn trong config hiểu theo gốc repo (xem _pick bên dưới).
    cwd0 = os.getcwd()

    # Đọc num_frames của infer để so với N thật của engine.
    cfg_path = os.path.join(PROJ, "configs", "config_jetson.yaml")
    if "--config" in rest:
        cfg_path = rest[rest.index("--config") + 1]
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except OSError:
        cfg = {}
    inf_cfg = cfg.get("infer", {}) or {}
    # Phải đi qua pipeline_lock: `num_frames` có thể được đè ở block `infer`, còn
    # thiếu thì mới lấy `train.num_frames`. Đọc YAML thô sẽ ra giá trị khác với
    # giá trị `infer.py` thực dùng.
    try:
        from pipeline_lock import resolve_pipeline
        inf_cfg = resolve_pipeline(cfg, "infer", script="infer_trt.py",
                                   verbose=False)
    except ImportError:
        print("  CẢNH BÁO: không import được pipeline_lock; đọc thẳng block infer")
    if inf_cfg.get("num_frames") is None:
        inf_cfg = dict(inf_cfg)
        inf_cfg["num_frames"] = (cfg.get("train") or {}).get("num_frames", 16)
    want_n = int(inf_cfg["num_frames"])
    force_n = int(args.num_frames or 0)
    if force_n:
        want_n = force_n

    trt_cfg = cfg.get("trt") or {}

    def _pick(cli_val, cfg_val):
        """CLI tính theo CWD lúc gọi, config tính theo gốc repo."""
        for val, base in ((cli_val, cwd0), (cfg_val, PROJ)):
            if val:
                return os.path.normpath(
                    val if os.path.isabs(val) else os.path.join(base, val))
        return None

    eng_bb = _pick(args.engine_backbone, trt_cfg.get("backbone_engine"))
    eng_tp = _pick(args.engine_temporal, trt_cfg.get("temporal_engine"))
    if not eng_bb or not eng_tp:
        print("  Thiếu đường dẫn engine. Truyền --engine-backbone/--engine-temporal")
        print("  hoặc khai trong config:")
        print("    trt:")
        print("      backbone_engine: ./v1_backbone.engine")
        print("      temporal_engine: ./v1_temporal.engine")
        return 2
    dev = args.engine_device
    if dev is None:
        dev = int(trt_cfg.get("device", 0))

    # num_frames=0: lấy N từ engine, rồi mới so với config — nếu truyền want_n
    # vào thì Pipeline sẽ yêu cầu shape N=want_n và dừng ngay khi engine khác.
    engine = ReIDEngine(eng_bb, eng_tp, num_frames=0, device=dev, bgr=False)
    print("=" * 66)
    print(f"  {engine}")
    print(f"  engine backbone = {os.path.basename(eng_bb)}")
    print(f"  engine temporal = {os.path.basename(eng_tp)}")
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

    _install(engine, force_n)

    # Đường dẫn trong config (`./data`, `./best_model_v1.pth`, `./infer_output`)
    # đều tính từ gốc repo, nên chạy infer.py ở đó; đường dẫn người dùng gõ thì
    # đổi sang tuyệt đối trước để không đổi nghĩa.
    rest = _absolutize(rest, cwd0)
    if os.path.abspath(os.getcwd()) != PROJ:
        os.chdir(PROJ)

    # Chuyển các cờ còn lại cho infer.py.
    sys.argv = [sys.argv[0]] + rest
    infer.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
