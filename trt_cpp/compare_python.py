"""Đối chiếu embedding của engine C++ (TensorRT) với PyTorch cho CÙNG một clip.

Đây mới là phép kiểm "đúng chưa". Cosine giữa hai frame liền kề trong
`example_swap.py` cao là chuyện bình thường (cùng một vật, cách nhau 1/30 giây),
nó chỉ kiểm bộ đệm có trượt hay không, KHÔNG nói gì về tính đúng đắn.

So ở HAI tầng để biết lệch ở đâu nếu có:
    1. backbone  (2560) — kiểm tiền xử lý + engine backbone
    2. fused     (3072) — kiểm cả temporal + head

Chạy (phải dùng uav_env vì có torch):
    ../uav_env/bin/python compare_python.py \\
        --config ../configs/config_jetson.yaml \\
        --model ../checkpoints/best_model.pth \\
        --backbone ../v1_backbone.engine --temporal ../v1_temporal.engine \\
        --video ../data/UAV-Anti-UAV/Test/UAV-Anti-UAV_Test_000169/UAV-Anti-UAV_Test_000169.mp4 \\
        --start 100 --bbox 640 300 200 160 \\
        --other ../data/UAV-Anti-UAV/Test/UAV-Anti-UAV_Test_000001/UAV-Anti-UAV_Test_000001.mp4
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

# GASNET_PATH phải đặt TRƯỚC khi import model (giống evaluate_reid.py).
import yaml


def _set_gasnet(cfg: dict) -> None:
    gasnet_dir = (cfg.get("paths") or {}).get("gasnet_dir", "")
    if gasnet_dir:
        os.environ["GASNET_PATH"] = os.path.abspath(gasnet_dir)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True, help="checkpoint .pth")
    ap.add_argument("--backbone", required=True, help="v1_backbone.engine")
    ap.add_argument("--temporal", required=True, help="v1_temporal.engine")
    ap.add_argument("--num-frames", type=int, default=16)
    ap.add_argument("--frames", default=None, help="thư mục ảnh (ưu tiên)")
    ap.add_argument("--video", default=None)
    ap.add_argument("--start", type=int, default=0, help="frame bắt đầu trong video")
    ap.add_argument("--bbox", type=int, nargs=4, default=None,
                    metavar=("X", "Y", "W", "H"))
    ap.add_argument("--backbone-name", default=None,
                    choices=["resnet50_ibn", "dinov3_convnext"],
                    help="ghi đè; mặc định suy từ chiều đặc trưng của engine")
    ap.add_argument("--econvnext", action="store_true",
                    help="bật E-ConvNeXt (chỉ dùng với dinov3_convnext)")
    ap.add_argument("--temporal-type", default=None,
                    choices=["mamba", "attention"],
                    help="phải khớp lúc train, nếu không cosine sẽ lệch vô cớ")
    ap.add_argument("--temporal-pool", default=None, choices=["mean", "attn"])
    ap.add_argument("--temporal-pe", default=None, type=int, choices=[0, 1])
    ap.add_argument("--other", default=None,
                    help="video/thư mục của vật KHÁC, để đo cosine impostor")
    ap.add_argument("--other-bbox", type=int, nargs=4, default=None,
                    metavar=("X", "Y", "W", "H"))
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    _set_gasnet(cfg)

    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torchvision import transforms

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, root)
    from model import UAVReIDNet, load_checkpoint_verbose
    from uav_reid_engine import ReIDEngine

    train = cfg.get("train", {}) or {}
    ev = cfg.get("eval", {}) or {}
    temporal_type = args.temporal_type or ev.get(
        "temporal_type", train.get("temporal_type", "mamba"))
    temporal_pool = args.temporal_pool or ev.get(
        "temporal_pool", train.get("temporal_pool", "attn"))
    temporal_pe = (bool(ev.get("temporal_pe", train.get("temporal_pe", True)))
                   if args.temporal_pe is None else bool(args.temporal_pe))

    # Mở engine TRƯỚC để biết chiều đặc trưng thật, rồi suy kiến trúc backbone.
    # Config trong repo có thể không phải config đã dùng để train, nên chiều của
    # engine là nguồn tin đáng tin hơn.
    eng = ReIDEngine(args.backbone, args.temporal, num_frames=0, bgr=True)
    print(f"  {eng}")
    n_win = eng.num_frames
    if args.num_frames and args.num_frames != n_win:
        print(f"  CẢNH BÁO: --num-frames={args.num_frames} nhưng engine build với "
              f"N={n_win}; dùng N={n_win} để so sánh cho công bằng")
    by_dim = {2560: "resnet50_ibn", 960: "dinov3_convnext"}
    guessed = by_dim.get(eng.visual_dim)
    backbone_name = args.backbone_name or guessed or "resnet50_ibn"
    if guessed is None:
        print(f"  CẢNH BÁO: visual_dim={eng.visual_dim} không khớp kiến trúc nào "
              "đã biết (2560=resnet50_ibn, 960=dinov3_convnext)")
    elif args.backbone_name and args.backbone_name != guessed:
        print(f"  CẢNH BÁO: --backbone-name={args.backbone_name} nhưng engine có "
              f"visual_dim={eng.visual_dim} (ứng với {guessed})")
    econvnext = bool(args.econvnext) or (backbone_name == "dinov3_convnext")
    print(f"  backbone={backbone_name} temporal={temporal_type} "
          f"pool={temporal_pool} pe={temporal_pe} econvnext={econvnext}")
    print("  (nếu config khác lần train thật, ghi đè bằng --backbone-name/"
          "--temporal-type/--temporal-pool/--temporal-pe)")

    model = UAVReIDNet(econvnext=econvnext, freeze_backbone=False,
                       backbone=backbone_name, temporal_type=temporal_type,
                       temporal_pool=temporal_pool, temporal_pe=temporal_pe)
    load_checkpoint_verbose(model, args.model, tag="cmp")
    model.cuda().eval()

    # Đúng chuỗi của evaluate_reid.py:283-288.
    tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    def grab(path: str, start: int, bbox, n: int):
        """Lấy n crop BGR. path có thể là thư mục ảnh hoặc file video."""
        out = []
        if os.path.isdir(path):
            files = sorted(glob.glob(os.path.join(path, "*")))
            files = [f for f in files if f.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp"))]
            for f in files[:n]:
                img = cv2.imread(f, cv2.IMREAD_COLOR)
                if bbox:
                    x, y, w, h = bbox
                    img = img[y:y + h, x:x + w]
                out.append(np.ascontiguousarray(img))
            return out
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise SystemExit(f"không mở được video: {path}")
        if start:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        while len(out) < n:
            ok, img = cap.read()
            if not ok:
                break
            if bbox:
                x, y, w, h = bbox
                img = img[y:y + h, x:x + w]
            out.append(np.ascontiguousarray(img))
        cap.release()
        return out

    crops = grab(args.frames or args.video, args.start, args.bbox, n_win)
    if not crops:
        raise SystemExit("không lấy được frame nào")
    if len(crops) < n_win:
        print(f"  CẢNH BÁO: chỉ có {len(crops)} frame < N={n_win}; "
              "sẽ lặp frame cuối (cùng quy ước hai bên)")

    def cos(a, b):
        return float(np.dot(a.ravel(), b.ravel()) /
                     (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    # ---- 1. Backbone: PyTorch vs C++ ----
    py_feats, cpp_feats = [], []
    for c in crops:
        t = tf(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)).unsqueeze(0).cuda()
        with torch.no_grad():
            fo = model.backbone(t)
            if isinstance(fo, tuple):
                g, s = (fo[0][0], fo[0][1]) if isinstance(fo[0], tuple) else (fo[0], fo[1])
                fo = torch.cat([g, s], dim=-1)
        py_feats.append(fo.squeeze(0).cpu().numpy())
        cpp_feats.append(eng.backbone(c))

    d = [cos(p, q) for p, q in zip(py_feats, cpp_feats)]
    l2 = [float(np.max(np.abs(p - q))) for p, q in zip(py_feats, cpp_feats)]
    print("\n  [1] BACKBONE (2560)")
    print(f"      cosine  min={min(d):.6f}  mean={sum(d)/len(d):.6f}")
    print(f"      |diff|  max={max(l2):.4e}")

    # ---- 2. Fused: PyTorch vs C++ ----
    pad = list(cpp_feats)
    while len(pad) < n_win:
        pad.append(pad[-1])

    py_seq = torch.from_numpy(np.stack(py_feats)).unsqueeze(0).cuda()
    if py_seq.shape[1] < n_win:
        k = n_win - py_seq.shape[1]
        py_seq = torch.cat([py_seq, py_seq[:, -1:].repeat(1, k, 1)], dim=1)
    with torch.no_grad():
        py_fused = F.normalize(model(py_seq), p=2, dim=1).squeeze(0).cpu().numpy()

    cpp_fused = eng.temporal(np.stack(pad))
    print("\n  [2] FUSED (3072)")
    print(f"      cosine = {cos(py_fused, cpp_fused):.6f}   "
          f"(>= 0.999 là đạt)")
    print(f"      |diff| max = {float(np.max(np.abs(py_fused - cpp_fused))):.4e}")
    print(f"      |py| = {np.linalg.norm(py_fused):.6f}   "
          f"|cpp| = {np.linalg.norm(cpp_fused):.6f}  (phải = 1.0)")

    # ---- 3. Impostor: cùng clip nhưng vật khác ----
    if args.other:
        other = grab(args.other, 0, args.other_bbox, n_win)
        if len(other) >= 2:
            of = [eng.backbone(c) for c in other]
            while len(of) < n_win:
                of.append(of[-1])
            o_fused = eng.temporal(np.stack(of[:n_win]))
            print("\n  [3] PHÂN TÁCH")
            print(f"      cosine với vật KHÁC = {cos(cpp_fused, o_fused):.6f}")
            print("      kỳ vọng: thấp hơn hẳn ngưỡng đã calibrate (0.6667)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
