"""
calib_lite_20thg10.py — Đo nhanh độ tách GENUINE vs IMPOSTOR trong 4 không gian
(visual / temporal / pre_bn / fused) cho checkpoint resnet50_ibn, để chọn NGƯỠNG
HARD LOCK có cơ sở thay vì đoán. Tái dùng đúng pipeline của infer.py.

Chạy:
  uav_env/bin/python tools/calib_lite_20thg10.py --config configs/config_resnet50_ibn.yaml \
      --n-seqs 16 --out tools/calib_lite_20thg10.json
"""
import os, sys, json, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import yaml
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from infer import (extract_cnn_feature, SlidingWindowBuffer, compute_fused_vector,
                   compute_sharpness, crop_and_pad)
from model import UAVReIDNet, load_checkpoint_verbose


def build_model(cfg, device):
    tr = cfg.get('train', {}); inf = cfg.get('infer', {})
    backbone = inf.get('backbone', tr.get('backbone', 'resnet50_ibn'))
    tt = tr.get('temporal_type', 'mamba'); tp = tr.get('temporal_pool', 'mean')
    tpe = bool(tr.get('temporal_pe', True))
    model = UAVReIDNet(backbone=backbone, temporal_type=tt, temporal_pool=tp, temporal_pe=tpe)
    mp = inf.get('model_path', './best_model_v1.pth')
    load_checkpoint_verbose(model, mp, tag="calib")
    model.to(device).eval()
    return model


def read_seq(seq_dir):
    name = os.path.basename(os.path.normpath(seq_dir))
    mp4 = os.path.join(seq_dir, f"{name}.mp4")
    gt = np.loadtxt(os.path.join(seq_dir, "groundtruth_rect.txt"), delimiter=",").reshape(-1, 4)
    ab_path = os.path.join(seq_dir, "absent.txt")
    absent = []
    if os.path.exists(ab_path):
        with open(ab_path) as f:
            absent = [int(x.strip()) for x in f if x.strip().isdigit()]
    return name, mp4, gt, absent


def seq_windows(model, seq_dir, transform, device, num_frames, stride, pad, max_windows=6):
    """Trả về list dict{visual,temporal,raw,fused} (đã L2-norm) cho các cửa sổ hard-lock."""
    name, mp4, gt, absent = read_seq(seq_dir)
    cap = cv2.VideoCapture(mp4)
    feats = []  # feature 2560 của từng frame HIỆN DIỆN (theo thứ tự)
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        present = (idx >= len(absent) or absent[idx] == 0) and idx < len(gt)
        if present:
            x, y, w, h = [int(v) for v in gt[idx]]
            if w > 0 and h > 0:
                crop = crop_and_pad(frame, (x, y, w, h), pad)
                if crop is not None and crop.size > 0:
                    t = transform(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
                    feats.append((extract_cnn_feature(model, t), compute_sharpness(crop)))
        idx += 1
    cap.release()

    span = (num_frames - 1) * stride + 1   # số frame hiện diện cần cho 1 cửa sổ
    if len(feats) < span:
        return name, []
    windows = []
    starts = np.linspace(0, len(feats) - span, num=max_windows, dtype=int)
    for s in sorted(set(starts.tolist())):
        buf = SlidingWindowBuffer(num_frames, stride)
        for k in range(span):
            f, sh = feats[s + k]
            buf.add(f, sh)
        b = compute_fused_vector(model, buf, stride=stride)
        windows.append({
            'visual':   F.normalize(b.visual_plain, p=2, dim=1).squeeze(0).cpu(),
            'temporal': F.normalize(b.temporal_token, p=2, dim=1).squeeze(0).cpu(),
            'raw':      b.raw_feat.squeeze(0).cpu(),     # đã normalize trong compute
            'fused':    b.fused_feat.squeeze(0).cpu(),
        })
    return name, windows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/config_resnet50_ibn.yaml')
    ap.add_argument('--test-dir', default='./data/UAV-Anti-UAV/Test')
    ap.add_argument('--n-seqs', type=int, default=16)
    ap.add_argument('--max-windows', type=int, default=6)
    ap.add_argument('--out', default='tools/calib_lite_20thg10.json')
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    tr = cfg.get('train', {}); dp = cfg.get('data_pipeline', {})
    num_frames = tr.get('num_frames', 12)
    stride = dp.get('frame_stride', 4)
    pad = dp.get('bbox_padding', 0.2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(cfg, device)
    transform = transforms.Compose([
        transforms.ToPILImage(), transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    all_dirs = sorted(os.path.join(args.test_dir, d) for d in os.listdir(args.test_dir)
                      if os.path.isdir(os.path.join(args.test_dir, d)))
    per_seq = {}   # name -> list windows
    for d in all_dirs:
        if len(per_seq) >= args.n_seqs:
            break
        name, wins = seq_windows(model, d, transform, device, num_frames, stride, pad, args.max_windows)
        if len(wins) >= 2:
            per_seq[name] = wins
            print(f"  {name}: {len(wins)} windows")

    spaces = ['visual', 'temporal', 'raw', 'fused']
    gen = {s: [] for s in spaces}; imp = {s: [] for s in spaces}
    names = list(per_seq)
    for i, na in enumerate(names):
        wa = per_seq[na]
        # genuine: các cửa sổ cùng seq
        for x in range(len(wa)):
            for y in range(x + 1, len(wa)):
                for s in spaces:
                    gen[s].append(float(torch.dot(wa[x][s], wa[y][s])))
        # impostor: cặp seq khác nhau (lấy cửa sổ đầu để gọn)
        for nb in names[i + 1:]:
            wb = per_seq[nb]
            for x in wa[:3]:
                for y in wb[:3]:
                    for s in spaces:
                        imp[s].append(float(torch.dot(x[s], y[s])))

    report = {'config': args.config, 'n_seqs': len(per_seq),
              'num_frames': num_frames, 'stride': stride, 'spaces': {}}
    print("\n" + "=" * 78)
    print(f"{'space':<10} {'gen_med':>8} {'gen_p10':>8} {'imp_med':>8} {'imp_p99':>8} "
          f"{'thr@FAR1%':>10} {'TAR':>7}")
    print("-" * 78)
    for s in spaces:
        g = np.array(gen[s]); m = np.array(imp[s])
        thr = float(np.percentile(m, 99))        # ngưỡng để FAR ~ 1%
        tar = float((g >= thr).mean())
        report['spaces'][s] = {
            'gen_median': float(np.median(g)), 'gen_p10': float(np.percentile(g, 10)),
            'gen_min': float(g.min()), 'imp_median': float(np.median(m)),
            'imp_p99': float(np.percentile(m, 99)), 'imp_max': float(m.max()),
            'thr_far1pct': thr, 'tar_at_far1pct': tar,
            'n_gen': int(g.size), 'n_imp': int(m.size)}
        print(f"{s:<10} {np.median(g):>8.3f} {np.percentile(g,10):>8.3f} "
              f"{np.median(m):>8.3f} {np.percentile(m,99):>8.3f} {thr:>10.3f} {tar:>7.1%}")
    print("=" * 78)
    with open(args.out, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nĐã ghi: {args.out}")


if __name__ == "__main__":
    main()
