"""
calibrate_threshold.py
======================
Tách test set thành 2 phần (sequence-level, không leak):
  - cal  (calibration split) : tìm threshold tại FAR <= far_target
  - eval (holdout split)     : báo cáo TAR/FAR thật sự với threshold đã chọn

Chiến lược: Fixed FAR
  Tìm ngưỡng t* từ cal-split sao cho FAR(t*) <= far_target.
  Áp t* lên eval-split -> TAR@FAR không bị thiên lệch.

Output:
  <output_dir>/
    calibrated_threshold.json   <- threshold + metadata ('threshold' = không gian chính,
                                   'thresholds' = {fused, pre_bn} để dùng với evaluate_reid.py --space)
    score_distribution_<space>.png  <- histogram genuine vs impostor (cal split)
    det_curve_<space>.png           <- DET curve với t* được đánh dấu (cal split)
    roc_curve_comparison_<space>.png<- ROC cal vs eval trên cùng 1 plot
    calibration_report.json     <- full report ('spaces' = bảng so sánh 2 không gian)

Không gian feature (🛠️ 14/9):
  fused  (mặc định, hành vi cũ) : qua ReIDHead (BatchNorm1d)
  pre_bn                        : cat(visual, temporal) — ĐẦU VÀO bnneck (raw)
  Cả hai được calibrate trong cùng 1 lượt chạy với CÙNG một quy trình Fixed-FAR, để trả lời
  dứt khoát: BatchNorm1d có thật sự làm mất khả năng phân biệt, hay chỉ nén thang điểm?
  (xem md/15thg9.md)

Usage:
  python calibrate_threshold.py --config configs/config_local.yaml \\
      --cal-ratio 0.4 --far-target 0.001 --output-dir calib_results
"""

import os
import sys
import json
import yaml
import argparse
import time
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib không có - bỏ qua vẽ đồ thị. Cài bằng: pip install matplotlib")


# Dataset
class CalibDataset(Dataset):
    def __init__(self, data_dir, query_json, gallery_json,
                 allowed_seq_ids=None, transform=None, num_frames=16):
        self.data_dir = data_dir
        with open(query_json, 'r') as f:
            queries = json.load(f)
        with open(gallery_json, 'r') as f:
            galleries = json.load(f)

        self.transform = transform
        self.num_frames = num_frames

        g_dict = {(g['sequence_id'], g['event_index']): g for g in galleries}
        self.valid_pairs = []
        for q in queries:
            if allowed_seq_ids is not None and q['sequence_id'] not in allowed_seq_ids:
                continue
            key = (q['sequence_id'], q['event_index'])
            if key in g_dict:
                g = g_dict[key]
                if q.get('identity_id') is not None:
                    self.valid_pairs.append({
                        'identity_id': q['identity_id'],
                        'gallery_frames': g['frames'],
                        'gallery_dir': g['frame_dir'],
                        'query_frames': q['frames'],
                        'query_dir': q['frame_dir'],
                    })

    def __len__(self):
        return len(self.valid_pairs)

    def _load_clip(self, folder, frames):
        if len(frames) > self.num_frames:
            indices = np.linspace(0, len(frames) - 1, self.num_frames).astype(int)
            frames = [frames[i] for i in indices]
        elif len(frames) < self.num_frames:
            if len(frames) == 0:
                return torch.zeros((self.num_frames, 3, 224, 224))
            while len(frames) < self.num_frames:
                frames.append(frames[-1])
        clip = []
        for fn in frames:
            path = os.path.join(self.data_dir, folder, fn)
            try:
                img = Image.open(path).convert('RGB')
            except Exception:
                img = Image.new('RGB', (256, 256), (0, 0, 0))
            if self.transform:
                img = self.transform(img)
            clip.append(img)
        return torch.stack(clip, dim=0)

    def __getitem__(self, idx):
        pair = self.valid_pairs[idx]
        before_clip = self._load_clip(pair['gallery_dir'], list(pair['gallery_frames']))
        after_clip  = self._load_clip(pair['query_dir'],   list(pair['query_frames']))
        pid = pair['identity_id']
        return before_clip, after_clip, pid


# Feature extraction
def extract_features(model, dataloader, backbone_only=False, spaces=('fused',)):
    """
    Trích feature cho (các) không gian đánh giá trong MỘT lượt chạy backbone.

    spaces:
      'fused'    : qua ReIDHead (BatchNorm1d) — pipeline hiện tại (fine score)
      'pre_bn'   : cat(visual, temporal) — ĐẦU VÀO của bnneck (raw);
                   dùng để kiểm chứng giả thuyết "BatchNorm1d phá cosine" (xem md/15thg9.md)
      'backbone' : chỉ visual backbone (chỉ dùng khi backbone_only=True)

    Return: (feats, pids) với feats[space] = (qf, gf) đã concat theo batch.
    """
    acc = {s: {'qf': [], 'gf': []} for s in spaces}
    pids = []
    with torch.no_grad():
        for before, after, pid in dataloader:
            before, after = before.cuda(), after.cuda()
            if backbone_only:
                acc['backbone']['gf'].append(model(before, backbone_only=True).cpu())
                acc['backbone']['qf'].append(model(after,  backbone_only=True).cpu())
            else:
                v_g, t_g, _ = model.extract_features(before)
                v_q, t_q, _ = model.extract_features(after)
                if 'fused' in spaces:
                    acc['fused']['gf'].append(model.head(v_g, t_g).cpu())
                    acc['fused']['qf'].append(model.head(v_q, t_q).cpu())
                if 'pre_bn' in spaces:
                    acc['pre_bn']['gf'].append(torch.cat([v_g, t_g], dim=-1).cpu())
                    acc['pre_bn']['qf'].append(torch.cat([v_q, t_q], dim=-1).cpu())
            pids.extend(pid.numpy().tolist())

    feats = {s: (torch.cat(acc[s]['qf'], dim=0), torch.cat(acc[s]['gf'], dim=0)) for s in spaces}
    return feats, np.array(pids)


# Score computation
def compute_scores(qf, gf, q_pids, g_pids):
    """Trả về genuine_scores, impostor_scores (numpy arrays) từ pairwise cosine sim."""
    qf_n = F.normalize(qf, p=2, dim=1).numpy()
    gf_n = F.normalize(gf, p=2, dim=1).numpy()
    sim = qf_n @ gf_n.T  # (num_q, num_g)

    genuine, impostor = [], []
    for i in range(len(q_pids)):
        for j in range(len(g_pids)):
            if i == j:
                continue  # bỏ self-match
            if q_pids[i] == g_pids[j]:
                genuine.append(sim[i, j])
            else:
                impostor.append(sim[i, j])
    return np.array(genuine, dtype=np.float32), np.array(impostor, dtype=np.float32)


def rank_metrics(qf, gf, q_pids, g_pids):
    """
    🛠️ (14/9): Rank-1 + mAP trên CÙNG ma trận cosine với `compute_scores` (bỏ self-match i==j).

    Vì sao cần chỉ số này: TAR@FAR chỉ nói về NGƯỠNG TUYỆT ĐỐI.
      - Rank-1 CAO mà TAR@FAR thấp  -> embedding vẫn xếp hạng tốt, vấn đề là LUẬT QUYẾT ĐỊNH
        (ngưỡng cosine tuyệt đối), không phải đặc trưng -> không cần train lại.
      - Rank-1 ~ mức ngẫu nhiên (= tỉ lệ genuine) -> embedding thật sự mất khả năng phân biệt
        -> lỗi ở train / backbone / nhãn.
    Mốc ngẫu nhiên = số cặp genuine / tổng số cặp (in ra để so).

    Khác `evaluate_reid.eval_map_cmc`: hàm đó KHÔNG bỏ self-match (query/gallery là 2 list rời),
    còn ở đây bỏ để đồng nhất với `compute_scores` và các con số TAR/FAR cùng script.
    """
    qf_n = F.normalize(qf, p=2, dim=1)
    gf_n = F.normalize(gf, p=2, dim=1)
    sim = torch.mm(qf_n, gf_n.t()).cpu().numpy()
    n_q, n_g = sim.shape
    square = n_q == n_g
    if square:
        np.fill_diagonal(sim, -np.inf)  # bỏ self-match (đẩy xuống cuối)

    q_pids = np.asarray(q_pids)
    g_pids = np.asarray(g_pids)
    indices = np.argsort(-sim, axis=1)
    matches = (g_pids[indices] == q_pids[:, None]).astype(np.float32)
    if square:
        # ...và loại HẲN khỏi tập match (nếu chỉ đẩy xuống cuối thì mAP bị thổi lên vì mỗi
        # truy vấn luôn được tặng 1 cặp genuine ở hạng cuối). Đồng nhất với `compute_scores`.
        matches[indices == np.arange(n_q)[:, None]] = 0.0

    num_rel = matches.sum(axis=1)
    valid = num_rel > 0
    if not valid.any():
        return float('nan'), float('nan'), float('nan')

    m = matches[valid]
    n_rel = num_rel[valid]
    rank1 = float(m[:, 0].mean())
    tmp = np.cumsum(m, axis=1) / (np.arange(m.shape[1]) + 1.0)
    mAP = float(((tmp * m).sum(axis=1) / n_rel).mean())
    denom = (n_q * (n_g - 1)) if square else matches.size
    chance = float(matches.sum() / max(denom, 1))  # tỉ lệ genuine -> mức Rank-1 ngẫu nhiên
    return rank1, mAP, chance


# Calibration - Fixed FAR
def calibrate_fixed_far(impostor_scores, far_target):
    """
    Tìm threshold t* nhỏ nhất sao cho FAR(t*) <= far_target.
    FAR(t) = P(impostor >= t)  ->  t* = quantile(impostor, 1 - far_target)
    """
    t_star = float(np.quantile(impostor_scores, 1.0 - far_target))
    actual_far = float(np.mean(impostor_scores >= t_star))
    return t_star, actual_far


def eval_at_threshold(genuine_scores, impostor_scores, threshold):
    tar = float(np.mean(genuine_scores >= threshold))
    far = float(np.mean(impostor_scores >= threshold))
    frr = 1.0 - tar
    return tar, far, frr


# Plotting
def plot_score_distribution(genuine, impostor, threshold, far_target, out_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(
        min(genuine.min(), impostor.min()),
        max(genuine.max(), impostor.max()),
        80
    )
    ax.hist(impostor, bins=bins, alpha=0.55, color='tomato',    label='Impostor scores', density=True)
    ax.hist(genuine,  bins=bins, alpha=0.55, color='steelblue', label='Genuine scores',  density=True)
    ax.axvline(threshold, color='black', linestyle='--', linewidth=1.8,
               label=f'Threshold t*={threshold:.4f}  (FAR<={far_target*100:.2f}%)')
    ax.set_xlabel('Cosine Similarity Score', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Score Distribution -- Calibration Split', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [PLOT] Score distribution -> {out_path}")


def build_roc(genuine, impostor, n_points=500):
    """Trả về (far_arr, tar_arr) cho ROC curve."""
    all_scores = np.concatenate([genuine, impostor])
    thresholds = np.linspace(all_scores.min(), all_scores.max(), n_points)
    far_arr, tar_arr = [], []
    for t in thresholds:
        far_arr.append(np.mean(impostor >= t))
        tar_arr.append(np.mean(genuine >= t))
    return np.array(far_arr), np.array(tar_arr)


def plot_det_curve(genuine, impostor, threshold, far_target, out_path):
    """DET curve: FAR (x) vs FRR (y), log-log scale."""
    far_arr, tar_arr = build_roc(genuine, impostor)
    frr_arr = 1.0 - tar_arr

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(far_arr * 100, frr_arr * 100, color='steelblue', linewidth=2, label='DET (cal split)')

    op_far = np.mean(impostor >= threshold) * 100
    op_frr = (1.0 - np.mean(genuine >= threshold)) * 100
    ax.scatter([op_far], [op_frr], color='red', zorder=5, s=80,
               label=f'Operating point  FAR={op_far:.3f}%  FRR={op_frr:.2f}%')
    ax.axvline(far_target * 100, color='gray', linestyle=':', linewidth=1.2,
               label=f'FAR target = {far_target*100:.2f}%')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('FAR (%)', fontsize=12)
    ax.set_ylabel('FRR (%)', fontsize=12)
    ax.set_title('DET Curve -- Calibration Split', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [PLOT] DET curve -> {out_path}")


def plot_roc_comparison(gen_cal, imp_cal, gen_eval, imp_eval, threshold, far_target, out_path):
    """ROC curve của cal-split và eval-split trên cùng 1 plot."""
    far_cal, tar_cal   = build_roc(gen_cal, imp_cal)
    far_eval, tar_eval = build_roc(gen_eval, imp_eval)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(far_cal  * 100, tar_cal  * 100, color='steelblue',  linewidth=2, label='Cal split')
    ax.plot(far_eval * 100, tar_eval * 100, color='darkorange', linewidth=2,
            linestyle='--', label='Eval split (holdout)')

    op_far = np.mean(imp_eval >= threshold) * 100
    op_tar = np.mean(gen_eval >= threshold) * 100
    ax.scatter([op_far], [op_tar], color='red', zorder=5, s=80,
               label=f'Eval OP  TAR={op_tar:.2f}%  FAR={op_far:.3f}%')
    ax.axvline(far_target * 100, color='gray', linestyle=':', linewidth=1.2,
               label=f'FAR target = {far_target*100:.2f}%')

    ax.set_xlabel('FAR (%)', fontsize=12)
    ax.set_ylabel('TAR (%)', fontsize=12)
    ax.set_title('ROC Curve -- Cal vs Eval Split', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [PLOT] ROC comparison -> {out_path}")


# Sequence-level split
def split_sequences(query_json, cal_ratio, seed):
    """
    Tách danh sách sequence_id thành (cal_seqs, eval_seqs) theo tỉ lệ cal_ratio.
    Mỗi sequence nằm HOÀN TOÀN ở 1 trong 2 split -> không leak identity.
    """
    with open(query_json, 'r') as f:
        queries = json.load(f)

    seq_sample_count = defaultdict(int)
    for q in queries:
        seq_sample_count[q['sequence_id']] += 1

    all_seqs = list(seq_sample_count.keys())
    rng = random.Random(seed)
    rng.shuffle(all_seqs)

    n_cal = max(1, int(len(all_seqs) * cal_ratio))
    cal_seqs  = set(all_seqs[:n_cal])
    eval_seqs = set(all_seqs[n_cal:])

    cal_samples  = sum(seq_sample_count[s] for s in cal_seqs)
    eval_samples = sum(seq_sample_count[s] for s in eval_seqs)
    print(f"  Sequences  : {len(cal_seqs)} cal / {len(eval_seqs)} eval  (ratio={cal_ratio}, seed={seed})")
    print(f"  Samples    : {cal_samples} cal / {eval_samples} eval")

    return cal_seqs, eval_seqs


# Main
def main():
    parser = argparse.ArgumentParser(description='Threshold Calibration (Fixed FAR)')
    parser.add_argument('--config',     default='configs/config_local.yaml')
    parser.add_argument('--cal-ratio',  type=float, default=None,
                        help='Ti le sequences dung lam cal-split (0 < x < 1). Default lay tu config[calibration.cal_ratio] hoac 0.4')
    parser.add_argument('--far-target', type=float, default=None,
                        help='Muc FAR muc tieu. Default lay tu config[calibration.far_target] hoac 0.001 (0.1%%)')
    parser.add_argument('--seed',       type=int,   default=None,
                        help='Random seed. Default lay tu config[calibration.seed] hoac 42')
    parser.add_argument('--output-dir', type=str,   default=None,
                        help='Output dir. Default lay tu config[calibration.output_dir] hoac calib_results')
    parser.add_argument('--batch-size', type=int,   default=None)
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    ec           = cfg.get('eval', {})
    cc           = cfg.get('calibration', {})   # calibration section
    data_dir     = cfg.get('paths', {}).get('data_dir', './processed')
    query_json   = ec.get('query_json',   './processed/query_test.json')
    gallery_json = ec.get('gallery_json', './processed/gallery_test.json')
    model_path   = ec.get('model_path',   'checkpoints/best_model.pth')
    backbone     = ec.get('backbone',     'resnet50_ibn')
    backbone_only= ec.get('backbone_only', False)
    num_frames   = cfg.get('train', {}).get('num_frames', 16)
    batch_size   = args.batch_size or ec.get('batch_size', 32)
    num_workers  = ec.get('num_workers', 4)

    # CLI args override config; config overrides hardcoded defaults
    cal_ratio  = args.cal_ratio  if args.cal_ratio  is not None else cc.get('cal_ratio',  0.4)
    far_target = args.far_target if args.far_target is not None else cc.get('far_target', 0.001)
    seed       = args.seed       if args.seed       is not None else cc.get('seed',       42)
    output_dir = args.output_dir if args.output_dir is not None else cc.get('output_dir', 'calib_results')

    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  THRESHOLD CALIBRATION -- Fixed FAR <= {far_target*100:.2f}%")
    print(f"{'='*60}")
    print(f"  Config    : {args.config}")
    print(f"  Model     : {model_path}")
    print(f"  Cal ratio : {cal_ratio}  (seed={seed})")
    print(f"  FAR target: {far_target}")
    print(f"  Output    : {output_dir}")
    print()


    # 1. Sequence-level split
    print("[1/5] Splitting sequences...")
    cal_seqs, eval_seqs = split_sequences(query_json, cal_ratio, seed)

    # 2. Load model
    print("\n[2/5] Loading model...")
    gasnet_dir = cfg.get('paths', {}).get('gasnet_dir', '')
    if gasnet_dir:
        os.environ['GASNET_PATH'] = os.path.abspath(gasnet_dir)

    from model import UAVReIDNet, load_checkpoint_verbose
    model = UAVReIDNet(freeze_backbone=False, backbone=backbone)
    if not backbone_only and os.path.exists(model_path):
        # 🛠️ (14/9): báo cáo đầy đủ missing/unexpected/shape-mismatch (xem model.load_checkpoint_verbose)
        load_checkpoint_verbose(model, model_path, tag="calibrate")
        print(f"  Loaded weights: {model_path}")
    else:
        print(f"  WARNING: {model_path} not found -- using random weights!")
    model.cuda().eval()

    # Transform
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    test_dir = os.path.join(data_dir, 'test')

    # 3. Extract features
    # 🛠️ (14/9): trích đồng thời 2 không gian để trả lời câu hỏi
    # "BatchNorm1d trong ReIDHead có thật sự làm mất khả năng phân biệt không?"
    #   fused  = qua head (BatchNorm1d) — pipeline hiện tại
    #   pre_bn = cat(visual, temporal)  — đầu vào bnneck (raw)
    spaces = ['backbone'] if backbone_only else ['fused', 'pre_bn']
    print("\n[3/5] Extracting features...")
    print(f"  Spaces    : {', '.join(spaces)}")
    print(f"  -> Cal split ({len(cal_seqs)} sequences)...")
    t0 = time.time()
    ds_cal = CalibDataset(test_dir, query_json, gallery_json,
                          allowed_seq_ids=cal_seqs,
                          transform=transform, num_frames=num_frames)
    if len(ds_cal) == 0:
        raise RuntimeError("Cal split rong! Tang --cal-ratio hoac kiem tra query/gallery JSON.")
    dl_cal = DataLoader(ds_cal, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    feats_cal, pids_cal = extract_features(model, dl_cal, backbone_only, spaces)
    print(f"     {len(ds_cal)} pairs, {time.time()-t0:.1f}s")

    print(f"  -> Eval split ({len(eval_seqs)} sequences)...")
    t0 = time.time()
    ds_eval = CalibDataset(test_dir, query_json, gallery_json,
                           allowed_seq_ids=eval_seqs,
                           transform=transform, num_frames=num_frames)
    if len(ds_eval) == 0:
        raise RuntimeError("Eval split rong! Giam --cal-ratio.")
    dl_eval = DataLoader(ds_eval, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=True)
    feats_eval, pids_eval = extract_features(model, dl_eval, backbone_only, spaces)
    print(f"     {len(ds_eval)} pairs, {time.time()-t0:.1f}s")

    # 4. Compute scores (cho từng không gian)
    print("\n[4/5] Computing pairwise scores...")
    scores = {}
    for s in spaces:
        qf_cal, gf_cal = feats_cal[s]
        qf_eval, gf_eval = feats_eval[s]
        gen_cal, imp_cal = compute_scores(qf_cal, gf_cal, pids_cal, pids_cal)
        gen_eval, imp_eval = compute_scores(qf_eval, gf_eval, pids_eval, pids_eval)
        scores[s] = {'gen_cal': gen_cal, 'imp_cal': imp_cal,
                     'gen_eval': gen_eval, 'imp_eval': imp_eval}
        print(f"  [{s}] cal: genuine={len(gen_cal):,} impostor={len(imp_cal):,} | "
              f"eval: genuine={len(gen_eval):,} impostor={len(imp_eval):,}")

    # 5. Calibrate (từng không gian, cùng một quy trình Fixed-FAR → so sánh được)
    print("\n[5/5] Calibrating threshold (Fixed FAR)...")
    results = {}
    for s in spaces:
        sc = scores[s]
        if len(sc['imp_cal']) == 0 or len(sc['gen_cal']) == 0:
            print(f"  [{s}] SKIP: cal-split thiếu genuine/impostor pairs. Tang --cal-ratio.")
            continue
        t_star, cal_actual_far = calibrate_fixed_far(sc['imp_cal'], far_target)
        cal_tar, _, cal_frr = eval_at_threshold(sc['gen_cal'], sc['imp_cal'], t_star)
        eval_tar, eval_actual_far, eval_frr = eval_at_threshold(sc['gen_eval'], sc['imp_eval'], t_star)
        # 🛠️ (14/9): Rank-1/mAP để tách "ngưỡng tuyệt đối tệ" khỏi "embedding mất khả năng phân biệt"
        qf_cal, gf_cal = feats_cal[s]
        qf_eval, gf_eval = feats_eval[s]
        rank1_cal, map_cal, chance_cal = rank_metrics(qf_cal, gf_cal, pids_cal, pids_cal)
        rank1_eval, map_eval, chance_eval = rank_metrics(qf_eval, gf_eval, pids_eval, pids_eval)
        results[s] = {
            'threshold': float(t_star),
            'cal_actual_far': float(cal_actual_far), 'cal_tar': float(cal_tar), 'cal_frr': float(cal_frr),
            'eval_tar': float(eval_tar), 'eval_far': float(eval_actual_far), 'eval_frr': float(eval_frr),
            'n_genuine_cal': int(len(sc['gen_cal'])), 'n_impostor_cal': int(len(sc['imp_cal'])),
            'n_genuine_eval': int(len(sc['gen_eval'])), 'n_impostor_eval': int(len(sc['imp_eval'])),
            'rank1_cal': rank1_cal, 'mAP_cal': map_cal,
            'rank1_eval': rank1_eval, 'mAP_eval': map_eval, 'rank1_chance': chance_eval,
        }

    if not results:
        raise RuntimeError("Khong calibrate duoc khong gian nao (thieu impostor pairs o cal-split).")

    primary = spaces[0] if spaces[0] in results else next(iter(results))

    # 5b. Bảng so sánh không gian — đây là kết luận chính của lần chạy
    print(f"\n  === SO SANH KHONG GIAN DAC TRUNG (Fixed FAR <= {far_target*100:.2f}%) ===")
    print(f"  {'Space':<10} {'t*':>10} {'TAR@t* (cal)':>13} {'TAR@t* (eval)':>14} {'FAR@t* (eval)':>14} "
          f"{'Rank-1 (eval)':>14} {'mAP':>8}")
    for s, r in results.items():
        print(f"  {s:<10} {r['threshold']:>10.6f} {r['cal_tar']*100:>12.2f}% "
              f"{r['eval_tar']*100:>13.2f}% {r['eval_far']*100:>13.4f}% "
              f"{r['rank1_eval']*100:>13.2f}% {r['mAP_eval']*100:>7.2f}%")

    # 🛠️ (14/9): tách "ngưỡng tuyệt đối tệ" khỏi "embedding mất khả năng phân biệt"
    chance = next(iter(results.values())).get('rank1_chance', float('nan'))
    print(f"\n  Moc ngau nhien cua Rank-1 (ti le genuine) = {chance*100:.2f}%")
    for s, r in results.items():
        r1 = r['rank1_eval']
        if np.isnan(r1):
            continue
        if r1 <= chance * 2.5:
            tag = "=> Rank-1 gan muc ngau nhien: EMBEDDING mat kha nang phan biet (loi train/backbone/nhan)"
        elif r1 >= 0.50 and r['eval_tar'] < 0.30:
            tag = "=> Rank-1 tot nhung TAR@FAR thap: XEP HANG van dung -> van de la LUAT QUYET DINH (nguong cosine tuyet doi), KHONG can train lai"
        elif r1 >= chance * 2.5:
            tag = "=> Rank-1 tren muc ngau nhien nhung con thap: da co tin hieu, can xem lai train + luat quyet dinh"
        print(f"    [{s}] Rank-1={r1*100:.2f}% (mAP={r['mAP_eval']*100:.2f}%)  {tag}")
    if 'fused' in results and 'pre_bn' in results:
        d_tar = results['pre_bn']['eval_tar'] - results['fused']['eval_tar']
        verdict = ("pre_bn TOT HON ro ret -> BatchNorm1d that su lam mat kha nang phan biet"
                   if d_tar > 0.02 else
                   "pre_bn KEM HON -> BN chi nen thang diem, KHONG lam mat thong tin"
                   if d_tar < -0.02 else
                   "pre_bn ~= fused -> BN vo hai ve mat phan biet")
        print(f"\n  ΔTAR(eval, pre_bn - fused) = {d_tar*100:+.2f}%  -> {verdict}")
        # ⚠️ So sánh TƯƠNG ĐỐI chỉ nói BN có phải thủ phạm hay không; còn phải xét MỨC TUYỆT ĐỐI.
        best_tar = max(results[s]['eval_tar'] for s in ('fused', 'pre_bn'))
        if best_tar < 0.30:
            print(f"  ⚠️ CA HAI khong gian deu TAR rat thap (tot nhat {best_tar*100:.2f}% @FAR<={far_target*100:.2f}%)")
            print(f"     -> Nut that KHONG phai BatchNorm1d. Xem Rank-1 o tren:")
            print(f"        Rank-1 tot  -> doi LUAT QUYET DINH (dung xep hang/margin thay vi nguong cosine tuyet doi)")
            print(f"        Rank-1 thap -> EMBEDDING mat kha nang phan biet (train / backbone / nhan)")

    # 🛠️ (14/9): TAR tại NHIỀU mốc FAR — threshold calibrate trên CAL, đo TAR trên EVAL (out-of-sample).
    # Vì sao cần: TAR@FAR=0.1% chỉ là MỘT điểm làm việc. Nếu TAR tăng vọt ở FAR 1–5% thì vấn đề nằm ở
    # việc CHỌN ĐIỂM LÀM VIỆC (luật quyết định), không phải ở chất lượng embedding.
    far_grid = [0.001, 0.01, 0.05, 0.10]
    header = " ".join(f"{('FAR ' + str(f*100).rstrip('0').rstrip('.') + '%'):>11}" for f in far_grid)
    print(f"\n  === TAR (eval-split) theo cac moc FAR — threshold calibrate tren cal-split ===")
    print(f"  {'Space':<10} {header}")
    far_curve = {}
    for s in results:
        row, vals = [], []
        for f in far_grid:
            t_f, _ = calibrate_fixed_far(scores[s]['imp_cal'], f)
            tar_f, far_act, _ = eval_at_threshold(scores[s]['gen_eval'], scores[s]['imp_eval'], t_f)
            row.append(f"{tar_f*100:>10.1f}%")
            vals.append({'far_target': f, 'threshold': float(t_f),
                         'tar_eval': float(tar_f), 'far_eval': float(far_act)})
        print(f"  {s:<10} " + " ".join(row))
        far_curve[s] = vals
    print(f"  -> Doc bang nay: neu TAR tang VOT khi noi FAR thi diem lam viec 0.1% moi la van de,")
    print(f"     khong phai embedding. Chon nguong theo nhu cau that (latency vs lock sai), khong theo FAR 0.1%.")

    t_star = results[primary]['threshold']
    cal_actual_far = results[primary]['cal_actual_far']
    cal_tar = results[primary]['cal_tar']
    cal_frr = results[primary]['cal_frr']
    eval_tar = results[primary]['eval_tar']
    eval_actual_far = results[primary]['eval_far']
    eval_frr = results[primary]['eval_frr']
    gen_cal, imp_cal = scores[primary]['gen_cal'], scores[primary]['imp_cal']
    gen_eval, imp_eval = scores[primary]['gen_eval'], scores[primary]['imp_eval']

    print(f"\n  +-- Calibration Result (cal-split, space={primary}) ---+")
    print(f"  | Threshold t*     : {t_star:.6f}                  |")
    print(f"  | FAR target       : {far_target*100:.4f}%                    |")
    print(f"  | Actual FAR (cal) : {cal_actual_far*100:.4f}%                    |")
    print(f"  | TAR @ t* (cal)   : {cal_tar*100:.2f}%                     |")
    print(f"  | FRR @ t* (cal)   : {cal_frr*100:.2f}%                     |")
    print(f"  +-------------------------------------------------+")

    print(f"\n  +-- Holdout Eval Result (eval-split, space={primary}) --+")
    print(f"  | Threshold t*     : {t_star:.6f}  (from cal)      |")
    print(f"  | TAR @ t* (eval)  : {eval_tar*100:.2f}%                     |")
    print(f"  | FAR @ t* (eval)  : {eval_actual_far*100:.4f}%                    |")
    print(f"  | FRR @ t* (eval)  : {eval_frr*100:.2f}%                     |")
    print(f"  +-------------------------------------------------+")

    # Save threshold
    threshold_out = os.path.join(output_dir, 'calibrated_threshold.json')
    threshold_data = {
        # 'threshold' giữ = không gian chính (fused) để tương thích ngược với evaluate_reid.py
        'threshold': float(t_star),
        'space': primary,
        'thresholds': {s: float(r['threshold']) for s, r in results.items()},
        'far_target': float(far_target),
        'cal_ratio': float(cal_ratio),
        'seed': seed,
        'calibrated_on': 'test_cal_split',
        'model_path': model_path,
        'backbone': backbone,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(threshold_out, 'w') as f:
        json.dump(threshold_data, f, indent=4)

    # Full report
    report = {
        'split_config': {
            'cal_ratio': cal_ratio,
            'seed': seed,
            'n_cal_sequences': len(cal_seqs),
            'n_eval_sequences': len(eval_seqs),
            'n_cal_samples': len(ds_cal),
            'n_eval_samples': len(ds_eval),
        },
        'primary_space': primary,
        'spaces': results,
        # 🛠️ (14/9): TAR tại nhiều mốc FAR (threshold calibrate trên cal, đo trên eval)
        'far_curve': far_curve,
        # backward compat: giữ nguyên hình dạng cũ cho không gian chính
        'calibration': {
            'threshold': float(t_star),
            'far_target': float(far_target),
            'actual_far_cal': float(cal_actual_far),
            'tar_cal': float(cal_tar),
            'frr_cal': float(cal_frr),
            'n_genuine_cal': int(len(gen_cal)),
            'n_impostor_cal': int(len(imp_cal)),
        },
        'holdout_eval': {
            'tar': float(eval_tar),
            'far': float(eval_actual_far),
            'frr': float(eval_frr),
            'n_genuine_eval': int(len(gen_eval)),
            'n_impostor_eval': int(len(imp_eval)),
            'threshold_source': 'calibrated_fixed_far',
        },
    }
    report_out = os.path.join(output_dir, 'calibration_report.json')
    with open(report_out, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"\n  Saved: {threshold_out}")
    print(f"  Saved: {report_out}")

    # Plots (mỗi không gian 1 bộ; tên file có hậu tố space khi >1 không gian)
    if HAS_MPL:
        print("\n  Generating plots...")
        for s in results:
            sc = scores[s]
            suffix = f"_{s}" if len(results) > 1 else ""
            plot_score_distribution(sc['gen_cal'], sc['imp_cal'], results[s]['threshold'], far_target,
                                    os.path.join(output_dir, f'score_distribution{suffix}.png'))
            plot_det_curve(sc['gen_cal'], sc['imp_cal'], results[s]['threshold'], far_target,
                           os.path.join(output_dir, f'det_curve{suffix}.png'))
            plot_roc_comparison(sc['gen_cal'], sc['imp_cal'], sc['gen_eval'], sc['imp_eval'],
                                results[s]['threshold'], far_target,
                                os.path.join(output_dir, f'roc_curve_comparison{suffix}.png'))

    print(f"\n{'='*60}")
    print(f"  DONE. Dung threshold trong evaluate_reid.py:")
    print(f"    --threshold-file {threshold_out}                 (space={primary})")
    if 'pre_bn' in results:
        print(f"    --threshold-file {threshold_out} --space pre_bn  (so sanh)")
    print(f"{'='*60}\n")



if __name__ == '__main__':
    main()
