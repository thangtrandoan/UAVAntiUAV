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
                    # 🛠️ (14/9) GIỮ LẠI metadata định danh để chẩn đoán nhiễu nhãn/trùng danh tính
                    # (`diagnose_top_impostors`). KHÔNG đổi `__getitem__`/dataloader: DataLoader dùng
                    # `shuffle=False` nên hàng feature khớp 1:1 với `valid_pairs` -> truy cập trực tiếp.
                    self.valid_pairs.append({
                        'identity_id': q['identity_id'],
                        'sequence_id': q['sequence_id'],
                        'event_index': q.get('event_index'),
                        'gallery_frames': g['frames'],
                        'gallery_dir': g['frame_dir'],
                        'query_frames': q['frames'],
                        'query_dir': q['frame_dir'],
                    })

    def __len__(self):
        return len(self.valid_pairs)

    def _load_clip(self, folder, frames, take_last=False):
        # 🛠️ (15/9) ĐỒNG BỘ frame_stride — BỎ `np.linspace` (xem train_reid.py::_load_clip).
        # `np.linspace` làm bước thời gian hiệu dụng > frame_stride → calibration lệch
        # với phân phối temporal lúc train và lúc infer.
        if len(frames) > self.num_frames:
            frames = frames[-self.num_frames:] if take_last else frames[:self.num_frames]
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
        before_clip = self._load_clip(pair['gallery_dir'], list(pair['gallery_frames']), take_last=True)
        after_clip  = self._load_clip(pair['query_dir'],   list(pair['query_frames']),   take_last=False)
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
      'visual'   : CHỈ nhánh visual (2560-d) — ablation nhánh temporal
      'temporal' : CHỈ token temporal (512-d) — ablation nhánh visual

    🛠️ (14/9) Thêm 'visual' và 'temporal' để trả lời "nhánh temporal có thực sự đóng góp không?".
    ⚠️ PHƯƠNG PHÁP: **không được so MỨC điểm** giữa các nhánh. `temporal ≈ 0.938 < visual ≈ 0.990`
    KHÔNG có nghĩa temporal kém hơn — cosine ở các không gian/số chiều khác nhau không so sánh
    được (đúng loại lỗi đã mắc với `pre_bn` vs `fused`). Phải so **khả năng phân biệt**:
    Rank-1 / mAP / TAR@FAR.

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
                if 'visual' in spaces:
                    acc['visual']['gf'].append(v_g.cpu())
                    acc['visual']['qf'].append(v_q.cpu())
                if 'temporal' in spaces:
                    acc['temporal']['gf'].append(t_g.cpu())
                    acc['temporal']['qf'].append(t_q.cpu())
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
def similarity_matrix(qf, gf):
    """Cosine similarity matrix (num_q, num_g) — L2-normalize rồi matmul."""
    qf_n = F.normalize(qf, p=2, dim=1)
    gf_n = F.normalize(gf, p=2, dim=1)
    return torch.mm(qf_n, gf_n.t()).cpu().numpy()


def scores_from_matrix(sim, q_pids, g_pids):
    """
    Tách genuine/impostor từ ma trận similarity. Vector hoá (nhanh hơn vòng lặp Python).

    Ngữ nghĩa GIỐNG HỆT `compute_scores` bản cũ: bỏ self-match i==j khỏi **cả hai** tập.
    """
    q_pids = np.asarray(q_pids)
    g_pids = np.asarray(g_pids)
    same = (g_pids[None, :] == q_pids[:, None])
    valid = np.ones_like(same, dtype=bool)
    if sim.shape[0] == sim.shape[1]:
        np.fill_diagonal(valid, False)  # bỏ self-match
    genuine = sim[same & valid]
    impostor = sim[(~same) & valid]
    return genuine.astype(np.float32), impostor.astype(np.float32)


def compute_scores(qf, gf, q_pids, g_pids):
    """Trả về genuine_scores, impostor_scores (numpy arrays) từ pairwise cosine sim."""
    return scores_from_matrix(similarity_matrix(qf, gf), q_pids, g_pids)


def diagnose_top_impostors(qf, gf, pids, meta, space='fused', top_k=50,
                           threshold=None, data_dir=None, top_pid_pairs=20):
    """
    🛠️ (14/9) CHẨN ĐOÁN "NHIỄU NHÃN / TRÙNG DANH TÍNH".

    Vì sao cần: TAR@FAR thấp bất thường (8.45% @FAR 0.1%) có thể do **một số ít cặp "impostor"
    thực chất là CÙNG MỘT VẬT** — cùng drone mang 2 `identity_id`, hoặc frame gần trùng. Chúng đẩy
    phân vị 99.9% của impostor lên rất cao → mọi ngưỡng tuyệt đối bị vô hiệu. **z-norm KHÔNG THỂ
    sửa loại lỗi này** (§11.2/§11.3), nên phải kiểm tra trực tiếp.

    Ba đầu ra:
      (A) `top_pairs`      : top-K cặp impostor điểm cao nhất, kèm `identity_id`/`sequence_id`/
                             khoảng cách frame/**đường dẫn ảnh** để mở ra xem tận mắt.
      (B) `top_pid_pairs`  : các CẶP DANH TÍNH dễ nhầm nhất (theo sim trung bình) — nếu một cặp
                             danh tính có sim TB ~0.9 thì gần như chắc chắn là cùng một vật.
      (C) `concentration`  : khối lượng impostor vượt ngưỡng tập trung vào bao nhiêu cặp danh tính?
                             Tập trung cao = dấu hiệu mạnh của nhiễu nhãn.

    DIỄN GIẢI (quan trọng, tránh kết luận sai):
      • `same_seq=True`  → distractor KHÁC danh tính trong CÙNG chuỗi = **hard negative hợp lệ**
        (đúng loại pipeline gặp), KHÔNG phải nhiễu nhãn.
      • `same_seq=False` + ảnh giống hệt → **nghi nhiễu nhãn / trùng danh tính** (phải xem ảnh).
      • Cặp danh tính có sim TB rất cao mà khác `identity_id` → nhiễu nhãn gần như chắc chắn.

    Trả về dict (JSON-serializable) để lưu vào report.
    """
    sim = similarity_matrix(qf, gf)
    q = np.asarray(pids)
    n_q, n_g = sim.shape
    imp_mask = ~(q[:, None] == q[None, :])
    if n_q == n_g:
        np.fill_diagonal(imp_mask, False)          # bỏ self-match

    n_imp = int(imp_mask.sum())
    out = {'space': space, 'n_impostor_pairs': n_imp, 'top_k': int(top_k),
           'threshold': None if threshold is None else float(threshold)}

    # ---------- (A) top-K cặp impostor điểm cao nhất ----------
    if n_imp == 0:
        out['top_pairs'] = []
        return out
    k = min(int(top_k), n_imp)
    flat = np.where(imp_mask, sim, -np.inf).ravel()
    idx = np.argpartition(-flat, k - 1)[:k]
    idx = idx[np.argsort(-flat[idx])]
    ii, jj = np.unravel_index(idx, sim.shape)

    def _paths(entry, which):
        d = entry.get(f'{which}_dir')
        fr = entry.get(f'{which}_frames') or []
        if data_dir is not None and d is not None and fr:
            return [os.path.join(data_dir, d, f) for f in list(fr)[:3]]
        return [str(f) for f in list(fr)[:3]]

    def _n_frames(entry, which):
        return len(entry.get(f'{which}_frames') or [])

    top_pairs = []
    for a, b in zip(ii, jj):
        me, mg = meta[int(a)], meta[int(b)]
        top_pairs.append({
            'score': float(sim[a, b]),
            'q_identity_id': me['identity_id'],
            'g_identity_id': mg['identity_id'],
            'q_sequence_id': me.get('sequence_id'),
            'g_sequence_id': mg.get('sequence_id'),
            'same_sequence': bool(me.get('sequence_id') == mg.get('sequence_id')),
            'q_event_index': me.get('event_index'),
            'g_event_index': mg.get('event_index'),
            'q_n_frames': _n_frames(me, 'query'),
            'g_n_frames': _n_frames(mg, 'gallery'),
            'q_image_paths': _paths(me, 'query'),
            'g_image_paths': _paths(mg, 'gallery'),
            'above_threshold': None if threshold is None else bool(sim[a, b] >= threshold),
        })
    out['top_pairs'] = top_pairs

    # ---------- (B) các CẶP DANH TÍNH dễ nhầm nhất (theo sim trung bình) ----------
    pid_pairs = {}
    for a in range(n_q):
        row = imp_mask[a]
        if not row.any():
            continue
        cols = np.nonzero(row)[0]
        for b in cols:
            # gộp theo cặp KHÔNG THỨ TỰ (n_pairs = tổng số cặp giữa 2 danh tính)
            key = (min(int(q[a]), int(q[b])), max(int(q[a]), int(q[b])))
            v = pid_pairs.get(key)
            if v is None:
                pid_pairs[key] = [1, float(sim[a, b]), float(sim[a, b])]   # n, sum, max
            else:
                v[0] += 1; v[1] += float(sim[a, b])
                v[2] = max(v[2], float(sim[a, b]))
    ranked = sorted(pid_pairs.items(), key=lambda kv: -(kv[1][1] / kv[1][0]))[:int(top_pid_pairs)]
    out['top_pid_pairs'] = [{
        'pid_a': ka, 'pid_b': kb, 'n_pairs': v[0],
        'mean_sim': v[1] / v[0], 'max_sim': v[2],
    } for (ka, kb), v in ranked]

    # ---------- (C) độ TẬP TRUNG của khối lượng impostor vượt ngưỡng ----------
    if threshold is not None:
        over = imp_mask & (sim >= threshold)
        n_over = int(over.sum())
        out['n_impostor_above_threshold'] = n_over
        out['far_above_threshold'] = n_over / max(n_imp, 1)
        # đếm theo cặp danh tính
        cnt = {}
        ii2, jj2 = np.nonzero(over)
        for a, b in zip(ii2, jj2):
            # gộp theo cặp KHÔNG THỨ TỰ (n_pairs = tổng số cặp giữa 2 danh tính)
            key = (min(int(q[a]), int(q[b])), max(int(q[a]), int(q[b])))
            cnt[key] = cnt.get(key, 0) + 1
        # theo chuỗi: cùng chuỗi (hard negative hợp lệ) vs khác chuỗi (nghi nhiễu nhãn)
        same_seq_over = sum(1 for a, b in zip(ii2, jj2)
                            if meta[int(a)].get('sequence_id') == meta[int(b)].get('sequence_id'))
        out['concentration'] = {
            'n_distinct_pid_pairs_total': len(pid_pairs),
            'n_distinct_pid_pairs_above': len(cnt),
            'top_pid_pairs_share': [
                {'pid_a': kk[0], 'pid_b': kk[1], 'n': nn, 'share': nn / max(n_over, 1)}
                for kk, nn in sorted(cnt.items(), key=lambda kv: -kv[1])[:10]
            ],
            'frac_above_same_sequence': (same_seq_over / n_over) if n_over else None,
            'frac_above_cross_sequence': ((n_over - same_seq_over) / n_over) if n_over else None,
        }
    return out


def per_query_znorm(sim, q_pids=None, g_pids=None, cohort='all'):
    """
    🛠️ (14/9): chuẩn hoá điểm theo TỪNG TRUY VẤN (z-norm / cohort normalization).

        z[i, j] = (s[i, j] - mu_i) / sd_i

    cohort='all'      : mu_i/sd_i trên TOÀN BỘ gallery của truy vấn i. Không dùng nhãn.
    cohort='impostor' : mu_i/sd_i CHỈ trên các cột KHÁC danh tính với truy vấn i.

    ⚠️ Vì sao phải có cả 2: gallery có ~21 cặp genuine mỗi truy vấn (2.9%), và chúng chính là
    các điểm CAO NHẤT → đưa chúng vào mu/sd sẽ làm mu, sd phồng lên và **đè z-score của chính
    genuine xuống**. Vì vậy `cohort='all'` là một phép thử THIÊN VỊ (thiên vị chống z-norm).
    `cohort='impostor'` là UPPER BOUND đúng của hướng này, nhưng nó DÙNG NHÃN để chọn cohort →
    deployment thật phải thay bằng một cohort tham chiếu CÓ NHÃN (t-norm), không phải toàn gallery.

    Đây là biến đổi affine theo từng hàng → **Rank-1 không đổi**, chỉ DET/TAR đổi.
    """
    sim = np.asarray(sim, dtype=np.float64)
    if cohort == 'impostor' and q_pids is not None and g_pids is not None:
        q = np.asarray(q_pids)
        g = np.asarray(g_pids)
        mask = (g[None, :] != q[:, None]).astype(np.float64)
        if sim.shape[0] == sim.shape[1]:
            np.fill_diagonal(mask, 0.0)          # bỏ self-match
        n = np.maximum(mask.sum(axis=1, keepdims=True), 1.0)
        mu = (sim * mask).sum(axis=1, keepdims=True) / n
        var = (((sim - mu) ** 2) * mask).sum(axis=1, keepdims=True) / n
    else:
        mu = sim.mean(axis=1, keepdims=True)
        var = sim.var(axis=1, keepdims=True)
    return (sim - mu) / (np.sqrt(np.maximum(var, 0.0)) + 1e-12)


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
    # 🛠️ (24/9) `temporal_pool` / `temporal_pe` CŨNG phải khớp lúc TRAIN, không chỉ `temporal_type`.
    # Trước đây 2 key này không được truyền -> rơi về default của `UAVReIDNet` (`pool='attn'`).
    # Với checkpoint train bằng `pool='mean'`, `attn_pool` KHÔNG có trong file nên giữ zero-init
    # -> `attn_pool(x) = mean(x)` **KHÔNG nhân `sqrt(N)`**, trong khi lúc train là `mean(x)*sqrt(N)`
    # => token temporal ở eval LỆCH so với train (N=12: lệch hệ số 3.464, và vì `out_mlp` có bias
    #    nên BatchNorm1d KHÔNG bù được). Biểu hiện: space `temporal` tụt mạnh nhất.
    temporal_type = cfg.get('train', {}).get('temporal_type', 'mamba')
    temporal_pool = cfg.get('train', {}).get('temporal_pool', 'attn')
    temporal_pe = bool(cfg.get('train', {}).get('temporal_pe', True))
    print(f"  Temporal encoder: type={temporal_type}, pool={temporal_pool}, pe={temporal_pe}")
    model = UAVReIDNet(freeze_backbone=False, backbone=backbone, temporal_type=temporal_type,
                       temporal_pool=temporal_pool, temporal_pe=temporal_pe)
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
    spaces = ['backbone'] if backbone_only else ['fused', 'pre_bn', 'visual', 'temporal']
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
            # ⚠️ KHONG ket luan ngay "luat quyet dinh": 2 gia thuyet cung giai thich duoc hien tuong:
            #   (a) diem cosine khong so sanh duoc giua cac truy van -> SUA LUAT QUYET DINH
            #   (b) diem cosine da tuong thich, nhung genuine/impostor CHONG LAN that su
            #       -> van de o embedding/du lieu, phai train lai
            # Khoi z-norm (chay SAU) moi phan biet duoc (a) vs (b).
            tag = ("=> Rank-1 tot nhung TAR@FAR thap: HAI gia thuyet — (a) luat quyet dinh "
                   "vs (b) embedding/du lieu. Doc khoi Z-NORM ben duoi de phan biet")
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
        # 🛠️ (14/9) FIX: bản cũ hardcode "CẢ HAI" + "8.45%" và kết luận "đổi LUẬT QUYẾT ĐỊNH" —
        # SAI, vì khối z-norm ở dưới đã bác bỏ giả thuyết đó (tốt nhất +0.22%). Giờ không kết luận
        # thay, chỉ nêu 2 nhánh và để khối z-norm + ablation quyết định.
        best_space = max(results, key=lambda k: results[k]['eval_tar'])
        best_tar = results[best_space]['eval_tar']
        if best_tar < 0.30:
            n_sp = len(results)
            print(f"  ⚠️ TAR@FAR<={far_target*100:.2f}% rat thap o MOI khong gian "
                  f"({n_sp} khong gian, tot nhat {best_space}={best_tar*100:.2f}%)")
            print(f"     -> Nut that KHONG phai BatchNorm1d. Hai gia thuyet, KHONG ket luan thay:")
            print(f"        (a) LUAT QUYET DINH: diem cosine khong so sanh duoc giua cac truy van")
            print(f"            -> PHEP THU: khoi Z-NORM ben duoi (cohort=impostor la upper bound)")
            print(f"        (b) EMBEDDING/DU LIEU: genuine va impostor CHONG LAN that su")
            print(f"            -> PHEP THU: khoi ABLATION + kiem tra nhiem nhan/trung danh tinh")
            print(f"        Luu y: Rank-1 cao KHONG tu no phan biet duoc (a) va (b).")

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

    # === THU NGHIEM: CHUAN HOA DIEM THEO TUNG TRUY VAN (z-norm) =========================
    # Rank-1 cao + TAR@FAR thap co the do MUC diem khong so sanh duoc giua cac truy van.
    # Do ca 2 bien the de tranh ket luan sai:
    #   cohort='all'      : mu/sd tren toan bo gallery  -> THIEN VI (gallery chua ~21 genuine
    #                       diem CAO moi truy van, lam mu/sd phong len va de z_genuine xuong)
    #   cohort='impostor' : mu/sd chi tren cot KHAC danh tinh -> UPPER BOUND dung cua huong nay
    #                       (dung nhan -> deployment phai dung cohort tham chieu co nhan)
    # Rank-1 bat bien o ca 2 (bien doi affine theo hang), chi DET/TAR doi.
    print(f"\n  === THU NGHIEM: chuan hoa diem theo TUNG TRUY VAN (z-norm) ===")
    print(f"  {'Space':<10} {'cohort':<9} {'t* (z, cal)':>12} {'TAR@FAR=0.1%':>13} {'so voi goc':>11} "
          f"{'FAR 1%':>8} {'FAR 5%':>8} {'FAR 10%':>8}")
    znorm = {}
    for s in list(results.keys()):
        sim_cal = similarity_matrix(feats_cal[s][0], feats_cal[s][1])
        sim_eval = similarity_matrix(feats_eval[s][0], feats_eval[s][1])
        znorm[s] = {}
        for cohort in ('all', 'impostor'):
            gz_cal, iz_cal = scores_from_matrix(
                per_query_znorm(sim_cal, pids_cal, pids_cal, cohort=cohort), pids_cal, pids_cal)
            gz_eval, iz_eval = scores_from_matrix(
                per_query_znorm(sim_eval, pids_eval, pids_eval, cohort=cohort), pids_eval, pids_eval)

            t_z, far_z_cal = calibrate_fixed_far(iz_cal, far_target)
            tar_z, far_z, _ = eval_at_threshold(gz_eval, iz_eval, t_z)
            grid = []
            for f in far_grid:
                t_f, _ = calibrate_fixed_far(iz_cal, f)
                tar_f, far_a, _ = eval_at_threshold(gz_eval, iz_eval, t_f)
                grid.append({'far_target': f, 'threshold': float(t_f),
                             'tar_eval': float(tar_f), 'far_eval': float(far_a)})
            znorm[s][cohort] = {
                'threshold': float(t_z), 'far_target': float(far_target),
                'actual_far_cal': float(far_z_cal), 'tar_eval': float(tar_z), 'far_eval': float(far_z),
                'tar_raw_eval': float(results[s]['eval_tar']),
                'gain_vs_raw': float(tar_z - results[s]['eval_tar']),
                'far_curve': grid,
            }
            g1 = next((x['tar_eval'] for x in grid if abs(x['far_target'] - 0.01) < 1e-9), float('nan'))
            g5 = next((x['tar_eval'] for x in grid if abs(x['far_target'] - 0.05) < 1e-9), float('nan'))
            g10 = next((x['tar_eval'] for x in grid if abs(x['far_target'] - 0.10) < 1e-9), float('nan'))
            print(f"  {s:<10} {cohort:<9} {t_z:>12.4f} {tar_z*100:>12.2f}% "
                  f"{(tar_z - results[s]['eval_tar'])*100:>+10.2f}% "
                  f"{g1*100:>7.1f}% {g5*100:>7.1f}% {g10*100:>7.1f}%")
    best_gain = max((v['gain_vs_raw'] for d in znorm.values() for v in d.values()), default=0.0)
    if best_gain > 0.05:
        print(f"  => z-norm GIUP TAR tang {best_gain*100:+.2f}% o cung FAR -> diem cosine KHONG tuong thich")
        print(f"     giua cac truy van; huong dung la chuan hoa diem (cohort tham chieu co nhan).")
    else:
        print(f"  => z-norm KHONG giup dang ke (tot nhat {best_gain*100:+.2f}%) -> diem cosine da tuong")
        print(f"     thich giua cac truy van. Nut that nam o CHO KHAC (xem khoi TAI NGUONG TRIEN KHAI).")

    # === ABLATION: nhanh TEMPORAL co dong gop gi khong? ==================================
    # Tra loi cau hoi "diem temporal thap hon visual => temporal co y nghia khong?".
    # ⚠️ KHONG so muc diem (vo nghia giua cac khong gian) — so KHẢ NĂNG PHÂN BIỆT.
    if 'visual' in results and 'temporal' in results:
        print(f"\n  === ABLATION: nhanh TEMPORAL dong gop bao nhieu? (eval-split) ===")
        print(f"  {'Space':<10} {'dim':>6} {'Rank-1 (eval)':>14} {'mAP':>8} {'TAR@FAR0.1%':>12} {'TAR@FAR1%':>10}")
        # 🛠️ (24/9) TRƯỚC ĐÂY là hằng số 2560/3072 — SAI với `dinov3_convnext` (thật ra
        # visual=960, temporal=512, fused=pre_bn=1472). Nay suy trực tiếp từ model.
        _in_dim = int(model.head.bnneck.num_features)
        _vis_dim = 960 if backbone == "dinov3_convnext" else 2560
        dims = {'visual': _vis_dim, 'temporal': _in_dim - _vis_dim,
                'pre_bn': _in_dim, 'fused': _in_dim}
        for s in ('visual', 'temporal', 'pre_bn', 'fused'):
            if s not in results:
                continue
            r = results[s]
            g1 = next((x['tar_eval'] for x in far_curve.get(s, [])
                       if abs(x['far_target'] - 0.01) < 1e-9), float('nan'))
            print(f"  {s:<10} {dims.get(s, 0):>6} {r['rank1_eval']*100:>13.2f}% "
                  f"{r['mAP_eval']*100:>7.2f}% {r['eval_tar']*100:>11.2f}% {g1*100:>9.2f}%")
        ablation = {}
        for s in ('visual', 'temporal', 'pre_bn', 'fused'):
            if s in results:
                ablation[s] = {'rank1_eval': float(results[s]['rank1_eval']),
                               'mAP_eval': float(results[s]['mAP_eval']),
                               'tar_at_far01_eval': float(results[s]['eval_tar'])}
        if 'visual' in results:
            d_rank1 = results['fused']['rank1_eval'] - results['visual']['rank1_eval']
            d_map = results['fused']['mAP_eval'] - results['visual']['mAP_eval']
            d_tar = results['fused']['eval_tar'] - results['visual']['eval_tar']
            print(f"\n  Δ(fused − visual) :  Rank-1 {d_rank1*100:+.2f}%   mAP {d_map*100:+.2f}%   "
                  f"TAR@FAR0.1% {d_tar*100:+.2f}%")
            same_rank = (abs(d_rank1) < 0.02 and abs(d_map) < 0.02)
            same_op = abs(d_tar) < 0.01
            # 🛠️ (14/9): điểm quan trọng — temporal có thể giúp XẾP HẠNG mà KHÔNG giúp ĐIỂM LÀM VIỆC
            # (hoặc ngược lại). Phải tách 2 tiêu chí, không gộp thành một kết luận.
            if same_rank and same_op:
                print(f"  => fused ≈ visual o MOI chi so -> nhanh TEMPORAL KHONG dong gop gi.")
                print(f"     Hanh dong: bo temporal (visual-only) de giam chi phi.")
            elif not same_rank and not same_op and d_map > 0 and d_tar < 0:
                print(f"  => TACH BIET 2 TIEU CHI (ket qua quan trong nhat cua ablation nay):")
                print(f"     • XEP HANG  : fused TOT HON visual (mAP {d_map*100:+.2f}%, Rank-1 {d_rank1*100:+.2f}%)")
                print(f"                   -> temporal CO mang them thong tin danh tinh.")
                print(f"     • DIEM/FAR  : fused KEM HON visual (TAR@FAR0.1% {d_tar*100:+.2f}%)")
                print(f"                   -> fusion lam NOI RONG DUOI impostor nhieu hon genuine.")
                print(f"     Hanh dong: dung VISUAL cho cong quyet dinh theo nguong tuyet doi, va/hoac")
                print(f"     FUSE O MUC DIEM (2 cong doc lap / hoc trong so) thay vi concat roi BN.")
            elif d_map > 0.02 or d_rank1 > 0.02:
                print(f"  => fused TOT HON visual -> nhanh TEMPORAL CO dong gop (dù muc diem thap hon).")
            else:
                print(f"  => fused KEM hon visual o ca 2 tieu chi -> nhanh temporal dang GAY HAI.")
            # So rieng temporal-only voi moc ngau nhien
            if 'temporal' in results:
                r1t = results['temporal']['rank1_eval']
                mt = results['temporal']['mAP_eval']
                print(f"  temporal-only: Rank-1={r1t*100:.2f}%  mAP={mt*100:.2f}%  "
                      f"(moc ngau nhien {chance*100:.2f}%) -> "
                      f"{'CO thong tin danh tinh' if r1t > 3*chance else 'GAN NHU KHONG co thong tin danh tinh'}")
                print(f"  ⇒ Tra loi cau hoi 'diem temporal thap hon visual thi temporal co y nghia khong?':")
                print(f"     CO — temporal-only dat Rank-1={r1t*100:.2f}% ≫ moc ngau nhien {chance*100:.2f}%.")
                print(f"     'Diem thap hon' chi la khac biet THANG DO giua cac khong gian, khong phai chat luong.")

    # === CHAN DOAN NHIEU NHAN / TRUNG DANH TINH =========================================
    # Nghi pham hang dau con lai (§11.3): vai cap "impostor" thuc chat la CUNG MOT VAT -> day
    # phan vi 99.9% len cao -> pha moi nguong tuyet doi. z-norm KHONG sua duoc. Xem tan mat.
    diag = None
    if len(results) > 0:
        diag_space = primary if primary in results else next(iter(results))
        diag_thr = float(results[diag_space]['threshold'])
        diag = diagnose_top_impostors(
            feats_eval[diag_space][0], feats_eval[diag_space][1], pids_eval,
            ds_eval.valid_pairs, space=diag_space, top_k=50,
            threshold=diag_thr, data_dir=data_dir)
        print(f"\n  === CHAN DOAN NHIEU NHAN / TRUNG DANH TINH (eval, space={diag_space}, "
              f"nguong t*={diag_thr:.4f}) ===")
        print(f"  Tong cap impostor: {diag['n_impostor_pairs']:,} | "
              f"vuot nguong: {diag.get('n_impostor_above_threshold', 0):,} "
              f"(FAR={diag.get('far_above_threshold', 0)*100:.4f}%)")
        print(f"  Top {min(10, len(diag['top_pairs']))} cap impostor diem cao nhat "
              f"([VUOT] = >= t*):")
        for p in diag['top_pairs'][:10]:
            mark = "[VUOT]" if p.get('above_threshold') else "      "
            print(f"   {mark} {p['score']:.4f}  pid {p['q_identity_id']} -> {p['g_identity_id']}  "
                  f"seq {p['q_sequence_id']} vs {p['g_sequence_id']}  "
                  f"same_seq={p['same_sequence']}  ev {p['q_event_index']}/{p['g_event_index']}")
            print(f"            q: {p['q_image_paths']}")
            print(f"            g: {p['g_image_paths']}")
        print(f"  Cac CAP DANH TINH de nham nhat (sim trung binh):")
        for p in diag['top_pid_pairs'][:10]:
            print(f"    pid {p['pid_a']} <-> {p['pid_b']}: n={p['n_pairs']:>5}  "
                  f"mean={p['mean_sim']:.4f}  max={p['max_sim']:.4f}")
        conc = diag.get('concentration') or {}
        if conc:
            print(f"  Do TAP TRUNG cua khoi luong vuot nguong:")
            print(f"    {conc.get('n_distinct_pid_pairs_above')} cap pid / "
                  f"{conc.get('n_distinct_pid_pairs_total')} tong so cap (KHONG thu tu) "
                  f"chiem toan bo khoi vuot nguong")
            for s2 in conc.get('top_pid_pairs_share', [])[:5]:
                print(f"      pid {s2['pid_a']}<->{s2['pid_b']}: n={s2['n']} "
                      f"share={s2['share']*100:.1f}%")
            print(f"    Trong khoi vuot nguong: CUNG chuoi="
                  f"{(conc.get('frac_above_same_sequence') or 0)*100:.1f}%  "
                  f"KHAC chuoi={(conc.get('frac_above_cross_sequence') or 0)*100:.1f}%")
        print(f"  -> DIEN GIAI: mean_sim cua mot cap pid ~0.9 (bang muc genuine) => NHIEU NHAN.")
        print(f"     CUNG chuoi = hard negative hop le (khong phai loi nhan); KHAC chuoi + anh")
        print(f"     giong het = nghi trung danh tinh. Hay MO ANH o cac duong dan tren de xac nhan.")
        print(f"     (Khong ket luan thay: neu cac cap tren la drone KHAC NHAU that thi gia thuyet")
        print(f"      nhieu nhan bi loai, va van de nam o chat luong embedding/du lieu.)")

    # === TAI NGUONG DANG TRIEN KHAI =====================================================
    # Cau hoi: pipeline dung reid_threshold (vd 0.75) va chi HARD LOCK ~6% cua so. Offline tai
    # DUNG nguong do thi TAR la bao nhieu? Neu offline cao hon nhieu -> cua so T2_SEARCH kho hon
    # du lieu query/gallery (occlusion, crop loi), chu khong phai loi nguong.
    deployed_thr = float((cfg.get('infer') or {}).get('reid_threshold', 0.75))
    deployed = {}
    print(f"\n  === TAI NGUONG DANG TRIEN KHAI cua infer.py (reid_threshold = {deployed_thr}) ===")
    for s in results:
        tar_d, far_d, _ = eval_at_threshold(scores[s]['gen_eval'], scores[s]['imp_eval'], deployed_thr)
        deployed[s] = {'threshold': deployed_thr, 'tar_eval': float(tar_d), 'far_eval': float(far_d)}
        print(f"    [{s}] TAR(eval)={tar_d*100:>6.2f}%   FAR(eval)={far_d*100:>8.4f}%")
    print(f"    -> So TAR nay voi ti le HARD LOCK thuc te cua pipeline (md/15thg9.md §1.2: <=6.08%).")
    print(f"       Neu offline cao hon nhieu -> cua so T2_SEARCH kho hon du lieu query/gallery.")

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
        # 🛠️ (14/9): thử nghiệm z-norm theo từng truy vấn (upper bound của "đổi luật quyết định")
        'znorm': znorm,
        # 🛠️ (14/9): TAR/FAR offline tại đúng ngưỡng pipeline đang dùng (cầu nối offline <-> online)
        'deployed_threshold': deployed,
        # 🛠️ (14/9): chẩn đoán nhiễu nhãn / trùng danh tính (top cặp impostor + độ tập trung)
        'label_noise_diagnosis': diag,
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
