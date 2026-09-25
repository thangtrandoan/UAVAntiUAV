import os
import sys
import json
import yaml
import argparse
import time
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw
import numpy as np

# model import moved to main()
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return getattr(self.terminal, 'isatty', lambda: False)()

class EvalDataset(Dataset):
    def __init__(self, data_dir, query_json, gallery_json, transform=None, num_frames=16):
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
                        'attributes': q.get('attributes', []),
                        'sequence_id': q['sequence_id']
                    })

    def __len__(self):
        return len(self.valid_pairs)
        
    def _load_clip(self, folder, frames, take_last=False):
        # 🛠️ (15/9) ĐỒNG BỘ frame_stride — BỎ `np.linspace` (xem train_reid.py::_load_clip).
        # `np.linspace` làm bước thời gian hiệu dụng > frame_stride → offline eval lệch
        # với phân phối temporal lúc train và lúc infer.
        # 🛠️ (22/9) COPY list — cùng bug như `UAVReIDDataset._load_clip`: dòng
        # `frames.append(...)` bên dưới làm list TRONG `valid_pairs` phình VĨNH VIỄN.
        # Hiện chưa gây hại vì mỗi N dùng một `EvalDataset` riêng (đọc lại JSON), nhưng
        # nếu dùng lại MỘT dataset cho nhiều N thì đây là lỗi thật (pad tới N nhỏ rồi đọc
        # N lớn -> toàn frame lặp). Xem md/22thg9.md §29.
        frames = list(frames)
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
            except:
                img = Image.new('RGB', (256, 256), (0,0,0))
            if self.transform:
                img = self.transform(img)
            clip.append(img)
        return torch.stack(clip, dim=0)

    def __getitem__(self, idx):
        pair = self.valid_pairs[idx]
        
        before_frames = pair['gallery_frames']
        after_frames = pair['query_frames']
        
        vis_path_b = os.path.join(self.data_dir, pair['gallery_dir'], before_frames[len(before_frames)//2]) if before_frames else ""
        vis_path_a = os.path.join(self.data_dir, pair['query_dir'], after_frames[len(after_frames)//2]) if after_frames else ""
        
        before_clip = self._load_clip(pair['gallery_dir'], before_frames, take_last=True)
        after_clip = self._load_clip(pair['query_dir'], after_frames, take_last=False)
        
        pid = pair['identity_id']
        attrs = pair.get('attributes', [])
        seq_id = pair['sequence_id']
        return before_clip, after_clip, pid, str(attrs), vis_path_a, vis_path_b, seq_id

def draw_border(image, color, width=5):
    draw = ImageDraw.Draw(image)
    draw.rectangle([(0, 0), (image.size[0]-1, image.size[1]-1)], outline=color, width=width)
    return image

def eval_map_cmc(qf, gf, q_pids, g_pids, q_seq_ids=None, g_seq_ids=None, intra_sequence=False):
    qf = F.normalize(qf, p=2, dim=1)
    gf = F.normalize(gf, p=2, dim=1)
    
    distmat = 1 - torch.mm(qf, gf.t())
    distmat = distmat.cpu().numpy()
    
    if intra_sequence and q_seq_ids is not None and g_seq_ids is not None:
        q_seqs = np.asarray(q_seq_ids)
        g_seqs = np.asarray(g_seq_ids)
        mask = (q_seqs[:, np.newaxis] != g_seqs[np.newaxis, :])
        distmat[mask] = np.inf
        
    q_pids = np.asarray(q_pids)
    g_pids = np.asarray(g_pids)
    
    num_q, num_g = distmat.shape
    indices = np.argsort(distmat, axis=1)
    
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0
    
    for q_idx in range(num_q):
        orig_cmc = matches[q_idx]
        if not np.any(orig_cmc):
            continue
            
        cmc = orig_cmc
        num_valid_q += 1.
        
        pos_indices = np.where(orig_cmc == 1)[0]
        max_pos_idx = np.max(pos_indices)
        inp = orig_cmc.sum() / (max_pos_idx + 1.0)
        all_INP.append(inp)
        
        cmc = np.cumsum(cmc)
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:50])
        
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
        
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)
    
    return all_cmc, mAP, mINP, indices, matches

def compute_tar_at_far(qf, gf, q_pids, g_pids, far_target=0.001, threshold=None):
    """
    Tính TAR@FAR=far_target dùng pairwise cosine similarity.
    - Genuine pairs : (i, j) có q_pids[i] == g_pids[j]  (trừ cặp i==j)
    - Impostor pairs: (i, j) có q_pids[i] != g_pids[j]

    threshold=None  -> in-sample: tự tìm t* = quantile(impostor, 1-far_target) từ chính
                       dữ liệu này (THIÊN LỆCH, nhưng dùng để so sánh CÔNG BẰNG giữa các
                       không gian đặc trưng vì cùng một quy trình).
    threshold=<float> -> dùng threshold ngoài (đã calibrate trên cal-split → không thiên lệch).

    Trả về (tar, far, threshold_đã_dùng).
    """
    qf_n = F.normalize(qf, p=2, dim=1).cpu().numpy()
    gf_n = F.normalize(gf, p=2, dim=1).cpu().numpy()
    sim = np.dot(qf_n, gf_n.T)          # (num_q, num_g)

    q_pids = np.asarray(q_pids)
    g_pids = np.asarray(g_pids)

    num_q, num_g = sim.shape
    genuine_scores  = []
    impostor_scores = []

    for i in range(num_q):
        for j in range(num_g):
            if i == j:      # loại self-match (query và gallery là cùng clip)
                continue
            if q_pids[i] == g_pids[j]:
                genuine_scores.append(sim[i, j])
            else:
                impostor_scores.append(sim[i, j])

    genuine_scores  = np.array(genuine_scores,  dtype=np.float32)
    impostor_scores = np.array(impostor_scores, dtype=np.float32)

    if len(impostor_scores) == 0 or len(genuine_scores) == 0:
        return float('nan'), float('nan'), float('nan')

    # Tìm threshold sao cho FAR <= far_target
    # FAR(t) = P(impostor > t)  =>  sắp xếp giảm dần, lấy phần vị (1 - far_target)
    t_star = float(threshold) if threshold is not None \
        else float(np.quantile(impostor_scores, 1.0 - far_target))

    tar = float(np.mean(genuine_scores >= t_star))
    far = float(np.mean(impostor_scores >= t_star))
    return tar, far, t_star

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json", type=str, help="Path to config file")
    parser.add_argument("--threshold-file", default=None, type=str,
                        help="Path to calibrated_threshold.json (từ calibrate_threshold.py). "
                             "Nếu không cung cấp, threshold sẽ được tính từ chính test set (in-sample, thiên lệch).")
    parser.add_argument("--space", default=None, type=str, choices=["fused", "pre_bn"],
                        help="Không gian feature dùng làm CHÍNH (áp calibrated threshold + visualization). "
                             "fused = qua ReIDHead (BatchNorm1d) — mặc định; pre_bn = cat(visual, temporal), "
                             "ĐẦU VÀO bnneck. Cả hai không gian luôn được đánh giá để so sánh (xem md/15thg9.md).")
    # 🛠️ (25/9) Cho phép đặt threshold THỦ CÔNG ngay trên dòng lệnh.
    parser.add_argument("--threshold", default=None, type=float,
                        help="Đặt threshold THỦ CÔNG (ưu tiên cao nhất, ghi đè --threshold-file "
                             "và config). Dùng để thử nhanh một ngưỡng cụ thể, vd: --threshold 0.85")
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    ec = cfg.get('eval', {})
    args.model_path    = ec.get('model_path', 'checkpoints/best_model.pth')
    args.data_dir      = cfg.get('paths', {}).get('data_dir', './processed')
    args.query_json    = ec.get('query_json', './processed/query_test.json')
    args.gallery_json  = ec.get('gallery_json', './processed/gallery_test.json')
    args.output_dir    = ec.get('output_dir', 'eval_results')
    args.batch_size    = ec.get('batch_size', 32)
    args.num_workers   = ec.get('num_workers', 4)
    args.backbone_only = ec.get('backbone_only', False)
    args.intra_sequence = ec.get('intra_sequence', False)
    args.max_correct_vis = ec.get('max_correct_vis', -1)
    args.num_frames    = cfg.get('train', {}).get('num_frames', 16)
    args.backbone      = ec.get('backbone', 'resnet50_ibn')
    args.gpu_jetson    = cfg.get('device', {}).get('gpu_jetson', False)
    # threshold_file: CLI arg override config nếu được truyền vào
    if args.threshold_file is None:
        args.threshold_file = ec.get('threshold_file', None)
    # space: CLI arg override config; mặc định 'fused' (giữ nguyên hành vi cũ)
    args.space = args.space or ec.get('space', 'fused')


    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logger
    log_path = os.path.join(args.output_dir, f"eval_{time.strftime('%Y%m%d_%H%M%S')}.log")
    sys.stdout = Logger(log_path)
    
    print(f"=== Bắt đầu đánh giá lúc {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    print(f"Tham số chạy:\n{json.dumps(vars(args), indent=4)}")
    
    transform_test = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dir = os.path.join(args.data_dir, "test")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(
            f"Không tìm thấy test dir tại {test_dir}. "
            "KHÔNG được fallback về train dir để eval (sẽ cho metric ảo cao). "
            "Chạy data_pipeline trước để tạo test set + query_test.json/gallery_test.json."
        )
        
    dataset = EvalDataset(test_dir, args.query_json, args.gallery_json, transform=transform_test, num_frames=args.num_frames)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # --- Setup GASNet Path ---
    gasnet_dir = cfg.get('paths', {}).get('gasnet_dir', '')
    if gasnet_dir:
        os.environ['GASNET_PATH'] = os.path.abspath(gasnet_dir)
        
    from model import UAVReIDNet, load_checkpoint_verbose
    # 🛠️ (24/9) `temporal_pool`/`temporal_pe` phải khớp lúc TRAIN (chi tiết: calibrate_threshold.py).
    temporal_type = cfg.get('eval', {}).get('temporal_type', cfg.get('train', {}).get('temporal_type', 'mamba'))
    temporal_pool = cfg.get('eval', {}).get('temporal_pool', cfg.get('train', {}).get('temporal_pool', 'attn'))
    temporal_pe = bool(cfg.get('eval', {}).get('temporal_pe', cfg.get('train', {}).get('temporal_pe', True)))
    print(f"Temporal encoder: type={temporal_type}, pool={temporal_pool}, pe={temporal_pe}")
    model = UAVReIDNet(freeze_backbone=False, backbone=args.backbone, temporal_type=temporal_type,
                       temporal_pool=temporal_pool, temporal_pe=temporal_pe)
    if not args.backbone_only:
        if os.path.exists(args.model_path):
            # 🛠️ (14/9): báo cáo đầy đủ missing/unexpected/shape-mismatch thay vì "Loaded" mù quáng.
            load_checkpoint_verbose(model, args.model_path, tag="eval")
            print(f"Loaded {args.model_path}")
        else:
            print(f"Warning: {args.model_path} not found! Mamba head has random weights. (Use --backbone-only to evaluate pure GASNet)")
    else:
        print("INFO: Evaluating BASELINE GASNet only (Mamba head bypassed).")
    model.cuda()
    model.eval()
    
    # 🛠️ (14/9): trích đồng thời 2 không gian trong MỘT lượt backbone:
    #   fused  = qua ReIDHead (BatchNorm1d) — pipeline hiện tại (fine score)
    #   pre_bn = cat(visual, temporal)      — ĐẦU VÀO bnneck (raw)
    # Cả hai được đánh giá bằng CÙNG một quy trình để trả lời: BatchNorm1d có thật sự
    # làm mất khả năng phân biệt, hay chỉ nén thang điểm? (xem md/15thg9.md)
    if args.backbone_only:
        spaces = ['backbone']
    elif args.space == 'pre_bn':
        spaces = ['pre_bn', 'fused']
    else:
        spaces = ['fused', 'pre_bn']
    threshold_space = args.space if args.space in spaces else spaces[0]

    feats = {s: {'qf': [], 'gf': []} for s in spaces}
    q_pids, g_pids = [], []
    q_seq_ids, g_seq_ids = [], []
    attributes_list = []
    vis_paths_q, vis_paths_g = [], []
    
    print(f"Extracting features... (spaces: {', '.join(spaces)})")
    start_time = time.time()
    with torch.no_grad():
        for i, (before, after, pids, attrs, v_q, v_g, seq_ids) in enumerate(dataloader):
            before, after = before.cuda(), after.cuda()
            if args.backbone_only:
                feats['backbone']['gf'].append(model(before, backbone_only=True))
                feats['backbone']['qf'].append(model(after, backbone_only=True))
            else:
                v_g_t, t_g, _ = model.extract_features(before)
                v_q_t, t_q, _ = model.extract_features(after)
                if 'fused' in spaces:
                    feats['fused']['gf'].append(model.head(v_g_t, t_g))
                    feats['fused']['qf'].append(model.head(v_q_t, t_q))
                if 'pre_bn' in spaces:
                    feats['pre_bn']['gf'].append(torch.cat([v_g_t, t_g], dim=-1))
                    feats['pre_bn']['qf'].append(torch.cat([v_q_t, t_q], dim=-1))
            
            g_pids.extend(pids.numpy())
            q_pids.extend(pids.numpy())
            q_seq_ids.extend(seq_ids)
            g_seq_ids.extend(seq_ids)
            attributes_list.extend(attrs)
            vis_paths_q.extend(v_q)
            vis_paths_g.extend(v_g)
            
            if (i + 1) % 10 == 0 or (i + 1) == len(dataloader):
                elapsed = time.time() - start_time
                print(f"  -> Đã trích xuất {i + 1}/{len(dataloader)} batches (Mất {elapsed:.2f}s)")
                start_time = time.time()
            
    for s in spaces:
        feats[s]['qf'] = torch.cat(feats[s]['qf'], dim=0)
        feats[s]['gf'] = torch.cat(feats[s]['gf'], dim=0)

    # --- Load calibrated threshold (nếu có) ---
    threshold_source = "in-sample (thiên lệch)"
    calibrated_threshold = None

    if args.threshold is not None:
        # 🛠️ (25/9) Ưu tiên CAO NHẤT: cho phép thử một ngưỡng cụ thể mà KHÔNG cần tạo file JSON
        # và KHÔNG cần chạy calibrate_threshold.py. Một giá trị áp cho không gian đang chọn.
        calibrated_threshold = float(args.threshold)
        threshold_source = f"THỦ CÔNG (--threshold={calibrated_threshold:.6f}, KHÔNG calibrate)"
        print(f"  Dùng threshold THỦ CÔNG ({threshold_space}): {calibrated_threshold:.6f}  "
              f"[{threshold_source}]")
    elif args.threshold_file and os.path.exists(args.threshold_file):
        with open(args.threshold_file, 'r') as f:
            calib_data = json.load(f)
        thresholds_by_space = calib_data.get('thresholds', {}) or {}
        # Ưu tiên threshold calibrate riêng cho không gian được chọn (--space)
        calibrated_threshold = thresholds_by_space.get(threshold_space, calib_data.get('threshold'))
        threshold_source = (
            f"calibrated[{threshold_space}] (Fixed FAR<={calib_data.get('far_target',0.001)*100:.2f}%, "
            f"cal_ratio={calib_data.get('cal_ratio','?')}, seed={calib_data.get('seed','?')})"
        )
        print(f"  Dùng calibrated threshold ({threshold_space}): {calibrated_threshold:.6f}  [{threshold_source}]")
    elif args.threshold_file:
        print(f"  WARNING: --threshold-file '{args.threshold_file}' không tìm thấy!")
        print("           Fallback về in-sample threshold (thiên lệch).")

    # --- Tính metric cho TỪNG không gian bằng CÙNG một quy trình ---
    print("Computing metrics...")
    results = {}
    artifacts = {}
    for s in spaces:
        qf_s, gf_s = feats[s]['qf'], feats[s]['gf']
        cmc, mAP, mINP, indices, matches = eval_map_cmc(
            qf_s, gf_s, q_pids, g_pids, q_seq_ids, g_seq_ids, args.intra_sequence)
        tar_is, far_is, thr_is = compute_tar_at_far(qf_s, gf_s, q_pids, g_pids,
                                                    far_target=0.001, threshold=None)
        results[s] = {
            'rank1': float(cmc[0]), 'rank5': float(cmc[4]),
            'mAP': float(mAP), 'mINP': float(mINP),
            'tar_at_far_0.1_in_sample': float(tar_is),
            'far_in_sample': float(far_is),
            'threshold_in_sample': float(thr_is),
            'tar_at_far_0.1_calibrated': None,
            'far_calibrated': None,
            'threshold_calibrated': None,
        }
        if s == threshold_space and calibrated_threshold is not None:
            tar_c, far_c, thr_c = compute_tar_at_far(qf_s, gf_s, q_pids, g_pids,
                                                     far_target=0.001, threshold=calibrated_threshold)
            results[s].update({
                'tar_at_far_0.1_calibrated': float(tar_c),
                'far_calibrated': float(far_c),
                'threshold_calibrated': float(thr_c),
            })
        artifacts[s] = (cmc, mAP, mINP, indices, matches)

    primary_space = args.space if args.space in results else spaces[0]
    cmc, mAP, mINP, indices, matches = artifacts[primary_space]
    qf, gf = feats[primary_space]['qf'], feats[primary_space]['gf']

    print("\n=== OFFLINE REID EVALUATION (STATIC PROTOCOL) ===")
    print(f"Không gian chính : {primary_space}   |   So sánh: {', '.join(results.keys())}")
    print(f"  {'Space':<9} {'Rank-1':>8} {'Rank-5':>8} {'mAP':>8} {'mINP':>8} "
          f"{'TAR@FAR=0.1%':>14} {'thr(in-sample)':>15}")
    for s, r in results.items():
        print(f"  {s:<9} {r['rank1']*100:>7.2f}% {r['rank5']*100:>7.2f}% "
              f"{r['mAP']*100:>7.2f}% {r['mINP']*100:>7.2f}% "
              f"{r['tar_at_far_0.1_in_sample']*100:>13.2f}% {r['threshold_in_sample']:>15.6f}")
    if 'fused' in results and 'pre_bn' in results:
        d_tar = results['pre_bn']['tar_at_far_0.1_in_sample'] - results['fused']['tar_at_far_0.1_in_sample']
        d_rank1 = results['pre_bn']['rank1'] - results['fused']['rank1']
        print(f"\n  Delta(pre_bn - fused): Rank-1 {d_rank1*100:+.2f}% | TAR@FAR=0.1% {d_tar*100:+.2f}%"
              f"   (cùng quy trình in-sample -> so sánh được)")
        if d_tar > 0.02:
            print("  -> pre_bn TỐT HƠN rõ rệt: BatchNorm1d thật sự làm mất khả năng phân biệt.")
        elif d_tar < -0.02:
            print("  -> pre_bn KÉM HƠN: BN chỉ nén thang điểm, KHÔNG làm mất thông tin.")
        else:
            print("  -> pre_bn ~ fused: BN vô hại về mặt phân biệt; chỉ cần calibrate lại threshold.")

    primary = results[primary_space]
    tar_01 = primary['tar_at_far_0.1_calibrated']
    actual_far = primary['far_calibrated']
    t_star = primary['threshold_calibrated']
    if tar_01 is None:
        tar_01 = primary['tar_at_far_0.1_in_sample']
        actual_far = primary['far_in_sample']
        t_star = primary['threshold_in_sample']

    print(f"\nRank-1 Accuracy  : {primary['rank1']*100:.2f}%")
    print(f"Rank-5 Accuracy  : {primary['rank5']*100:.2f}%")
    print(f"mAP              : {primary['mAP']*100:.2f}%")
    print(f"mINP             : {primary['mINP']*100:.2f}%")
    if not np.isnan(tar_01):
        print(f"TAR@FAR=0.1%     : {tar_01*100:.2f}%  (actual FAR={actual_far*100:.4f}%)")
        print(f"  [threshold={t_star:.6f}, source={threshold_source}]")
    else:
        print("TAR@FAR=0.1%     : N/A (không đủ genuine/impostor pairs)")

    # Save Report (flat keys = không gian chính -> tương thích ngược; 'spaces' = bảng so sánh)
    report = {
        "Rank-1": float(primary['rank1']),
        "Rank-5": float(primary['rank5']),
        "mAP": float(primary['mAP']),
        "mINP": float(primary['mINP']),
        "TAR@FAR=0.1%": float(tar_01) if not np.isnan(tar_01) else None,
        "actual_FAR": float(actual_far) if not np.isnan(tar_01) else None,
        "threshold": float(t_star) if not np.isnan(tar_01) else None,
        "threshold_source": threshold_source,
        "primary_space": primary_space,
        "spaces": results,
    }
    with open(os.path.join(args.output_dir, "evaluation_report.json"), "w") as f:
        json.dump(report, f, indent=4)

        
    # Pre-calculate similarity matrix for visualization and online eval
    qf_norm = F.normalize(qf, p=2, dim=1)
    gf_norm = F.normalize(gf, p=2, dim=1)
    sim_matrix = torch.mm(qf_norm, gf_norm.t()).cpu().numpy()

    # Visualization for all evaluation cases
    if args.max_correct_vis == 0:
        print("\nSkipping visualization (max_correct_vis = 0)...")
        # Save empty eval_info just in case other scripts depend on the file
        with open(os.path.join(args.output_dir, "eval_cases_info.json"), "w") as f:
            json.dump({}, f)
        return

    print("\nGenerating visualization for all evaluated cases (Contact Sheets)...")
    from PIL import ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except IOError:
        font = ImageFont.load_default()
        
    eval_info = {}
    saved_correct_count = 0
    for q_idx in range(len(q_pids)):
        if matches[q_idx].sum() > 0:
            is_correct = matches[q_idx][0]
            if is_correct and args.max_correct_vis >= 0 and saved_correct_count >= args.max_correct_vis:
                continue
            if is_correct:
                saved_correct_count += 1
                
            q_img_path = vis_paths_q[q_idx]
            if not os.path.exists(q_img_path): continue
            
            q_img = Image.open(q_img_path).resize((128, 128))
            q_img = draw_border(q_img, "black")
            
            sheet = Image.new('RGB', (128 * 6 + 20, 128), (255, 255, 255))
            sheet.paste(q_img, (0, 0))
            
            top5_idx = indices[q_idx][:5]
            top5_paths = []
            for k, g_idx in enumerate(top5_idx):
                if args.intra_sequence and q_seq_ids[q_idx] != g_seq_ids[g_idx]:
                    continue
                    
                g_img_path = vis_paths_g[g_idx]
                top5_paths.append(g_img_path)
                if os.path.exists(g_img_path):
                    g_img = Image.open(g_img_path).resize((128, 128))
                    color = "green" if g_pids[g_idx] == q_pids[q_idx] else "red"
                    g_img = draw_border(g_img, color, width=5)
                    
                    # Draw confidence score
                    conf_score = sim_matrix[q_idx][g_idx]
                    draw = ImageDraw.Draw(g_img)
                    text = f"{conf_score:.2f}"
                    
                    if hasattr(draw, 'textbbox'):
                        bbox = draw.textbbox((0, 0), text, font=font)
                        text_w = bbox[2] - bbox[0]
                        text_h = bbox[3] - bbox[1]
                    else:
                        text_w, text_h = draw.textsize(text, font=font)
                        
                    draw.rectangle([(0, 0), (text_w + 4, text_h + 4)], fill="black")
                    draw.text((2, 2), text, fill="white", font=font)
                    
                    sheet.paste(g_img, (128 * (k+1) + 20, 0))
                    
            if args.intra_sequence:
                seq_id = q_seq_ids[q_idx]
                seq_out_dir = os.path.join(args.output_dir, seq_id)
                os.makedirs(seq_out_dir, exist_ok=True)
                
                # Highlight if correct or error in filename
                prefix = "correct" if is_correct else "error"
                result_filename = f"{prefix}_q{q_idx}.jpg"
                sheet.save(os.path.join(seq_out_dir, result_filename))
                
                if seq_id not in eval_info:
                    eval_info[seq_id] = {}
                    
                eval_info[seq_id][result_filename] = {
                    "query_path": q_img_path,
                    "top5_gallery_paths": top5_paths,
                    "is_rank1_correct": bool(is_correct)
                }
            else:
                prefix = "correct" if is_correct else "error"
                result_filename = f"{prefix}_q{q_idx}.jpg"
                sheet.save(os.path.join(args.output_dir, result_filename))
                eval_info[result_filename] = {
                    "query_path": q_img_path,
                    "top5_gallery_paths": top5_paths,
                    "is_rank1_correct": bool(is_correct)
                }
            
    if args.intra_sequence:
        for seq_id, info in eval_info.items():
            with open(os.path.join(args.output_dir, seq_id, "eval_cases_info.json"), "w") as f:
                json.dump(info, f, indent=4)
    else:
        with open(os.path.join(args.output_dir, "eval_cases_info.json"), "w") as f:
            json.dump(eval_info, f, indent=4)

    # (Đã xóa khối "ONLINE SEQUENTIAL EVALUATION" — trước đây bịa latency bằng
    #  np.random.randint(1,5), không đo được gì thật. Latency theo frame chỉ có
    #  ở infer.py qua SeqReIDPipeline.reid_latency_frames.)

if __name__ == '__main__':
    main()
